"""
[EXPERIMENTAL API — outside the semver contract until stabilized; see README
"API stability and versioning".]

Differentiable thermodynamic engine using PyTorch.

This allows optimizing the nearest neighbor parameters using backpropagation.
Computes the exact Partition Function using a batched DP algorithm.
"""

import math
import torch
import torch.nn as nn

from strider.thermo.parameters_rna import (
    TERMINAL_MISMATCH as RNA_TERM_MISMATCH,
    HAIRPIN_SIZE as RNA_HAIRPIN_SIZE,
    INTERIOR_SIZE as RNA_INTERIOR_SIZE,
    BULGE_SIZE as RNA_BULGE_SIZE,
    ASYMMETRY_NINIO as RNA_ASYMMETRY_NINIO,
    DANGLE_3 as RNA_DANGLE_3,
    DANGLE_5 as RNA_DANGLE_5,
    ML_BASE as RNA_ML_BASE,
    ML_INIT as RNA_ML_INIT,
    ML_PAIR as RNA_ML_PAIR
)

INF = 1e9  # Use a large number instead of float('inf') to avoid NaNs in gradients
R = 1.987e-3  # kcal / (mol · K)
T = 310.15 # 37 C
RT = R * T
MIN_HAIRPIN_LOOP = 3

def _can_pair(a: str, b: str, material: str) -> bool:
    if material == "rna":
        wc = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
    else:
        wc = {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")}
    return (a, b) in wc

def logsumexp_RT(energies: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Computes -RT * ln(sum(exp(-E/RT))) for a list of energy tensors.
    """
    if not energies:
        raise ValueError("energies list must not be empty")
    if len(energies) == 1:
        return energies[0]
    
    scaled = [-e / RT for e in energies]
    stacked = torch.stack(scaled, dim=dim)
    return -RT * torch.logsumexp(stacked, dim=dim)

class ThermoParameters(nn.Module):
    def __init__(self, material="rna"):
        super().__init__()
        self.material = material
        
        # Load material-specific parameters
        if material == "rna":
            from strider.thermo.parameters_rna import (
                TERMINAL_MISMATCH as P_TERM_MISMATCH,
                HAIRPIN_SIZE as P_HAIRPIN_SIZE,
                INTERIOR_SIZE as P_INTERIOR_SIZE,
                BULGE_SIZE as P_BULGE_SIZE,
                ASYMMETRY_NINIO as P_ASYMMETRY_NINIO,
                DANGLE_3 as P_DANGLE_3,
                DANGLE_5 as P_DANGLE_5,
                ML_BASE as P_ML_BASE,
                ML_INIT as P_ML_INIT,
                ML_PAIR as P_ML_PAIR,
                TERMINAL_PENALTY as P_TERMINAL_PENALTY,
                HAIRPIN_TRILOOP as P_TRILOOP,
                HAIRPIN_TETRALOOP as P_TETRALOOP,
                STACK as P_STACK
            )
            default_stack_val = -2.0
        else:
            from strider.thermo.parameters_dna import (
                TERMINAL_MISMATCH as P_TERM_MISMATCH,
                HAIRPIN_SIZE as P_HAIRPIN_SIZE,
                INTERIOR_SIZE as P_INTERIOR_SIZE,
                BULGE_SIZE as P_BULGE_SIZE,
                ASYMMETRY_NINIO as P_ASYMMETRY_NINIO,
                DANGLE_3 as P_DANGLE_3,
                DANGLE_5 as P_DANGLE_5,
                ML_BASE as P_ML_BASE,
                ML_INIT as P_ML_INIT,
                ML_PAIR as P_ML_PAIR,
                TERMINAL_PENALTY as P_TERMINAL_PENALTY,
                HAIRPIN_TRILOOP as P_TRILOOP,
                HAIRPIN_TETRALOOP as P_TETRALOOP,
                STACK as P_STACK
            )
            default_stack_val = -1.5
            
        self.terminal_penalty_dict = P_TERMINAL_PENALTY
        self.triloop_dict = P_TRILOOP
        self.tetraloop_dict = P_TETRALOOP

        # 4×4 LUT for terminal-pair (AU/GU/AT) penalty, indexed by base
        # integer codes (A=0, C=1, G=2, T/U=3).  Same values as
        # P_TERMINAL_PENALTY but reshaped for tensorised lookup in the DP.
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
        tp_lut = torch.zeros(4, 4, dtype=torch.float64)
        for k, v in P_TERMINAL_PENALTY.items():
            tp_lut[base_map[k[0]], base_map[k[1]]] = v
        self.terminal_penalty_lut = nn.Parameter(tp_lut)

        # 1. Stacking parameters — the FULL nearest-neighbor stack table indexed
        #    by all four bases of the stacked step (5'-(i)(i+1)-3' / 3'-(j)(j-1)-5'),
        #    flat-indexed b_i·64 + b_{i+1}·16 + b_{j-1}·4 + b_j (A=0,C=1,G=2,T/U=3).
        #    This is the 36-entry P_STACK table — covering GU-wobble stacks, not
        #    just the 16 Watson-Crick dinucleotides — so wobble-containing helices
        #    are no longer mis-scored with WC stack energies.  Keys are T-form
        #    (U maps to 3, matching the model's pair-set convention in ensemble.py).
        base_map4 = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
        stack_tbl = torch.full((256,), float(default_stack_val), dtype=torch.float64)
        for k, v in P_STACK.items():
            si = base_map4[k[0]] * 64 + base_map4[k[1]] * 16 + base_map4[k[2]] * 4 + base_map4[k[3]]
            stack_tbl[si] = v
        self.stack_table = nn.Parameter(stack_tbl)

        # 2. Hairpin Size Penalties
        self.hairpin_sizes = nn.Parameter(torch.tensor(P_HAIRPIN_SIZE, dtype=torch.float64))
        
        # 3. Terminal Mismatch
        self.term_mismatch = nn.Parameter(torch.zeros(256, dtype=torch.float64))
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
        for k, v in P_TERM_MISMATCH.items():
            idx = base_map[k[0]]*64 + base_map[k[1]]*16 + base_map[k[2]]*4 + base_map[k[3]]
            self.term_mismatch.data[idx] = v
        self.default_term = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        
        # 4. Multiloop penalties
        self.ml_base_raw = nn.Parameter(torch.log(torch.tensor(max(P_ML_BASE, 1e-3), dtype=torch.float64)))
        self.ml_init_raw = nn.Parameter(torch.log(torch.tensor(max(P_ML_INIT, 1e-3), dtype=torch.float64)))
        self.ml_pair_raw = nn.Parameter(torch.log(torch.tensor(max(P_ML_PAIR, 1e-3), dtype=torch.float64)))

        # 5. Interior Loops & Bulges
        self.interior_sizes = nn.Parameter(torch.tensor(P_INTERIOR_SIZE, dtype=torch.float64))
        self.bulge_sizes_raw = nn.Parameter(torch.log(torch.clamp(torch.tensor(P_BULGE_SIZE, dtype=torch.float64), min=1e-3)))
        self.asymmetry_ninio = nn.Parameter(torch.tensor(P_ASYMMETRY_NINIO, dtype=torch.float64))

        # 6. Neural DP for Interior Loops
        self.mlp_1_1 = nn.Sequential(
            nn.Linear(6 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(torch.float64)
        self.mlp_1_2 = nn.Sequential(
            nn.Linear(7 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(torch.float64)
        self.mlp_2_2 = nn.Sequential(
            nn.Linear(8 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(torch.float64)

        # 7. Dangles
        self.dangle_3 = nn.Parameter(torch.zeros(64, dtype=torch.float64))
        self.dangle_5 = nn.Parameter(torch.zeros(64, dtype=torch.float64))
        for k, v in P_DANGLE_3.items():
            if len(k) == 3:
                idx = base_map[k[0]]*16 + base_map[k[1]]*4 + base_map[k[2]]
                self.dangle_3.data[idx] = v
        for k, v in P_DANGLE_5.items():
            if len(k) == 3:
                idx = base_map[k[0]]*16 + base_map[k[1]]*4 + base_map[k[2]]
                self.dangle_5.data[idx] = v
                
        # Zero-initialize the final layer of each MLP so they start with 0.0 impact
        for mlp in [self.mlp_1_1, self.mlp_1_2, self.mlp_2_2]:
            last_layer = mlp[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                if last_layer.bias is not None:
                    nn.init.zeros_(last_layer.bias)

    def get_hairpin_size_energy(self, loop_len: int) -> torch.Tensor:
        if loop_len < len(self.hairpin_sizes):
            return self.hairpin_sizes[loop_len]
        return self.hairpin_sizes[-1] + 1.75 * RT * math.log(loop_len / float(len(self.hairpin_sizes) - 1))

    def get_interior_size_energy(self, loop_len_tensor: torch.Tensor) -> torch.Tensor:
        idx = loop_len_tensor - 1
        safe_idx = torch.clamp(idx, max=len(self.interior_sizes) - 1)
        base_e = self.interior_sizes[safe_idx]
        extrap_e = self.interior_sizes[-1] + 1.75 * RT * torch.log(loop_len_tensor.float() / float(len(self.interior_sizes)))
        return torch.where(idx < len(self.interior_sizes), base_e, extrap_e)

    def get_bulge_size_energy(self, loop_len_tensor: torch.Tensor) -> torch.Tensor:
        bulge_sizes = torch.exp(self.bulge_sizes_raw)
        idx = loop_len_tensor - 1
        safe_idx = torch.clamp(idx, max=len(bulge_sizes) - 1)
        base_e = bulge_sizes[safe_idx]
        extrap_e = bulge_sizes[-1] + 1.75 * RT * torch.log(loop_len_tensor.float() / float(len(bulge_sizes)))
        return torch.where(idx < len(bulge_sizes), base_e, extrap_e)

class BatchedPartitionFunction(nn.Module):
    def __init__(self, params: ThermoParameters):
        super().__init__()
        self.params = params

    def forward(self, sequences: list[str], bias: torch.Tensor | None = None,
                nicks: list[int] | None = None) -> torch.Tensor:
        """
        Ensemble free energy ΔG (kcal/mol) for a batch of discrete sequences.

        ``bias`` is an optional per-pair energy field of shape ``(B, max_N, max_N)``
        added to the closing-pair entry ``V[i, j]`` of the DP.  It is the hook
        that makes base-pairing probabilities fall out by autograd: because the
        returned free energy is ``F = -RT ln Z`` and ``bias[i, j]`` perturbs the
        energy of *every* structure in which ``(i, j)`` is paired, the gradient
        ``∂F/∂bias[i, j]`` equals the McCaskill pair probability ``P(i, j)``.  A
        single zero-valued ``bias`` plus one backward pass therefore yields the
        whole BPP matrix (see :meth:`pair_probabilities`).  Default ``None``
        leaves the original fast path untouched.
        """
        material = self.params.material
        normalized_seqs = []
        for seq in sequences:
            seq = seq.upper()
            seq = seq.replace("U", "T") if material == "dna" else seq.replace("T", "U")
            normalized_seqs.append(seq)
            
        B = len(normalized_seqs)
        max_N = max(len(s) for s in normalized_seqs)
        device = self.params.stack_table.device
        
        # Compute strictly positive thermodynamic parameters
        ml_base = torch.exp(self.params.ml_base_raw)
        ml_init = torch.exp(self.params.ml_init_raw)
        ml_pair = torch.exp(self.params.ml_pair_raw)
        
        if max_N == 0:
            return torch.zeros(B, device=device)
            
        padded_seqs = [s + 'X'*(max_N - len(s)) for s in normalized_seqs]
        
        base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'X': 0}

        # Base indices — built as a host list then transferred in one shot
        # (per-element tensor assignment forces a sync per cell; a single tensor
        # construct is orders of magnitude cheaper, especially on GPU).
        base_indices = torch.tensor(
            [[base_to_idx.get(ch, 0) for ch in seq] for seq in padded_seqs],
            dtype=torch.long, device=device,
        )

        # Pairability — a 4×4 lookup applied with two gathers, then masked to
        # real (non-padded) positions.  Replaces the O(B·N²) Python double loop.
        pair_lut = torch.zeros(4, 4, dtype=torch.bool, device=device)
        wc_codes = (
            [(0, 3), (3, 0), (2, 1), (1, 2), (2, 3), (3, 2)]  # RNA: + GU/UG wobble
            if material == "rna"
            else [(0, 3), (3, 0), (2, 1), (1, 2)]              # DNA: Watson-Crick
        )
        for a, c in wc_codes:
            pair_lut[a, c] = True
        can = pair_lut[base_indices.unsqueeze(2), base_indices.unsqueeze(1)]
        lengths_t = torch.tensor([len(s) for s in sequences], device=device)
        real = torch.arange(max_N, device=device).unsqueeze(0) < lengths_t.unsqueeze(1)
        can_pair_mask = can & real.unsqueeze(2) & real.unsqueeze(1)

        # ── Nick / separation masks (multi-strand complexes) ─────────────────
        # ``nicks`` are the cumulative strand lengths (0 < nick < N), i.e. the
        # positions where one strand ends and the next begins.  A pair (i, j)
        # is *inter*-strand when a nick lies in (i, j] and *spans* a nick when
        # one lies strictly inside (i, j).  Matching ensemble.py:_can_pair_nicks
        # / _qb_val: cross-nick pairs may close at any separation (no minimum
        # loop), hairpins may not span a nick, and an inter-strand pair whose
        # immediate inner pair cannot form carries a terminal-pair penalty leaf
        # (the base case of an inter-strand helix).  With ``nicks=None`` every
        # mask collapses to the original single-strand behaviour.
        row = torch.arange(max_N, device=device)
        sep_gt = (row.unsqueeze(0) - row.unsqueeze(1)) > MIN_HAIRPIN_LOOP  # [i,j]: j-i>3
        if nicks:
            nset = sorted({int(k) for k in nicks if 0 < k < max_N})
            if nset:
                nt = torch.tensor(nset, device=device)
                prefix = (nt.unsqueeze(0) <= row.unsqueeze(1)).sum(dim=1)  # #nicks <= t
            else:
                prefix = torch.zeros(max_N, dtype=torch.long, device=device)
            is_inter_m = (prefix.unsqueeze(0) - prefix.unsqueeze(1)) > 0      # nick in (i,j]
            jm1 = torch.clamp(row - 1, min=0)
            spans_m = (prefix[jm1].unsqueeze(0) - prefix.unsqueeze(1)) > 0    # nick in (i,j)
            no_span_m = ~spans_m
            pair_allowed_sep = is_inter_m | sep_gt
        else:
            is_inter_m = torch.zeros(max_N, max_N, dtype=torch.bool, device=device)
            no_span_m = torch.ones(max_N, max_N, dtype=torch.bool, device=device)
            pair_allowed_sep = sep_gt
        # Pairs that may actually close, base-compatibility AND separation/nick
        # rule.  Equivalent to ``can_pair_mask`` on the single-strand path (where
        # the DP's V=INF already encodes the separation rule), so reusing it in
        # the stack masks below leaves single-strand results unchanged.
        pair_allowed = can_pair_mask & pair_allowed_sep
        has_nicks = bool(nicks)

        # Hairpin closing-pair bonus (terminal penalty + tri/tetraloop tables).
        # The special-loop lookups are inherently dict-based; we accumulate into
        # a host list and transfer once rather than writing cell-by-cell.
        hb = [[[0.0] * max_N for _ in range(max_N)] for _ in range(B)]
        tp_dict = self.params.terminal_penalty_dict
        tri_dict = self.params.triloop_dict
        tet_dict = self.params.tetraloop_dict
        for b, seq in enumerate(normalized_seqs):
            L = len(sequences[b])
            for i in range(L):
                for j in range(i + 4, L):  # loop size >= 3 -> j - i >= 4
                    loop_size = j - i - 1
                    dG_bonus = tp_dict.get(seq[i] + seq[j], 0.0)
                    if loop_size == 3:
                        dG_bonus += tp_dict.get(seq[j] + seq[i], 0.0)
                        dG_bonus += tri_dict.get(seq[i:j + 1], 0.0)
                    elif loop_size == 4:
                        dG_bonus += tet_dict.get(seq[i:j + 1], 0.0)
                    hb[b][i][j] = dG_bonus
        hairpin_bonus = torch.tensor(hb, dtype=torch.float64, device=device)

        # Intermediate Dynamic Programming Array Allocation
        V = torch.full((B, max_N, max_N), INF, dtype=torch.float64, device=device)
        W = torch.zeros((B, max_N, max_N), dtype=torch.float64, device=device) 
        M = torch.full((B, max_N, max_N), INF, dtype=torch.float64, device=device) 
        M1 = torch.full((B, max_N, max_N), INF, dtype=torch.float64, device=device) 
        
        for d in range(1, max_N):
            num_i = max_N - d
            i_idx = torch.arange(num_i, device=device)
            j_idx = i_idx + d
            
            pair_mask = can_pair_mask[:, i_idx, j_idx]
            v_options = []
            
            # 1. Hairpin Loops
            if d > MIN_HAIRPIN_LOOP:
                hp_len = d - 1
                hp_e = self.params.get_hairpin_size_energy(hp_len)
                hp_e_tensor = hp_e.expand(B, num_i)

                # Mathews-Turner: the first-mismatch (TERMINAL_MISMATCH) bonus at
                # the closing pair applies only to tetraloops and larger.  For
                # triloops (hp_len == 3) the special-loop table already encodes
                # the loop-specific stability, so adding the mismatch would
                # double-count.  ensemble.py:_hairpin_loop_energy follows the
                # same convention by returning early on loop_size == 3.
                if hp_len > 3:
                    j_minus_1 = torch.clamp(j_idx - 1, min=0)
                    i_plus_1 = torch.clamp(i_idx + 1, max=max_N - 1)
                    idx = (
                        base_indices[:, j_minus_1] * 64
                        + base_indices[:, j_idx] * 16
                        + base_indices[:, i_idx] * 4
                        + base_indices[:, i_plus_1]
                    )
                    term_penalty = self.params.term_mismatch[idx]
                else:
                    term_penalty = torch.zeros(B, num_i, dtype=torch.float64, device=device)

                bonus = hairpin_bonus[:, i_idx, j_idx]
                hp_mask = pair_mask & no_span_m[i_idx, j_idx] if has_nicks else pair_mask
                v_hp = torch.where(hp_mask, hp_e_tensor + term_penalty + bonus, torch.tensor(INF, dtype=torch.float64, device=device))
                v_options.append(v_hp)

            # 1b. Inter-strand helix base case: an inter-strand pair (i,j) whose
            #     immediate inner pair (i+1, j-1) cannot form carries a
            #     terminal-pair penalty leaf (ensemble.py:_qb_val, is_inter branch).
            if has_nicks:
                inter_ij = is_inter_m[i_idx, j_idx]
                inner_allowed = pair_allowed[:, torch.clamp(i_idx + 1, max=max_N - 1),
                                             torch.clamp(j_idx - 1, min=0)]
                leaf_mask = pair_mask & inter_ij.unsqueeze(0) & (~inner_allowed)
                tp_leaf = self.params.terminal_penalty_lut[
                    base_indices[:, i_idx], base_indices[:, j_idx]]
                v_leaf = torch.where(leaf_mask, tp_leaf,
                                     torch.tensor(INF, dtype=torch.float64, device=device))
                v_options.append(v_leaf)

            # 2. Base Pair Stacking
            if d > MIN_HAIRPIN_LOOP + 1 or (has_nicks and d >= 3):
                inner_pair_mask = pair_allowed[:, i_idx+1, j_idx-1]
                valid_stack = pair_allowed[:, i_idx, j_idx] & inner_pair_mask

                # Full 4-base stack lookup: key seq[i],seq[i+1],seq[j-1],seq[j]
                # (matches ensemble._stack_energy), so GU-wobble stacks use their
                # own value instead of the Watson-Crick dinucleotide energy.
                stk_idx = (
                    base_indices[:, i_idx] * 64
                    + base_indices[:, i_idx + 1] * 16
                    + base_indices[:, j_idx - 1] * 4
                    + base_indices[:, j_idx]
                )
                stack_params = self.params.stack_table[stk_idx]

                inner_v = V[:, i_idx+1, j_idx-1]
                v_stack = torch.where(valid_stack, stack_params + inner_v, torch.tensor(INF, dtype=torch.float64, device=device))
                v_options.append(v_stack)
                
            # 3. Interior Loops and Bulges
            max_n = min(30, d - 2)
            if max_n >= 1:
                for n in range(1, max_n + 1):
                    d_prime = d - 2 - n
                    if d_prime <= 0:
                        continue
                        
                    nl_tensor = torch.arange(n + 1, device=device)
                    nr_tensor = n - nl_tensor
                    
                    ip_idx = i_idx.unsqueeze(1) + 1 + nl_tensor.unsqueeze(0)
                    jp_idx = j_idx.unsqueeze(1) - 1 - nr_tensor.unsqueeze(0)
                    
                    valid_ip_jp = (ip_idx < max_N) & (jp_idx >= 0) & (ip_idx < jp_idx)
                    ip_safe = torch.clamp(ip_idx, max=max_N-1)
                    jp_safe = torch.clamp(jp_idx, min=0)
                    
                    inner_v = V[:, ip_safe, jp_safe]
                    inner_pair_mask = pair_allowed[:, ip_safe, jp_safe]
                    outer_ok = (pair_allowed[:, i_idx, j_idx] if has_nicks else pair_mask)
                    valid_interior = valid_ip_jp.unsqueeze(0) & inner_pair_mask & outer_ok.unsqueeze(2)
                    
                    is_bulge = (nl_tensor == 0) | (nr_tensor == 0)
                    n_tensor = torch.tensor(n, device=device)
                    dG_penalty = torch.where(
                        is_bulge,
                        self.params.get_bulge_size_energy(n_tensor),
                        self.params.get_interior_size_energy(n_tensor)
                    )
                    
                    asym = torch.abs(nl_tensor - nr_tensor)
                    ninio_number = torch.clamp(torch.min(nl_tensor, nr_tensor), max=4) - 1
                    ninio_number_safe = torch.clamp(ninio_number, min=0)
                    asym_penalty = torch.min(
                        self.params.asymmetry_ninio[4],
                        asym * self.params.asymmetry_ninio[ninio_number_safe]
                    )
                    dG_penalty = dG_penalty + torch.where(is_bulge, torch.tensor(0.0, dtype=torch.float64, device=device), asym_penalty)
                    dG_penalty_exp = dG_penalty.unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)

                    # Terminal-pair (TP) contribution at the helix junctions of
                    # the interior loop / bulge.  Matches the universal
                    # correction in ensemble.py:_interior_bulge_energy:
                    #   RNA single-base bulge:        +TP_outer - TP_inner
                    #   RNA multi-base bulge:         +2·TP_outer
                    #   RNA general interior loop:    +2·TP_outer
                    #   DNA (all interior loops/bulges): +TP_outer - TP_inner
                    # TP_outer / TP_inner are looked up from the 4×4 LUT keyed
                    # by the closing-pair base indices.
                    tp_lut = self.params.terminal_penalty_lut
                    b_i = base_indices[:, i_idx]                       # (B, num_i)
                    b_j = base_indices[:, j_idx]                       # (B, num_i)
                    b_ip = base_indices[:, ip_safe]                    # (B, num_i, n+1)
                    b_jp = base_indices[:, jp_safe]                    # (B, num_i, n+1)
                    TP_outer = tp_lut[b_i, b_j]                        # (B, num_i)
                    TP_inner = tp_lut[b_ip, b_jp]                      # (B, num_i, n+1)

                    if material == "rna":
                        if n == 1:
                            # All n==1 entries are bulges (single-base bulge).
                            tp_correction = TP_outer.unsqueeze(-1) - TP_inner
                        else:
                            # +2·TP_outer for both multi-base bulges and
                            # general interior loops under the RNA convention.
                            tp_correction = (2.0 * TP_outer).unsqueeze(-1).expand(-1, -1, n + 1)
                    else:
                        # DNA: only the universal correction TP_outer - TP_inner;
                        # INTERIOR_MISMATCH at the junctions is not yet wired
                        # into the diff engine.
                        tp_correction = TP_outer.unsqueeze(-1) - TP_inner

                    dG_penalty_exp = dG_penalty_exp + tp_correction

                    # Single-base bulge: the helix stacks *across* the bulge, so
                    # the closing and inner pairs contribute a nearest-neighbor
                    # stack (ensemble.py:_interior_bulge_energy, n==1 branch).
                    # Key seq[i],seq[ip],seq[jp],seq[j] — material-agnostic.
                    if n == 1:
                        stk_idx_b = (
                            b_i.unsqueeze(-1) * 64
                            + b_ip * 16
                            + b_jp * 4
                            + b_j.unsqueeze(-1)
                        )
                        dG_penalty_exp = dG_penalty_exp + self.params.stack_table[stk_idx_b]

                    # Neural DP Global Context Fine-Tuning
                    if n == 2:
                        b1, b2 = base_indices[:, i_idx], base_indices[:, j_idx]
                        b3, b4 = base_indices[:, i_idx + 1], base_indices[:, j_idx - 1]
                        b5, b6 = base_indices[:, i_idx + 2], base_indices[:, j_idx - 2]
                        
                        bases = torch.stack([b1, b2, b3, b4, b5, b6], dim=-1)
                        one_hot = torch.nn.functional.one_hot(bases, num_classes=4).to(torch.float64)
                        e_1_1 = self.params.mlp_1_1(one_hot.view(B, num_i, 24)).squeeze(-1)
                        
                        # Ensure the MLP only touches symmetric 1x1 loops, not 0x2 or 2x0 bulges
                        is_true_1_1 = (nl_tensor == 1).unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)
                        dG_penalty_exp = dG_penalty_exp + torch.where(is_true_1_1, e_1_1.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float64, device=device))
                        
                    elif n == 3:
                        b1, b2 = base_indices[:, i_idx], base_indices[:, j_idx]
                        b3, b4 = base_indices[:, i_idx + 1], base_indices[:, j_idx - 1]
                        b5, b6 = base_indices[:, j_idx - 2], base_indices[:, i_idx + 2]
                        b7 = base_indices[:, j_idx - 3]
                        
                        bases = torch.stack([b1, b2, b3, b4, b5, b6, b7], dim=-1)
                        one_hot = torch.nn.functional.one_hot(bases, num_classes=4).to(torch.float64)
                        e_1_2 = self.params.mlp_1_2(one_hot.view(B, num_i, 28)).squeeze(-1)
                        
                        # Isolate 1x2 interior loops
                        is_true_1_2 = (nl_tensor == 1).unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)
                        dG_penalty_exp = dG_penalty_exp + torch.where(is_true_1_2, e_1_2.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float64, device=device))
                        
                        b1_2, b2_2 = base_indices[:, j_idx], base_indices[:, i_idx]
                        b3_2, b4_2 = base_indices[:, j_idx - 1], base_indices[:, i_idx + 1]
                        b5_2, b6_2 = base_indices[:, i_idx + 2], base_indices[:, j_idx - 2]
                        b7_2 = base_indices[:, i_idx + 3]
                        
                        bases_2 = torch.stack([b1_2, b2_2, b3_2, b4_2, b5_2, b6_2, b7_2], dim=-1)
                        one_hot_2 = torch.nn.functional.one_hot(bases_2, num_classes=4).to(torch.float64)
                        e_2_1 = self.params.mlp_1_2(one_hot_2.view(B, num_i, 28)).squeeze(-1)
                        
                        # Isolate 2x1 interior loops
                        is_true_2_1 = (nl_tensor == 2).unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)
                        dG_penalty_exp = dG_penalty_exp + torch.where(is_true_2_1, e_2_1.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float64, device=device))
                        
                    elif n == 4:
                        b1, b2 = base_indices[:, i_idx], base_indices[:, j_idx]
                        b3, b4 = base_indices[:, i_idx + 1], base_indices[:, i_idx + 2]
                        b5, b6 = base_indices[:, j_idx - 1], base_indices[:, j_idx - 2]
                        b7, b8 = base_indices[:, i_idx + 3], base_indices[:, j_idx - 3]
                        
                        bases = torch.stack([b1, b2, b3, b4, b5, b6, b7, b8], dim=-1)
                        one_hot = torch.nn.functional.one_hot(bases, num_classes=4).to(torch.float64)
                        e_2_2 = self.params.mlp_2_2(one_hot.view(B, num_i, 32)).squeeze(-1)
                        
                        # Isolate 2x2 interior loops
                        is_true_2_2 = (nl_tensor == 2).unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)
                        dG_penalty_exp = dG_penalty_exp + torch.where(is_true_2_2, e_2_2.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float64, device=device))
                        
                    interior_energies = inner_v + dG_penalty_exp
                    int_scaled = -interior_energies / RT
                    int_scaled = torch.where(valid_interior, int_scaled, torch.tensor(-INF, dtype=torch.float64, device=device))
                    
                    v_int = -RT * torch.logsumexp(int_scaled, dim=-1)
                    v_int = torch.where(torch.isinf(v_int), torch.tensor(INF, dtype=torch.float64, device=device), v_int)
                    v_int = torch.where(pair_mask, v_int, torch.tensor(INF, dtype=torch.float64, device=device))
                    v_options.append(v_int)
                    
            # 4. Multiloop Bifurcation (Autograd-Safe 1D Coordinate Loop)
            #
            # Charge accounting (linear multiloop model, matches
            # ensemble.py:_qb_val via bm_ml_init_pair = boltz(ML_INIT + ML_PAIR)
            # at the outer pair, plus ML_PAIR per inner stem via QM):
            #   - 1 × ml_init      (the constant initiation cost)
            #   - 1 × ml_pair      (for the OUTER closing pair (i, j))
            #   - 1 × ml_pair      (for the FIRST inner stem V[i+1, k])
            #   - already inside M[k+1, j-1]: one ml_pair per remaining inner stem
            # Total: ml_init + (N + 1) × ml_pair for a multiloop with N inner
            # stems, matching the native engine.
            L_bif = d - 3
            if L_bif > 0:
                bif_bastes = []
                for m in range(L_bif):
                    k_vals = i_idx + 2 + m
                    v_k = V[:, i_idx + 1, k_vals]
                    w_k = M[:, k_vals + 1, j_idx - 1]
                    bif_bastes.append(v_k + w_k)

                bif_energies = torch.stack(bif_bastes, dim=-1)
                # ml_pair here is the cost of the first inner stem V[i+1, k];
                # the remaining inner stems are already charged inside M.
                bif_scaled = -(bif_energies + ml_pair) / RT
                v_bif = -RT * torch.logsumexp(bif_scaled, dim=-1)
                # Outer closing pair (i, j) charges ml_init + ml_pair, matching
                # ensemble.py:395 (bm_ml_init_pair = boltz(ML_INIT + ML_PAIR)).
                v_bif = v_bif + ml_init + ml_pair
                bif_mask = pair_allowed[:, i_idx, j_idx] if has_nicks else pair_mask
                v_bif = torch.where(bif_mask, v_bif, torch.tensor(INF, dtype=torch.float64, device=device))
                v_options.append(v_bif)
                
            if v_options:
                v_ij = logsumexp_RT(v_options, dim=0)
                if bias is not None:
                    v_ij = v_ij + bias[:, i_idx, j_idx]
                V[:, i_idx, j_idx] = v_ij

            # 5. Populate External Loop (W) and Multiloop (M) Segments
            
            # A. Compute M1 (Exactly one stem, flush right, grows left)
            m1_options = [V[:, i_idx, j_idx] + ml_pair] # Base case: just the stem flush right
            if d >= 1:
                m1_options.append(M1[:, i_idx+1, j_idx] + ml_base) # Add unpaired bases strictly to the left
            M1[:, i_idx, j_idx] = logsumexp_RT(m1_options, dim=0)

            # B. Compute M (One or more stems, grows right)
            m_options = [
                M1[:, i_idx, j_idx],                         # Case 1: It's just a single stem segment
                M[:, i_idx, j_idx-1] + ml_base               # Case 2: Extend trailing unpaired space to the right
            ]

            # Case 3: Multiloop Bifurcation
            for m_w in range(d):
                k_w = i_idx + m_w
                m_options.append(M[:, i_idx, k_w] + M1[:, k_w+1, j_idx])

            M[:, i_idx, j_idx] = logsumexp_RT(m_options, dim=0)

            # C. Compute W (External loop, grows right).
            #
            # Mirrors the four-state external-loop decoration in
            # ensemble.py:_fill_dp_nicks (BARE / D5 / D3 / D5+D3), with each
            # dangle decoration sign-gated (only added when ΔG_dangle < 0).
            # Dangle bases are explicitly removed from the left context: the
            # D5 cases use W[i, k-2] rather than W[i, k-1], so the dangle base
            # at k-1 is not double-counted as unpaired.
            INF_T = torch.full((B, num_i), INF, dtype=torch.float64, device=device)
            ZERO_T = torch.zeros(B, num_i, dtype=torch.float64, device=device)

            w_options = [W[:, i_idx, j_idx - 1]]  # j unpaired (extend W right)

            for m_w in range(d + 1):
                k_w = i_idx + m_w  # stem 5' index, k in [i, j]
                k_minus_1_safe = torch.clamp(k_w - 1, min=0)
                k_minus_2_safe = torch.clamp(k_w - 2, min=0)

                # w_left  = W[i, k-1] when k > i, else empty seg (energy 0)
                # w_left5 = W[i, k-2] when k > i+1; if k == i+1 the dangle
                #           base sits at i so left5 is the empty seg (energy 0).
                if m_w == 0:
                    w_left = ZERO_T
                else:
                    w_left = W[:, i_idx, k_minus_1_safe]
                if m_w >= 2:
                    w_left5 = W[:, i_idx, k_minus_2_safe]
                else:
                    w_left5 = ZERO_T  # only consulted when m_w >= 1

                # ── Stem (k_w, j_idx) ──────────────────────────────────────
                v_right = V[:, k_w, j_idx]

                # BARE: no decoration
                w_options.append(w_left + v_right)

                # D5: dangle base at k-1 (requires m_w >= 1)
                if m_w >= 1:
                    idx_d5 = (
                        base_indices[:, k_w] * 16
                        + base_indices[:, j_idx] * 4
                        + base_indices[:, k_minus_1_safe]
                    )
                    d5 = self.params.dangle_5[idx_d5]
                    d5_term = w_left5 + v_right + d5
                    d5_term = torch.where(d5 < 0, d5_term, INF_T)
                    w_options.append(d5_term)

                # ── Stem (k_w, j_idx - 1) with D3 dangle at j_idx ──────────
                if m_w < d:
                    j_m1 = j_idx - 1
                    v_right_d3 = V[:, k_w, j_m1]

                    idx_d3 = (
                        base_indices[:, j_idx] * 16
                        + base_indices[:, j_m1] * 4
                        + base_indices[:, k_w]
                    )
                    d3 = self.params.dangle_3[idx_d3]
                    d3_valid = d3 < 0

                    # D3 alone
                    d3_term = w_left + v_right_d3 + d3
                    d3_term = torch.where(d3_valid, d3_term, INF_T)
                    w_options.append(d3_term)

                    # D5 + D3 (requires m_w >= 1 for the D5 base)
                    if m_w >= 1:
                        idx_d5_53 = (
                            base_indices[:, k_w] * 16
                            + base_indices[:, j_m1] * 4
                            + base_indices[:, k_minus_1_safe]
                        )
                        d5_53 = self.params.dangle_5[idx_d5_53]
                        both_term = w_left5 + v_right_d3 + d5_53 + d3
                        both_term = torch.where((d5_53 < 0) & d3_valid, both_term, INF_T)
                        w_options.append(both_term)

            W[:, i_idx, j_idx] = logsumexp_RT(w_options, dim=0)

        lengths = [len(s) for s in normalized_seqs]
        res = []
        for b, L in enumerate(lengths):
            if L == 0:
                res.append(torch.tensor(0.0, dtype=torch.float64, device=device))
            else:
                res.append(W[b, 0, L - 1])
            
        return torch.stack(res)

    def soft_forward(self, probs: torch.Tensor, bias: torch.Tensor | None = None,
                     nicks: list[int] | None = None) -> torch.Tensor:
        """
        Sequence-differentiable ensemble free energy ΔG (kcal/mol).

        ``forward`` above takes discrete strings, so its autograd graph reaches
        the *parameters* but not the *sequence* (the bases are hard integer
        indices).  ``soft_forward`` instead consumes a continuous per-position
        base distribution ``probs`` of shape ``(B, N, 4)`` — a point on the
        product of probability simplices over {A, C, G, U/T} — and runs the same
        McCaskill DP with every hard table lookup replaced by its expectation
        under those distributions (a tensor contraction) and the hard pairability
        mask replaced by a soft penalty ``-RT·log(p_pair)``.  This is a mean-field
        relaxation that **reduces to the exact engine in the one-hot limit** and
        is differentiable w.r.t. ``probs`` — which is exactly what gradient-based
        sequence design needs and what a closed C kernel cannot provide.

        The neural interior-loop MLP corrections are omitted; their final layers
        are zero-initialised, so for an untrained :class:`ThermoParameters` they
        contribute nothing and the relaxation stays faithful to the base engine.
        Returns a 1-D tensor of free energies, one per row of ``probs``.
        """
        p = self.params
        material = p.material
        B, N, _ = probs.shape
        device = probs.device
        dtype = probs.dtype

        ml_base = torch.exp(p.ml_base_raw)
        ml_init = torch.exp(p.ml_init_raw)
        ml_pair = torch.exp(p.ml_pair_raw)

        # Reshaped lookup tables: integer flat-index -> (4,4,…) so a gather
        # becomes an einsum against the position distributions.
        stack4 = p.stack_table.view(4, 4, 4, 4)      # [i][i+1][j-1][j]
        term4 = p.term_mismatch.view(4, 4, 4, 4)     # [j-1][j][i][i+1]
        d5t = p.dangle_5.view(4, 4, 4)               # [k][j][k-1]
        d3t = p.dangle_3.view(4, 4, 4)               # [j][j-1][k]
        tp = p.terminal_penalty_lut                  # (4,4)

        # Soft pairability weight p_pair(i,j) = P(bases i,j form a valid pair)
        # under independent per-position distributions, turned into an additive
        # closing-pair penalty.  One-hot -> 0 for a real pair, ~+12.8 kcal for an
        # impossible one (eps floor), recovering the hard mask in the limit.
        pairf = torch.zeros(4, 4, dtype=dtype, device=device)
        wc = ([(0, 3), (3, 0), (2, 1), (1, 2), (2, 3), (3, 2)]
              if material == "rna" else [(0, 3), (3, 0), (2, 1), (1, 2)])
        for a, c in wc:
            pairf[a, c] = 1.0
        cval = torch.einsum("biw,bjx,wx->bij", probs, probs, pairf)
        pairpen = -RT * torch.log(cval + 1e-9)

        # Nick / separation masks (see ``forward``): boolean, sequence-independent.
        row = torch.arange(N, device=device)
        sep_gt = (row.unsqueeze(0) - row.unsqueeze(1)) > MIN_HAIRPIN_LOOP  # [i,j]: j-i>3
        has_nicks = bool(nicks)
        if has_nicks:
            nset = sorted({int(k) for k in nicks if 0 < k < N})
            prefix = ((torch.tensor(nset, device=device).unsqueeze(0) <= row.unsqueeze(1)).sum(dim=1)
                      if nset else torch.zeros(N, dtype=torch.long, device=device))
            is_inter_m = (prefix.unsqueeze(0) - prefix.unsqueeze(1)) > 0
            jm1 = torch.clamp(row - 1, min=0)
            no_span_m = ~((prefix[jm1].unsqueeze(0) - prefix.unsqueeze(1)) > 0)
            pair_allowed_sep = is_inter_m | sep_gt
            INF_NN = torch.where(pair_allowed_sep, torch.zeros((), dtype=dtype, device=device),
                                 torch.full((), INF, dtype=dtype, device=device))
        else:
            is_inter_m = no_span_m = None
            pair_allowed_sep = sep_gt

        V = torch.full((B, N, N), INF, dtype=dtype, device=device)
        W = torch.zeros((B, N, N), dtype=dtype, device=device)
        M = torch.full((B, N, N), INF, dtype=dtype, device=device)
        M1 = torch.full((B, N, N), INF, dtype=dtype, device=device)

        for d in range(1, N):
            num_i = N - d
            i_idx = torch.arange(num_i, device=device)
            j_idx = i_idx + d
            ZERO_T = torch.zeros(B, num_i, dtype=dtype, device=device)
            INF_T = torch.full((B, num_i), INF, dtype=dtype, device=device)

            pi = probs[:, i_idx]            # (B, num_i, 4)
            pj = probs[:, j_idx]
            v_options = []

            # 1. Hairpin loop closed by (i, j)
            if d > MIN_HAIRPIN_LOOP:
                hp_len = d - 1
                hp_e = p.get_hairpin_size_energy(hp_len).expand(B, num_i)
                if hp_len > 3:
                    term_penalty = torch.einsum(
                        "bnw,bnx,bny,bnz,wxyz->bn",
                        probs[:, j_idx - 1], pj, pi, probs[:, i_idx + 1], term4)
                else:
                    term_penalty = ZERO_T
                bonus = torch.einsum("bnw,bnx,wx->bn", pi, pj, tp)  # terminal penalty
                hp = hp_e + term_penalty + bonus
                if has_nicks:  # no hairpin may span a nick
                    hp = torch.where(no_span_m[i_idx, j_idx].unsqueeze(0), hp, INF_T)
                v_options.append(hp)

            # 1b. Inter-strand helix base case (terminal-pair penalty leaf): an
            #     inter-strand pair whose immediate inner pair cannot form.
            if has_nicks:
                inter_ij = is_inter_m[i_idx, j_idx]
                ip1 = torch.clamp(i_idx + 1, max=N - 1)
                jm1c = torch.clamp(j_idx - 1, min=0)
                inner_sep_ok = pair_allowed_sep[ip1, jm1c]
                leaf = torch.einsum("bnw,bnx,wx->bn", pi, pj, tp)   # expected TP penalty
                leaf = torch.where((inter_ij & ~inner_sep_ok).unsqueeze(0), leaf, INF_T)
                v_options.append(leaf)

            # 2. Stacking onto the inner pair (i+1, j-1)
            if d > MIN_HAIRPIN_LOOP + 1 or (has_nicks and d >= 3):
                stack_val = torch.einsum(
                    "bnw,bnx,bny,bnz,wxyz->bn",
                    pi, probs[:, i_idx + 1], probs[:, j_idx - 1], pj, stack4)
                stack_opt = stack_val + V[:, i_idx + 1, j_idx - 1]
                if has_nicks:  # only inter pairs may stack at sub-loop separation
                    stack_opt = stack_opt + INF_NN[i_idx, j_idx].unsqueeze(0)
                v_options.append(stack_opt)

            # 3. Interior loops & bulges (size/asymmetry penalties are sequence
            #    independent; only the terminal-pair and single-bulge-stack
            #    corrections carry sequence dependence).
            max_n = min(30, d - 2)
            for n in range(1, max_n + 1):
                if d - 2 - n <= 0:
                    continue
                nl = torch.arange(n + 1, device=device)
                nr = n - nl
                ip = i_idx.unsqueeze(1) + 1 + nl.unsqueeze(0)
                jp = j_idx.unsqueeze(1) - 1 - nr.unsqueeze(0)
                valid = (ip < N) & (jp >= 0) & (ip < jp)
                ip_s = torch.clamp(ip, max=N - 1)
                jp_s = torch.clamp(jp, min=0)
                inner_v = V[:, ip_s, jp_s]                    # (B, num_i, n+1)

                is_bulge = (nl == 0) | (nr == 0)
                n_t = torch.tensor(n, device=device)
                dG = torch.where(is_bulge,
                                 p.get_bulge_size_energy(n_t),
                                 p.get_interior_size_energy(n_t))
                asym = torch.abs(nl - nr)
                ninio_i = torch.clamp(torch.clamp(torch.min(nl, nr), max=4) - 1, min=0)
                asym_pen = torch.min(p.asymmetry_ninio[4], asym * p.asymmetry_ninio[ninio_i])
                dG = dG + torch.where(is_bulge, torch.zeros_like(asym_pen.double()), asym_pen)
                dG_exp = dG.unsqueeze(0).unsqueeze(0).expand(B, num_i, n + 1)

                p_ip = probs[:, ip_s]                         # (B, num_i, n+1, 4)
                p_jp = probs[:, jp_s]
                TP_outer = torch.einsum("bnw,bnx,wx->bn", pi, pj, tp)
                TP_inner = torch.einsum("bnkw,bnkx,wx->bnk", p_ip, p_jp, tp)
                if material == "rna":
                    if n == 1:
                        tp_corr = TP_outer.unsqueeze(-1) - TP_inner
                    else:
                        tp_corr = (2.0 * TP_outer).unsqueeze(-1).expand(-1, -1, n + 1)
                else:
                    tp_corr = TP_outer.unsqueeze(-1) - TP_inner
                dG_exp = dG_exp + tp_corr

                if n == 1:
                    dG_exp = dG_exp + torch.einsum(
                        "bnw,bnkx,bnky,bnz,wxyz->bnk", pi, p_ip, p_jp, pj, stack4)

                interior = inner_v + dG_exp
                int_scaled = torch.where(valid.unsqueeze(0), -interior / RT,
                                         torch.tensor(-INF, dtype=dtype, device=device))
                v_int = -RT * torch.logsumexp(int_scaled, dim=-1)
                v_int = torch.where(torch.isinf(v_int),
                                    torch.tensor(INF, dtype=dtype, device=device), v_int)
                v_options.append(v_int)

            # 4. Multiloop bifurcation
            L_bif = d - 3
            if L_bif > 0:
                bif = []
                for m in range(L_bif):
                    k = i_idx + 2 + m
                    bif.append(V[:, i_idx + 1, k] + M[:, k + 1, j_idx - 1])
                bif_scaled = -(torch.stack(bif, dim=-1) + ml_pair) / RT
                v_bif = -RT * torch.logsumexp(bif_scaled, dim=-1) + ml_init + ml_pair
                v_options.append(v_bif)

            if v_options:
                # Add the soft closing-pair penalty once: every contribution to
                # V[i,j] requires (i,j) to actually pair.
                v_ij = logsumexp_RT(v_options, dim=0) + pairpen[:, i_idx, j_idx]
                if has_nicks:  # forbid disallowed-separation (sub-loop intra) pairs
                    v_ij = v_ij + INF_NN[i_idx, j_idx].unsqueeze(0)
                # Per-pair energy bias: ∂F/∂bias[i,j] = P(i,j) (soft BPP).
                if bias is not None:
                    v_ij = v_ij + bias[:, i_idx, j_idx]
                V[:, i_idx, j_idx] = v_ij

            # M1 / M (multiloop segments)
            m1_options = [V[:, i_idx, j_idx] + ml_pair]
            if d >= 1:
                m1_options.append(M1[:, i_idx + 1, j_idx] + ml_base)
            M1[:, i_idx, j_idx] = logsumexp_RT(m1_options, dim=0)

            m_options = [M1[:, i_idx, j_idx], M[:, i_idx, j_idx - 1] + ml_base]
            for m_w in range(d):
                k_w = i_idx + m_w
                m_options.append(M[:, i_idx, k_w] + M1[:, k_w + 1, j_idx])
            M[:, i_idx, j_idx] = logsumexp_RT(m_options, dim=0)

            # W (external loop) with sign-gated 5'/3' dangle decorations
            w_options = [W[:, i_idx, j_idx - 1]]
            for m_w in range(d + 1):
                k_w = i_idx + m_w
                k1 = torch.clamp(k_w - 1, min=0)
                k2 = torch.clamp(k_w - 2, min=0)
                w_left = ZERO_T if m_w == 0 else W[:, i_idx, k1]
                w_left5 = W[:, i_idx, k2] if m_w >= 2 else ZERO_T
                pk = probs[:, k_w]
                v_right = V[:, k_w, j_idx]
                w_options.append(w_left + v_right)

                if m_w >= 1:
                    d5 = torch.einsum("bnw,bnx,bny,wxy->bn", pk, pj, probs[:, k1], d5t)
                    w_options.append(torch.where(d5 < 0, w_left5 + v_right + d5, INF_T))

                if m_w < d:
                    j_m1 = j_idx - 1
                    pjm1 = probs[:, j_m1]
                    v_right_d3 = V[:, k_w, j_m1]
                    d3 = torch.einsum("bnw,bnx,bny,wxy->bn", pj, pjm1, pk, d3t)
                    d3_valid = d3 < 0
                    w_options.append(torch.where(d3_valid, w_left + v_right_d3 + d3, INF_T))
                    if m_w >= 1:
                        d5_53 = torch.einsum("bnw,bnx,bny,wxy->bn", pk, pjm1, probs[:, k1], d5t)
                        both = w_left5 + v_right_d3 + d5_53 + d3
                        w_options.append(torch.where((d5_53 < 0) & d3_valid, both, INF_T))

            W[:, i_idx, j_idx] = logsumexp_RT(w_options, dim=0)

        return W[:, 0, N - 1]

    # ── Base-pair probabilities (McCaskill BPP via autodiff) ─────────────────
    #
    # The identity ``P(i, j) = ∂F/∂ε_ij`` — pair probability equals the
    # derivative of the ensemble free energy with respect to an energy that is
    # charged whenever (i, j) is paired — turns one backward pass over a
    # zero-valued ``bias`` field into the full BPP matrix.  This is exact (not a
    # heuristic), reduces to McCaskill's outside recursion, and — crucially for
    # design — is differentiable a second time w.r.t. a soft sequence, so losses
    # built on the BPP (ensemble defect, accessibility, entropy) propagate
    # gradients back to ``probs``.

    def _bpp_from_grad(self, free_energy: torch.Tensor, bias: torch.Tensor,
                       create_graph: bool) -> torch.Tensor:
        """Differentiate ``F`` w.r.t. the per-pair ``bias`` field and symmetrize.

        The DP fills only the upper triangle ``i < j``, so the raw gradient is
        upper-triangular; adding its transpose yields a symmetric matrix with
        ``bpp[i, j] = bpp[j, i] = P(i, j)`` and a zero diagonal.
        """
        (grad,) = torch.autograd.grad(
            free_energy.sum(), bias, create_graph=create_graph)
        return grad + grad.transpose(-1, -2)

    def free_energy_and_bpp(self, sequences: list[str],
                            nicks: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(free_energy (B,), bpp (B, N, N))`` for discrete sequences.

        Folds once with an autograd hook on the closing-pair energies, so the
        free energy and the entire pair-probability matrix come out of a single
        forward + backward pass.  Runs under :func:`torch.enable_grad` even if
        called inside ``torch.no_grad`` (the gradient *is* the answer here).
        """
        device = self.params.stack_table.device
        B = len(sequences)
        max_N = max((len(s) for s in sequences), default=0)
        with torch.enable_grad():
            bias = torch.zeros(B, max_N, max_N, dtype=torch.float64,
                               device=device, requires_grad=True)
            free_energy = self.forward(sequences, bias=bias, nicks=nicks)
            bpp = self._bpp_from_grad(free_energy, bias, create_graph=False)
        return free_energy.detach(), bpp.detach()

    def pair_probabilities(self, sequences: list[str],
                           nicks: list[int] | None = None) -> torch.Tensor:
        """Pair-probability matrix ``(B, N, N)`` for discrete sequences."""
        return self.free_energy_and_bpp(sequences, nicks=nicks)[1]

    def soft_pair_probabilities(self, probs: torch.Tensor,
                                nicks: list[int] | None = None) -> torch.Tensor:
        """Sequence-differentiable pair-probability matrix ``(B, N, N)``.

        The returned tensor is differentiable w.r.t. ``probs`` (second-order
        autograd via ``create_graph=True``), which is what lets BPP-based design
        losses backpropagate to the soft sequence.  Pass ``nicks`` (cumulative
        strand lengths) for a multi-strand complex.
        """
        return self.soft_free_energy_and_bpp(probs, nicks=nicks)[1]

    def soft_free_energy_and_bpp(self, probs: torch.Tensor,
                                 nicks: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(free_energy (B,), bpp (B, N, N))`` for a soft sequence.

        Both outputs are differentiable w.r.t. ``probs`` (the free energy
        directly, the BPP via ``create_graph``) and share a single DP pass — so a
        design objective with both a ΔG term and a defect/accessibility term folds
        the sequence only once per evaluation.

        The internal bias-gradient pass always runs under :func:`torch.enable_grad`
        (the gradient *is* the BPP), so this works even when called inside
        ``torch.no_grad`` for plain scoring.  Second-order graph retention is
        switched on only when ``probs`` itself carries gradients — i.e. when the
        caller intends to backprop a BPP-based loss to the sequence.
        """
        B, N, _ = probs.shape
        create_graph = probs.requires_grad
        with torch.enable_grad():
            bias = torch.zeros(B, N, N, dtype=probs.dtype, device=probs.device,
                               requires_grad=True)
            free_energy = self.soft_forward(probs, bias=bias, nicks=nicks)
            bpp = self._bpp_from_grad(free_energy, bias, create_graph=create_graph)
        if not create_graph:
            free_energy, bpp = free_energy.detach(), bpp.detach()
        return free_energy, bpp


DifferentiableMFE = BatchedPartitionFunction


def seq_to_probs(sequences: list[str], material: str = "rna",
                 device: str | None = None) -> torch.Tensor:
    """One-hot encode sequences to a ``(B, N, 4)`` probability tensor (A,C,G,U/T)."""
    base_to_idx = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
    B = len(sequences)
    N = max(len(s) for s in sequences)
    probs = torch.zeros(B, N, 4, dtype=torch.float64, device=device)
    for b, s in enumerate(sequences):
        for i, ch in enumerate(s.upper()):
            probs[b, i, base_to_idx.get(ch, 0)] = 1.0
    return probs


def soft_free_energy(probs: torch.Tensor, material: str = "rna",
                     params: "ThermoParameters | None" = None) -> torch.Tensor:
    """
    Sequence-differentiable ensemble ΔG for a soft sequence ``probs`` (B,N,4).

    Thin wrapper around :meth:`BatchedPartitionFunction.soft_forward`; the
    autograd graph reaches ``probs`` (and the parameters), enabling
    gradient-based sequence design.  See ``soft_forward`` for the relaxation.
    """
    if params is None:
        params = ThermoParameters(material=material)
    if probs.device != params.stack_table.device:
        params = params.to(probs.device)
    return BatchedPartitionFunction(params).soft_forward(probs)


def pair_probabilities(
    sequences: list[str],
    material: str = "rna",
    device: str | None = None,
    params: "ThermoParameters | None" = None,
) -> torch.Tensor:
    """McCaskill base-pair-probability matrix ``(B, N, N)`` for discrete sequences.

    Computed by autodiff of the ensemble free energy (``∂F/∂ε_ij = P(i,j)``); see
    :meth:`BatchedPartitionFunction.free_energy_and_bpp`.  Detached / inference use.
    """
    if params is None:
        params = ThermoParameters(material=material)
    if device is not None:
        params = params.to(device)
    return BatchedPartitionFunction(params).pair_probabilities(sequences)


def soft_pair_probabilities(
    probs: torch.Tensor,
    material: str = "rna",
    params: "ThermoParameters | None" = None,
) -> torch.Tensor:
    """Sequence-differentiable BPP matrix ``(B, N, N)`` for a soft sequence.

    The result is differentiable w.r.t. ``probs`` — the foundation every
    BPP-based design loss (ensemble defect, accessibility, entropy) is built on.
    """
    if params is None:
        params = ThermoParameters(material=material)
    if probs.device != params.stack_table.device:
        params = params.to(probs.device)
    return BatchedPartitionFunction(params).soft_pair_probabilities(probs)


def batched_free_energy(
    sequences: list[str],
    material: str = "rna",
    device: str | None = None,
    params: "ThermoParameters | None" = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Batched ensemble free energy ΔG (kcal/mol) for many sequences at once.

    This is the differentiable engine's "fast path": the whole batch folds in a
    single vectorised McCaskill DP, so on CPU it runs ~5-12× faster than looping
    the pure-Python native engine, and on a GPU (``device='cuda'``) it scales
    further with batch size — while remaining **learnable** (pass a trained
    :class:`ThermoParameters` to fold with optimised tables, something a closed
    C kernel can never offer).

    Parameters
    ----------
    sequences     : list of DNA/RNA strings (mixed lengths are padded internally).
    material      : ``"rna"`` or ``"dna"`` (ignored if ``params`` is given).
    device        : e.g. ``"cuda"``; defaults to the params' current device.
    params        : optional pre-built / trained ``ThermoParameters``.
    requires_grad : keep the autograd graph (for training / sensitivity); default
                    runs under ``torch.no_grad()`` for inference speed.

    Returns a 1-D ``float64`` tensor of free energies, one per input sequence.
    """
    if params is None:
        params = ThermoParameters(material=material)
    if device is not None:
        params = params.to(device)
    model = BatchedPartitionFunction(params)
    if requires_grad:
        return model(sequences)
    with torch.no_grad():
        return model(sequences)


def concat_with_nicks(strands: list[str]) -> tuple[str, list[int]]:
    """Concatenate strands and return ``(joined, nicks)`` (cumulative lengths)."""
    nicks, pos = [], 0
    for s in strands[:-1]:
        pos += len(s)
        nicks.append(pos)
    return "".join(strands), nicks


def _symmetry_dg(strands: list[str]) -> float:
    """Rotational-symmetry free-energy correction ``+RT·ln σ`` (Dirks 2007).

    The nick-aware DP folds the *ordered* concatenation, so a homomeric complex
    over-counts rotationally by σ; native ``ThermoEngine.pfunc`` applies the same
    correction.  σ depends only on strand identity, so it shifts the free energy
    but leaves pair probabilities (and hence ensemble defect) unchanged.
    """
    try:
        from strider.equilibrium import cyclic_symmetry
        sigma = cyclic_symmetry(list(strands))
    except Exception:
        sigma = 1
    return RT * math.log(sigma) if sigma > 1 else 0.0


def _association_dg(strands: list[str], material: str) -> float:
    """Bimolecular association penalty ``(L−1)·ΔG_assoc`` for a connected complex.

    The nick-aware DP folds the ordered concatenation but does not charge the
    strand-association cost; native ``ThermoEngine.pfunc`` adds it after the DP
    (see ``engine._assoc_correction`` / the ``dG += self._assoc_correction(...)``
    line), so ``complex_free_energy`` must add the same term to match native.
    Each strand beyond the first in the connected complex costs one ``ΔG_assoc``
    (Dirks 2007), sourced from ``parameters_{dna,rna}.JOIN_PENALTY`` (DNA 1.96,
    RNA 4.09 kcal/mol at 37 °C).  Like σ, it is a structure-independent constant,
    so it shifts the free energy without changing pair probabilities or the
    sequence gradients.
    """
    n_assoc = len(strands) - 1
    if n_assoc <= 0:
        return 0.0
    if str(material).lower().startswith("d"):
        from strider.thermo.parameters_dna import JOIN_PENALTY
    else:
        from strider.thermo.parameters_rna import JOIN_PENALTY
    return n_assoc * float(JOIN_PENALTY)


def complex_free_energy(
    strands: list[str],
    material: str = "rna",
    params: "ThermoParameters | None" = None,
    symmetry: bool = True,
) -> torch.Tensor:
    """Ensemble free energy ΔG (kcal/mol) of a multi-strand complex.

    Folds the nick-aware concatenation and applies the same post-DP corrections
    native ``ThermoEngine.pfunc(*strands)`` applies: the rotational symmetry term
    (``+RT·ln σ``, by default) and the bimolecular association penalty
    ``(L−1)·ΔG_assoc``.  Both are structure-independent constants, so they shift
    the free energy to match native without perturbing the pair probabilities.
    The small coaxial-stacking correction native also adds (from the dominant
    structure) is left out; it stays within the engine residual (< 1 kcal/mol).
    """
    if params is None:
        params = ThermoParameters(material=material)
    model = BatchedPartitionFunction(params)
    seq, nicks = concat_with_nicks(strands)
    with torch.no_grad():
        fe = model([seq], nicks=nicks)
    if symmetry:
        fe = fe + _symmetry_dg(strands)
    fe = fe + _association_dg(strands, getattr(params, "material", material))
    return fe


def complex_pair_probabilities(
    strands: list[str],
    material: str = "rna",
    params: "ThermoParameters | None" = None,
) -> torch.Tensor:
    """Pair-probability matrix ``(N, N)`` of a multi-strand complex (N = total length).

    Symmetry-independent (σ cancels in the BPP), so no correction is applied.
    """
    if params is None:
        params = ThermoParameters(material=material)
    model = BatchedPartitionFunction(params)
    seq, nicks = concat_with_nicks(strands)
    return model.pair_probabilities([seq], nicks=nicks)[0]