#!/usr/bin/env python3
"""Oligo thermodynamics engine parity benchmark.

Generates 1000 random DNA oligos (18-24 nt, max 3 consecutive identical bases)
and computes four metrics from each of five engine variants:

  1. Duplex Tm  (Tm of the oligo annealing to its perfect complement)
  2. Hairpin dG (at 25 C)
  3. Hairpin Tm
  4. Homodimer dG (at 25 C)

Engines:
  - Strider_M:  Strider with mathews2004-dna parameters, dangles=2, applied
                uniformly to all four metrics (duplex/homodimer Tm via the
                dimer_thermo() two-state solver so the NN table is honored).
  - Strider_SL: Strider with the built-in "native" parameters (SantaLucia
                2004 DNA), same pipeline/salt handling as Strider_M so the
                two columns differ only in NN table.
  - primer3:    SantaLucia 2004 (default).
  - ViennaRNA:  DNA Mathews 2004; Tm derived via two-temperature dH/dS.
  - IDT:        OligoAnalyzer REST API (phase 2, needs credentials).

Conditions (all engines): Na=50 mM, Mg=3 mM, dNTP=0, OligoConc=0.25 uM.
dG reported at 25 C.

dG convention: strider, primer3, and ViennaRNA report raw dG25 including the
bimolecular initiation term. IDT reports structure-only dG without initiation.
A ~2 kcal/mol systematic offset is expected between IDT and the other three.

ViennaRNA has no native two-state Tm. Tm is derived by evaluating the MFE
structure at 25 C and 65 C, solving for dH and dS, then:
  - Hairpin (unimolecular):   Tm = dH / dS - 273.15
  - Duplex  (bimolecular):    Tm = dH / (dS + R*ln(CT/4)) - 273.15
  (self-complementary duplex uses ln(CT) instead of ln(CT/4))

Output: evidence/oligo_engine_parity.csv (21 columns: seq + 4 per engine variant).

Usage:
  python scripts/oligo_engine_parity.py              # phase 1 (local engines)
  python scripts/oligo_engine_parity.py --idt         # phase 2 (IDT, needs creds)
  python scripts/oligo_engine_parity.py --idt --resume  # resume IDT from checkpoint
  python scripts/oligo_engine_parity.py --refresh-strider  # recompute only the
      # strider_m / strider_sl columns of the existing CSV in place (keeps the
      # primer3 / vienna / idt columns and the row set), e.g. after a code or
      # parameter change on the strider side
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oligo_parity")

# ── Constants ────────────────────────────────────────────────────────────────

SEED = 42
N_OLIGOS = 1000
MIN_LEN = 18
MAX_LEN = 24
MAX_RUN = 3  # max consecutive identical nucleotides

# Salt/concentration conditions (shared across all engines)
MV_CONC = 50.0       # mM monovalent (Na+)
DV_CONC = 3.0         # mM divalent (Mg2+)
DNTP_CONC = 0.0       # mM
DNA_CONC_NM = 250.0   # nM strand concentration (= 0.25 uM)
OLIGO_CONC_M = 0.25e-6  # M

# von Ahsen (2001) sodium-equivalent: Na_eq = Na + 120*sqrt(free_Mg)
FREE_MG_MM = max(0.0, DV_CONC - DNTP_CONC)  # 3.0 mM
NA_EQ_M = (MV_CONC + 120.0 * math.sqrt(FREE_MG_MM)) / 1000.0  # ~0.2578 M
NA_M = MV_CONC / 1000.0     # 0.05 M (for strider's sodium_M param)
MG_M = FREE_MG_MM / 1000.0  # 0.003 M

# Reference temperature for dG
T_REF_C = 25.0
T_REF_K = T_REF_C + 273.15

# ViennaRNA two-temperature Tm derivation
VR_T1_C = 25.0
VR_T2_C = 65.0
VR_T1_K = VR_T1_C + 273.15
VR_T2_K = VR_T2_C + 273.15
R_KCAL = 1.987e-3  # kcal/(mol*K)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Prefer the local strider source over any stale globally-installed copy,
# regardless of cwd or how this script is invoked (python vs `python path/to/script.py`).
sys.path.insert(0, str(REPO_ROOT))
EVIDENCE_DIR = REPO_ROOT / "evidence"
CSV_PATH = EVIDENCE_DIR / "oligo_engine_parity.csv"
IDT_CHECKPOINT = EVIDENCE_DIR / "oligo_engine_parity_idt.jsonl"

ENGINES = ("strider_m", "strider_sl", "primer3", "vienna", "idt")

COLUMNS = [
    "seq",
    "strider_m_duplex_tm", "strider_m_hairpin_dg", "strider_m_hairpin_tm", "strider_m_homodimer_dg",
    "strider_sl_duplex_tm", "strider_sl_hairpin_dg", "strider_sl_hairpin_tm", "strider_sl_homodimer_dg",
    "primer3_duplex_tm", "primer3_hairpin_dg", "primer3_hairpin_tm", "primer3_homodimer_dg",
    "vienna_duplex_tm", "vienna_hairpin_dg", "vienna_hairpin_tm", "vienna_homodimer_dg",
    "idt_duplex_tm", "idt_hairpin_dg", "idt_hairpin_tm", "idt_homodimer_dg",
]

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


# ── Oligo generation ─────────────────────────────────────────────────────────

def generate_oligos(n: int = N_OLIGOS, seed: int = SEED) -> list[str]:
    """Generate n random DNA oligos (18-24 nt, max 3 consecutive identical bases)."""
    rng = random.Random(seed)
    bases = "ACGT"
    oligos: list[str] = []
    for _ in range(n):
        length = rng.randint(MIN_LEN, MAX_LEN)
        chars: list[str] = []
        run = 0
        prev: str | None = None
        for _ in range(length):
            if run >= MAX_RUN and prev is not None:
                choices = [b for b in bases if b != prev]
            else:
                choices = list(bases)
            base = rng.choice(choices)
            if base == prev:
                run += 1
            else:
                run = 1
            prev = base
            chars.append(base)
        oligos.append("".join(chars))
    return oligos


# ── Strider ──────────────────────────────────────────────────────────────────
#
# Both variants run the identical pipeline (dimer_thermo two-state solver for
# duplex/homodimer, hairpin_thermo + ThermoEngine.mfe for hairpin) and differ
# only in which NN parameter set is passed in, so the _m/_sl columns are a
# clean parameter-set comparison rather than a mix of methods.

def _strider_init():
    """Lazily import strider and build one (engine, paramset) pair per variant."""
    from strider import ThermoEngine
    from strider.thermo.hairpin import hairpin_thermo
    from strider.thermo.dimer_thermo import dimer_thermo
    from strider.thermo.parameters import load_parameters

    variants = {}
    for key, param_name in (("m", "mathews2004-dna"), ("sl", "native")):
        ps = load_parameters(param_name)
        eng = ThermoEngine(
            material="dna", celsius=T_REF_C,
            sodium=NA_M, magnesium=MG_M,
            parameter_set=ps, dangles=2,
        )
        variants[key] = (eng, ps)
    return hairpin_thermo, dimer_thermo, variants


_strider_cache = None


def _strider_variant_metrics(seq: str, eng, ps, hairpin_thermo, dimer_thermo) -> tuple:
    """Return (duplex_tm, hairpin_dg, hairpin_tm, homodimer_dg) for one paramset."""
    rc = _revcomp(seq)

    # Duplex Tm: two-state solver against the perfect complement, so the NN
    # table (ps) is actually honored (ThermoEngine.melting_temperature() is
    # hardcoded to the SantaLucia nn_dna module regardless of parameter_set).
    try:
        res = dimer_thermo(
            seq, rc, sodium_M=NA_M, magnesium_M=MG_M, material="dna",
            strand_conc_M=OLIGO_CONC_M, salt_model="auto",
            paramset=ps, dangles=2,
        )
        duplex_tm = res.tm_celsius
    except (ValueError, Exception):
        duplex_tm = None

    # Hairpin dG from MFE at 25 C
    mfe = eng.mfe(seq)
    if mfe.structure and "(" in mfe.structure:
        hairpin_dg = float(mfe.energy)
    else:
        hairpin_dg = 0.0

    # Hairpin Tm from two-state model (may fail for multiloops)
    try:
        res = hairpin_thermo(
            seq, sodium_M=NA_M, magnesium_M=MG_M, material="dna",
            paramset=ps, dangles=2,
        )
        hairpin_tm = res.tm_celsius
    except (ValueError, Exception):
        hairpin_tm = None

    # Homodimer dG
    try:
        res = dimer_thermo(
            seq, seq, sodium_M=NA_M, magnesium_M=MG_M, material="dna",
            strand_conc_M=OLIGO_CONC_M, salt_model="auto",
            paramset=ps, dangles=2,
        )
        homodimer_dg = res.dH - T_REF_K * res.dS / 1000.0
    except (ValueError, Exception):
        homodimer_dg = 0.0

    return duplex_tm, hairpin_dg, hairpin_tm, homodimer_dg


def strider_m_metrics(seq: str) -> tuple:
    """Return (duplex_tm, hairpin_dg, hairpin_tm, homodimer_dg) from Strider (Mathews 2004)."""
    global _strider_cache
    if _strider_cache is None:
        _strider_cache = _strider_init()
    hairpin_thermo, dimer_thermo, variants = _strider_cache
    eng, ps = variants["m"]
    return _strider_variant_metrics(seq, eng, ps, hairpin_thermo, dimer_thermo)


def strider_sl_metrics(seq: str) -> tuple:
    """Return (duplex_tm, hairpin_dg, hairpin_tm, homodimer_dg) from Strider (SantaLucia 2004)."""
    global _strider_cache
    if _strider_cache is None:
        _strider_cache = _strider_init()
    hairpin_thermo, dimer_thermo, variants = _strider_cache
    eng, ps = variants["sl"]
    return _strider_variant_metrics(seq, eng, ps, hairpin_thermo, dimer_thermo)


# ── primer3 ───────────────────────────────────────────────────────────────────

def _primer3_init():
    import primer3
    return primer3


_p3_cache = None


def primer3_metrics(seq: str) -> tuple:
    """Return (duplex_tm, hairpin_dg, hairpin_tm, homodimer_dg) from primer3."""
    global _p3_cache
    if _p3_cache is None:
        _p3_cache = _primer3_init()
    p3 = _p3_cache

    salt = {
        "mv_conc": MV_CONC, "dv_conc": DV_CONC,
        "dntp_conc": DNTP_CONC, "dna_conc": DNA_CONC_NM,
    }
    tm_kw = {**salt, "tm_method": "santalucia", "salt_corrections_method": "santalucia"}

    duplex_tm = p3.calc_tm(seq, **tm_kw)

    hp = p3.calc_hairpin(seq, temp_c=T_REF_C, **salt)
    hairpin_dg = (hp.dg or 0.0) / 1000.0  # cal/mol -> kcal/mol
    hairpin_tm = hp.tm if hp.structure_found else None

    hd = p3.calc_homodimer(seq, temp_c=T_REF_C, **salt)
    homodimer_dg = (hd.dg or 0.0) / 1000.0

    return duplex_tm, hairpin_dg, hairpin_tm, homodimer_dg


# ── ViennaRNA ─────────────────────────────────────────────────────────────────

def _vienna_init():
    import RNA
    RNA.params_load_DNA_Mathews2004()
    return RNA


_vr_cache = None


def _vr_md(rna, temp_c: float):
    md = rna.md()
    md.temperature = temp_c
    md.dna = 1
    md.salt = NA_EQ_M
    return md


def vienna_metrics(seq: str) -> tuple:
    """Return (duplex_tm, hairpin_dg, hairpin_tm, homodimer_dg) from ViennaRNA."""
    global _vr_cache
    if _vr_cache is None:
        _vr_cache = _vienna_init()
    RNA = _vr_cache

    md25 = _vr_md(RNA, VR_T1_C)
    md65 = _vr_md(RNA, VR_T2_C)

    # Hairpin dG (MFE at 25 C)
    fc25 = RNA.fold_compound(seq, md25)
    ss, dg25 = fc25.mfe()
    hairpin_dg = float(dg25)

    # Hairpin Tm (unimolecular: Tm = dH/dS, no concentration term)
    hairpin_tm = None
    if "(" in ss:
        g1 = fc25.eval_structure(ss)
        g2 = RNA.fold_compound(seq, md65).eval_structure(ss)
        dS = (g1 - g2) / (VR_T2_K - VR_T1_K)
        dH = g1 + VR_T1_K * dS
        if abs(dS) > 1e-10:
            hairpin_tm = dH / dS - 273.15

    # Duplex Tm (perfect duplex: seq vs revcomp, bimolecular two-state)
    rc = _revcomp(seq)
    combined = seq + "&" + rc
    struct = "(" * len(seq) + ")" * len(rc)
    g1d = RNA.fold_compound(combined, md25).eval_structure(struct)
    g2d = RNA.fold_compound(combined, md65).eval_structure(struct)
    dSd = (g1d - g2d) / (VR_T2_K - VR_T1_K)
    dHd = g1d + VR_T1_K * dSd
    self_comp = (seq == rc)
    ln_term = math.log(OLIGO_CONC_M) if self_comp else math.log(OLIGO_CONC_M / 4.0)
    duplex_tm = dHd / (dSd + R_KCAL * ln_term) - 273.15

    # Homodimer dG (cofold MFE at 25 C)
    fc_dim = RNA.fold_compound(seq + "&" + seq, md25)
    _, dg_dim = fc_dim.mfe()
    homodimer_dg = float(dg_dim)

    return duplex_tm, hairpin_dg, hairpin_tm, homodimer_dg


# ── IDT (phase 2) ────────────────────────────────────────────────────────────

def _idt_host(region: str) -> str:
    return "www.idtdna.com" if region.lower() == "us" else "eu.idtdna.com"


def get_idt_token(client_id: str, client_secret: str,
                  username: str, password: str, region: str = "eu") -> str:
    """Obtain an IDT OAuth bearer token (mirrors Oligool's /idt/token)."""
    import base64
    import requests
    host = _idt_host(region)
    url = f"https://{host}/IdentityServer/connect/token"
    auth_bytes = f"{client_id}:{client_secret}".encode("utf-8")
    auth_string = base64.b64encode(auth_bytes).decode("ascii")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": "Basic " + auth_string,
        "Accept": "application/json",
    }
    data = {
        "grant_type": "password",
        "scope": "test",
        "username": username,
        "password": password,
    }
    resp = requests.post(url, data=data, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()["access_token"]


def _extract_idt_dg(obj) -> float | None:
    """Extract DeltaG from an IDT response (list or dict), filtering sentinels.

    For list responses (Hairpin, SelfDimer), returns the most-negative DeltaG
    across all returned structures (the MFE).
    """
    if isinstance(obj, list):
        best = None
        for item in obj:
            val = _extract_idt_dg(item)
            if val is not None and (best is None or val < best):
                best = val
        return best
    if not isinstance(obj, dict):
        return None
    for k in ("DeltaG", "deltaG", "deltag", "delta_g", "dG", "Energy", "energy"):
        if k in obj:
            try:
                val = float(obj[k])
                if -200.0 < val < 50.0:
                    return val
            except (ValueError, TypeError):
                pass
    return None


def _extract_idt_tm(obj) -> float | None:
    """Extract Tm from an IDT response (list or dict), filtering sentinels.

    For list responses (Hairpin), returns the Tm from the first (MFE) element.
    """
    if isinstance(obj, list):
        for item in obj:
            val = _extract_idt_tm(item)
            if val is not None:
                return val
        return None
    if not isinstance(obj, dict):
        return None
    for k in ("Tm", "tm", "MeltingTemperature", "MeltTemp", "MeltingTemp",
              "meltingTemp", "thermo"):
        if k in obj:
            try:
                val = float(obj[k])
                if -100.0 < val < 200.0:
                    return val
            except (ValueError, TypeError):
                pass
    return None


class TokenExpired(Exception):
    """Raised when IDT returns 401, signalling the OAuth token expired."""


def idt_metrics(seq: str, token: str, region: str = "eu") -> tuple:
    """Return (duplex_tm, hairpin_dg, hairpin_tm, homodimer_dg) from IDT REST API.

    Raises TokenExpired on 401 so the caller can re-authenticate.
    """
    import requests
    host = _idt_host(region)
    base_url = f"https://{host}/restapi/v1/OligoAnalyzer"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    salt_payload = {
        "NaConc": MV_CONC, "MgConc": DV_CONC,
        "dNTPsConc": DNTP_CONC, "OligoConc": OLIGO_CONC_M * 1e6,
        "NucleotideType": "DNA",
    }

    def _post(endpoint: str, payload: dict, params: dict | None = None) -> dict:
        url = f"{base_url}/{endpoint}"
        for attempt in range(3):
            resp = requests.post(url, json=payload, params=params,
                                 headers=headers, timeout=30)
            if resp.ok:
                return resp.json()
            if resp.status_code == 401:
                raise TokenExpired(f"IDT {endpoint} returned 401")
            if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(2.0 ** attempt)
                continue
            log.warning("IDT %s error %d: %s", endpoint, resp.status_code, resp.text[:200])
            return {}
        return {}

    # Analyze (duplex Tm)
    analyze_payload = {**salt_payload, "Sequence": seq}
    analyze_res = _post("Analyze", analyze_payload)
    duplex_tm = _extract_idt_tm(analyze_res)

    # Hairpin (dG + Tm)
    hp_payload = {**salt_payload, "Sequence": seq, "FoldingTemp": T_REF_C}
    hp_res = _post("Hairpin", hp_payload)
    hairpin_dg = _extract_idt_dg(hp_res)
    hairpin_tm = _extract_idt_tm(hp_res)

    # SelfDimer (homodimer dG)
    sd_params = {"primary": seq, "secondary": seq}
    sd_res = _post("SelfDimer", salt_payload, params=sd_params)
    homodimer_dg = _extract_idt_dg(sd_res)

    return duplex_tm, hairpin_dg, hairpin_tm, homodimer_dg


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def _fmt(x) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.3f}"
    except (ValueError, TypeError):
        return ""


def _fmt_tm(x) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.2f}"
    except (ValueError, TypeError):
        return ""


def write_csv(rows: list[dict], path: Path = CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            out = {"seq": row["seq"]}
            for eng in ENGINES:
                out[f"{eng}_duplex_tm"] = _fmt_tm(row.get(f"{eng}_duplex_tm"))
                out[f"{eng}_hairpin_dg"] = _fmt(row.get(f"{eng}_hairpin_dg"))
                out[f"{eng}_hairpin_tm"] = _fmt_tm(row.get(f"{eng}_hairpin_tm"))
                out[f"{eng}_homodimer_dg"] = _fmt(row.get(f"{eng}_homodimer_dg"))
            writer.writerow(out)
    log.info("CSV written to %s (%d rows)", path, len(rows))


def read_csv(path: Path = CSV_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


# ── Phase 1: local engines ───────────────────────────────────────────────────

def run_phase1(output: Path = CSV_PATH, n_oligos: int = N_OLIGOS) -> None:
    oligos = generate_oligos(n_oligos)
    log.info("Generated %d oligos (seed=%d, len %d-%d, max run %d)",
             len(oligos), SEED, MIN_LEN, MAX_LEN, MAX_RUN)

    rows: list[dict] = []
    t0 = time.time()
    for i, seq in enumerate(oligos):
        row: dict = {"seq": seq}
        for eng, fn in (("strider_m", strider_m_metrics),
                        ("strider_sl", strider_sl_metrics),
                        ("primer3", primer3_metrics),
                        ("vienna", vienna_metrics)):
            try:
                tm, hdg, htm, oddg = fn(seq)
            except Exception as e:
                log.warning("%s failed for %s: %s", eng, seq, e)
                tm, hdg, htm, oddg = None, None, None, None
            row[f"{eng}_duplex_tm"] = tm
            row[f"{eng}_hairpin_dg"] = hdg
            row[f"{eng}_hairpin_tm"] = htm
            row[f"{eng}_homodimer_dg"] = oddg
        rows.append(row)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            log.info("  %d/%d (%.1fs, %.1fms/oligo)", i + 1, len(oligos),
                     elapsed, elapsed * 1000 / (i + 1))

    write_csv(rows, output)
    elapsed = time.time() - t0
    log.info("Phase 1 complete: %d oligos in %.1fs", len(oligos), elapsed)


# ── Phase 2: IDT ──────────────────────────────────────────────────────────────

def run_refresh_strider(csv_path: Path = CSV_PATH) -> None:
    """Recompute the strider_m / strider_sl columns of an existing CSV in place.

    The row set (same seed) and every other engine's columns are preserved,
    so a strider-side change can be re-measured without re-querying IDT.
    """
    rows = read_csv(csv_path)
    if not rows:
        log.error("No CSV found at %s. Run phase 1 first.", csv_path)
        sys.exit(1)

    hairpin_thermo, dimer_thermo, variants = _strider_init()
    t0 = time.time()
    for i, row in enumerate(rows):
        for key in ("m", "sl"):
            eng, ps = variants[key]
            try:
                tm, hdg, htm, oddg = _strider_variant_metrics(
                    row["seq"], eng, ps, hairpin_thermo, dimer_thermo)
            except Exception as e:
                log.warning("strider_%s failed for %s: %s", key, row["seq"], e)
                tm, hdg, htm, oddg = None, None, None, None
            row[f"strider_{key}_duplex_tm"] = tm
            row[f"strider_{key}_hairpin_dg"] = hdg
            row[f"strider_{key}_hairpin_tm"] = htm
            row[f"strider_{key}_homodimer_dg"] = oddg
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            log.info("  %d/%d (%.1fs, %.1fms/oligo)", i + 1, len(rows),
                     elapsed, elapsed * 1000 / (i + 1))

    write_csv(rows, csv_path)
    log.info("Strider columns refreshed: %d oligos in %.1fs", len(rows), time.time() - t0)



def run_phase2(resume: bool = False, csv_path: Path = CSV_PATH) -> None:
    import requests

    # Gather credentials
    region = os.environ.get("IDT_REGION", "eu")
    token = os.environ.get("IDT_TOKEN")
    client_id = os.environ.get("IDT_CLIENT_ID", "")
    client_secret = os.environ.get("IDT_CLIENT_SECRET", "")
    username = os.environ.get("IDT_USERNAME", "")
    password = os.environ.get("IDT_PASSWORD", "")
    can_reauth = all([client_id, client_secret, username, password])

    if not token and not can_reauth:
        log.error("IDT credentials not found. Set IDT_TOKEN or "
                  "IDT_CLIENT_ID/IDT_CLIENT_SECRET/IDT_USERNAME/IDT_PASSWORD "
                  "(and optionally IDT_REGION).")
        sys.exit(1)

    if not token:
        log.info("Requesting IDT OAuth token for user %s on region %s", username, region)
        token = get_idt_token(client_id, client_secret, username, password, region)
        log.info("IDT token obtained")

    rows = read_csv(csv_path)
    if not rows:
        log.error("No CSV found at %s. Run phase 1 first.", csv_path)
        sys.exit(1)

    # Load checkpoint, skipping entries where all metrics are None (token-expired
    # placeholders from a previous aborted run)
    done: dict[str, dict] = {}
    if resume and IDT_CHECKPOINT.exists():
        skipped = 0
        with open(IDT_CHECKPOINT) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("duplex_tm") is None and rec.get("hairpin_dg") is None \
                   and rec.get("homodimer_dg") is None:
                    skipped += 1
                    continue
                done[rec["seq"]] = rec
        if skipped:
            log.info("Skipped %d all-None checkpoint entries (token-expired placeholders)", skipped)
        log.info("Resumed from checkpoint: %d/%d IDT entries already done",
                 len(done), len(rows))

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_fh = open(IDT_CHECKPOINT, "a", encoding="utf-8")

    def _reauth():
        nonlocal token
        if not can_reauth:
            raise TokenExpired("Token expired and no credentials available to re-auth")
        log.info("IDT token expired, re-authenticating...")
        token = get_idt_token(client_id, client_secret, username, password, region)
        log.info("IDT token re-obtained")

    t0 = time.time()
    n_done = len(done)
    n_total = len(rows)
    for i, row in enumerate(rows):
        seq = row["seq"]
        if seq in done:
            rec = done[seq]
        else:
            for attempt in range(3):
                try:
                    tm, hdg, htm, oddg = idt_metrics(seq, token, region)
                    break
                except TokenExpired:
                    _reauth()
                    continue
                except Exception as e:
                    if attempt < 2:
                        log.warning("IDT call failed for %s (attempt %d): %s", seq, attempt + 1, e)
                        time.sleep(2.0 ** attempt)
                    else:
                        log.error("IDT gave up on %s: %s", seq, e)
                        tm, hdg, htm, oddg = None, None, None, None
            else:
                tm, hdg, htm, oddg = None, None, None, None
            rec = {"seq": seq, "duplex_tm": tm, "hairpin_dg": hdg,
                   "hairpin_tm": htm, "homodimer_dg": oddg}
            checkpoint_fh.write(json.dumps(rec) + "\n")
            checkpoint_fh.flush()
            done[seq] = rec
            n_done += 1

        row["idt_duplex_tm"] = rec["duplex_tm"]
        row["idt_hairpin_dg"] = rec["hairpin_dg"]
        row["idt_hairpin_tm"] = rec["hairpin_tm"]
        row["idt_homodimer_dg"] = rec["homodimer_dg"]

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (n_done - (len(done) - len(rows) if resume else 0)) / max(elapsed, 1)
            eta = (n_total - i - 1) / max(rate, 0.01)
            log.info("  %d/%d IDT done (%.1fs, ETA %.0fs)", i + 1, n_total, elapsed, eta)

    checkpoint_fh.close()

    # Rewrite CSV with IDT columns filled
    write_csv(rows, csv_path)
    log.info("Phase 2 complete: %d oligos in %.1fs", len(rows), time.time() - t0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--idt", action="store_true",
                        help="Run IDT phase 2 (needs IDT credentials in env).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume IDT phase from checkpoint.")
    parser.add_argument("--n", type=int, default=N_OLIGOS,
                        help="Number of oligos (default: %d)." % N_OLIGOS)
    parser.add_argument("--output", type=Path, default=CSV_PATH,
                        help="Output CSV path.")
    parser.add_argument("--refresh-strider", action="store_true",
                        help="Recompute only the strider columns of the existing CSV in place.")
    args = parser.parse_args()

    if args.refresh_strider:
        run_refresh_strider(csv_path=args.output)
    elif args.idt:
        run_phase2(resume=args.resume, csv_path=args.output)
    else:
        run_phase1(output=args.output, n_oligos=args.n)


if __name__ == "__main__":
    main()
