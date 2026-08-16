//! Rust port of `strider.thermo.dimer_thermo._dimer_mfe_candidates` (DNA).
//!
//! The dynamic program enumerates every antiparallel inter-strand helix start
//! state and ranks it by closed-state free energy — identical recurrence,
//! identical tables, identical tie-break order as the Python implementation:
//! stable sort by (energy asc, −number of base pairs).  Table data comes from
//! `tables::dna` (generated, bit-identical to `strider.thermo.parameters_dna`),
//! so result equivalence is mechanical, given the float ops follow the same
//! order — which they do, statement by statement.

use crate::tables::dna::*;
use crate::tables::{lookup, pack};

const INF: f64 = f64::INFINITY;
const MAX_LOOP: usize = 4;

// ─── small per-key helpers (mirroring strider.thermo.ensemble) ──────────────

#[inline]
fn norm(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'U' => b'T',
        other => other,
    }
}

/// DNA Watson–Crick pair check (`_wc_pairs('dna')`).
#[inline]
fn can_pair(a: u8, b: u8) -> bool {
    matches!(
        (norm(a), norm(b)),
        (b'A', b'T') | (b'T', b'A') | (b'G', b'C') | (b'C', b'G')
    )
}

/// `_terminal_pair_penalty`: TERMINAL_PENALTY.get(seq[i]+seq[j], 0.0)
#[inline]
fn terminal_penalty(a: u8, b: u8) -> f64 {
    lookup(TERMINAL_PENALTY, pack(&[norm(a), norm(b)])).unwrap_or(0.0)
}

// ─── the DP itself ───────────────────────────────────────────────────────────

/// Direct port of `_dimer_mfe_candidates` (DNA). `seq1`/`seq2` raw bytes;
/// normalization (uppercase, U→T) applied per access, matching the Python
/// `.upper().replace("U", "T")` normalization of both strands.
pub fn dimer_mfe_candidates_dna(seq1: &[u8], seq2: &[u8]) -> Vec<(f64, Vec<(usize, usize)>)> {
    let n1 = seq1.len();
    let n2 = seq2.len();
    if n1 == 0 || n2 == 0 {
        return Vec::new();
    }
    let n = n1 + n2;

    // Concatenated sequence view: helper returning normalized base at concat idx.
    let at = |k: usize| -> u8 {
        if k < n1 {
            norm(seq1[k])
        } else {
            norm(seq2[k - n1])
        }
    };

    let cp = |i: usize, j_loc: usize| can_pair(seq1[i], seq2[j_loc]);

    let mut inner = vec![INF; n1 * n2];
    // Trace: None = stop here (terminal pair at this cell), Some((ip, jp_loc)).
    let mut trace: Vec<Option<(usize, usize)>> = vec![None; n1 * n2];

    let idx = |i: usize, j_loc: usize| i * n2 + j_loc;

    // Dangle helpers — identical indexing/guards to the Python lambdas.
    let dangle5 = |abc: [u8; 3]| -> f64 {
        match lookup(DANGLE_5, pack(&[norm(abc[0]), norm(abc[1]), norm(abc[2])])) {
            Some(v) if v < 0.0 => v,
            _ => 0.0,
        }
    };
    let dangle3 = |abc: [u8; 3]| -> f64 {
        match lookup(DANGLE_3, pack(&[norm(abc[0]), norm(abc[1]), norm(abc[2])])) {
            Some(v) if v < 0.0 => v,
            _ => 0.0,
        }
    };

    let inner_dangles = |i: usize, j_loc: usize| -> f64 {
        let j_concat = n1 + j_loc;
        let mut total = 0.0;
        if j_concat >= n1 + 1 {
            // Python: if j_concat - 1 >= n1
            total += dangle5([at(j_concat), at(i), at(j_concat - 1)]);
        }
        if i >= 1 && i + 1 < n1 {
            total += dangle3([at(i - 1), at(i), at(i + 1)]);
        }
        total
    };
    let outer_dangles = |i: usize, j_loc: usize| -> f64 {
        let j_concat = n1 + j_loc;
        let mut total = 0.0;
        if i >= 1 {
            total += dangle5([at(i), at(j_concat), at(i - 1)]);
        }
        if j_concat + 1 < n {
            total += dangle3([at(j_concat - 1), at(j_concat), at(j_concat + 1)]);
        }
        total
    };

    // Fill inner[i][j_loc] from inside out (i decreasing, j_loc increasing).
    for i in (0..n1).rev() {
        for j_loc in 0..n2 {
            if !cp(i, j_loc) {
                continue;
            }
            let j_concat = n1 + j_loc;

            let stop_val = terminal_penalty(at(i), at(j_concat)) + inner_dangles(i, j_loc);

            let mut best_continue = INF;
            let mut best_next: Option<(usize, usize)> = None;

            for nl in 0..=MAX_LOOP {
                for nr in 0..=(MAX_LOOP - nl) {
                    let ip = i + 1 + nl;
                    if j_loc < 1 + nr {
                        continue;
                    }
                    let jp_loc = j_loc - 1 - nr;
                    if ip >= n1 || !cp(ip, jp_loc) {
                        continue;
                    }
                    let inner_next = inner[idx(ip, jp_loc)];
                    if inner_next == INF {
                        continue;
                    }
                    let e = if nl == 0 && nr == 0 {
                        stack_energy_c(&at, i, j_concat)
                    } else {
                        interior_c(&at, i, j_concat, ip, n1 + jp_loc, nl, nr)
                    };
                    let cand = e + inner_next;
                    if cand < best_continue {
                        best_continue = cand;
                        best_next = Some((ip, jp_loc));
                    }
                }
            }

            if best_next.is_none() || stop_val <= best_continue {
                inner[idx(i, j_loc)] = stop_val;
            } else {
                inner[idx(i, j_loc)] = best_continue;
                trace[idx(i, j_loc)] = best_next;
            }
        }
    }

    // Collect one candidate per outer (i, j_loc) with a finite closed-state energy.
    let mut candidates: Vec<(f64, Vec<(usize, usize)>)> = Vec::new();
    for i in 0..n1 {
        for j_loc in 0..n2 {
            if !cp(i, j_loc) {
                continue;
            }
            let j_concat = n1 + j_loc;
            let outer_val =
                terminal_penalty(at(i), at(j_concat)) + outer_dangles(i, j_loc) + inner[idx(i, j_loc)];
            if !outer_val.is_finite() {
                continue;
            }
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            let (mut ci, mut cj_loc) = (i, j_loc);
            loop {
                pairs.push((ci, n1 + cj_loc));
                match trace[idx(ci, cj_loc)] {
                    None => break,
                    Some((ip, jp)) => {
                        ci = ip;
                        cj_loc = jp;
                    }
                }
            }
            candidates.push((outer_val, pairs));
        }
    }

    // Python: candidates.sort(key=lambda x: (x[0], -len(x[1])))  — stable.
    candidates.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(b.1.len().cmp(&a.1.len()))
    });
    candidates
}

// Helpers that take the concat accessor (keeps the DP body readable).
#[inline]
fn stack_energy_c(at: &dyn Fn(usize) -> u8, i: usize, j: usize) -> f64 {
    let key = [at(i), at(i + 1), at(j - 1), at(j)];
    lookup(STACK, pack(&key)).unwrap_or(-1.5)
}

fn interior_c(
    at: &dyn Fn(usize) -> u8,
    i: usize,
    j: usize,
    ip: usize,
    jp: usize,
    nl: usize,
    nr: usize,
) -> f64 {
    let tp_outer = terminal_penalty(at(i), at(j));
    let tp_inner = terminal_penalty(at(ip), at(jp));

    if nl == 0 || nr == 0 {
        let n = nl + nr;
        let dg = if n <= 30 {
            BULGE_SIZE[n - 1]
        } else {
            BULGE_SIZE[BULGE_SIZE.len() - 1] + LOG_LOOP_PENALTY * (n as f64 / 30.0).ln()
        };
        if n == 1 {
            let key = [at(i), at(ip), at(jp), at(j)];
            let dg = dg + lookup(STACK, pack(&key)).unwrap_or(0.0);
            dg + tp_outer - tp_inner
        } else {
            dg + 2.0 * tp_outer
        }
    } else {
        if nl == 1 && nr == 1 {
            let key = [at(i), at(i + 1), at(ip), at(jp), at(j - 1), at(j)];
            if let Some(val) = lookup(INTERIOR_1_1, pack(&key)) {
                return tp_outer + val - tp_inner;
            }
        }
        if nl == 1 && nr == 2 {
            let key = [at(i), at(i + 1), at(ip), at(jp), at(jp + 1), at(j - 1), at(j)];
            if let Some(val) = lookup(INTERIOR_1_2, pack(&key)) {
                return tp_outer + val - tp_inner;
            }
        }
        if nl == 2 && nr == 1 {
            let key = [at(jp), at(j - 1), at(j), at(i), at(i + 1), at(i + 2), at(ip)];
            if let Some(val) = lookup(INTERIOR_1_2, pack(&key)) {
                return tp_outer + val - tp_inner;
            }
        }
        if nl == 2 && nr == 2 {
            let key = [
                at(i),
                at(i + 1),
                at(i + 2),
                at(ip),
                at(jp),
                at(jp + 1),
                at(j - 1),
                at(j),
            ];
            if let Some(val) = lookup(INTERIOR_2_2, pack(&key)) {
                return tp_outer + val - tp_inner;
            }
        }

        let n = nl + nr;
        let mut dg = if n <= 30 {
            INTERIOR_SIZE[n - 1]
        } else {
            INTERIOR_SIZE[INTERIOR_SIZE.len() - 1] + LOG_LOOP_PENALTY * (n as f64 / 30.0).ln()
        };
        let asym = nl.abs_diff(nr) as f64;
        let ninio_number = nl.min(nr).min(4) - 1;
        dg += ASYMMETRY_NINIO[4].min(asym * ASYMMETRY_NINIO[ninio_number]);

        let (outer_key, inner_key): ([u8; 4], [u8; 4]) =
            if (nl == 1 && nr > 2) || (nl > 2 && nr == 1) {
                (
                    [b'A', at(j), at(i), b'A'],
                    [b'A', at(ip), at(jp), b'A'],
                )
            } else {
                (
                    [at(j - 1), at(j), at(i), at(i + 1)],
                    [at(ip - 1), at(ip), at(jp), at(jp + 1)],
                )
            };
        dg += lookup(INTERIOR_MISMATCH, pack(&outer_key)).unwrap_or(0.0);
        dg += lookup(INTERIOR_MISMATCH, pack(&inner_key)).unwrap_or(0.0);

        dg + tp_outer - tp_inner
    }
}
