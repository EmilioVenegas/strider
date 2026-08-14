//! strider._native — optional Rust accelerator for the pure-Python
//! thermodynamics kernels in `strider.thermo.nn_dna` and `strider.thermo.salt`.
//!
//! Every exported function is bit-compatible with its Python counterpart
//! (verified by tests/test_native_parity.py across a 10k-sequence fuzz sweep);
//! the Python modules prefer these implementations when importable and fall
//! back to pure Python otherwise.
#![allow(non_snake_case)] // kwarg names mirror the Python API (sodium_M, ...)

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

const R: f64 = 1.987e-3; // kcal / (mol . K)

// ────────────────────────────────────────────────────────────────────────────
// DNA nearest-neighbor parameters (SantaLucia & Hicks 2004)
// Indexed by 2-bit dinucleotide code (b1<<2)|b2, A=0,C=1,G=2,T=3.
// Values identical to strider.thermo.nn_dna.DNA_NN (complement pairs share
// values by symmetry, exactly as the Python table spells them out).
// ────────────────────────────────────────────────────────────────────────────
const NN: [(f64, f64); 16] = [
    (-7.9, -22.2),   // AA = TT
    (-8.4, -22.4),   // AC = GT
    (-7.8, -21.0),   // AG = CT
    (-7.2, -20.4),   // AT
    (-8.5, -22.7),   // CA = TG
    (-8.0, -19.9),   // CC = GG
    (-10.6, -27.2),  // CG
    (-7.8, -21.0),   // CT = AG
    (-8.2, -22.2),   // GA = TC
    (-9.8, -24.4),   // GC
    (-8.0, -19.9),   // GG = CC
    (-8.4, -22.4),   // GT = AC
    (-7.2, -21.3),   // TA
    (-8.2, -22.2),   // TC = GA
    (-8.5, -22.7),   // TG = CA
    (-7.9, -22.2),   // TT = AA
];

const INIT_GC: (f64, f64) = (0.1, -2.8);   // terminal G-C or C-G pair
const INIT_AT: (f64, f64) = (2.3, 4.1);    // terminal A-T or T-A pair
const SYMMETRY_DS: f64 = -1.4;             // self-complementarity, entropy only

// 256-entry base → 2-bit map; 255 = not ACGT (U treated as T, DNA engine).
const fn build_base_code() -> [u8; 256] {
    let mut t = [255u8; 256];
    t[b'A' as usize] = 0;
    t[b'a' as usize] = 0;
    t[b'C' as usize] = 1;
    t[b'c' as usize] = 1;
    t[b'G' as usize] = 2;
    t[b'g' as usize] = 2;
    t[b'T' as usize] = 3;
    t[b't' as usize] = 3;
    t[b'U' as usize] = 3; // strider: .replace("U", "T")
    t[b'u' as usize] = 3;
    t
}
const BASE_CODE: [u8; 256] = build_base_code();

// ────────────────────────────────────────────────────────────────────────────
// Sequence helpers (Python-faithful: unknown chars survive translation, so
// self-complementarity checks are done on plain bytes, mirroring
// nn_dna.reverse_complement / is_self_complementary).
// ────────────────────────────────────────────────────────────────────────────

/// Python: str.upper() then .replace("U", "T")  (duplex/Tm path)
#[inline]
fn norm_upper(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'U' => b'T',
        other => other,
    }
}

/// DNA reverse-complement translation (nn_dna.COMPLEMENT): ACGT only —
/// U and any other character pass through unchanged, exactly like Python's
/// str.translate on a table limited to "ACGT"→"TGCA".
#[inline]
fn complement_byte(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'A' => b'T',
        b'T' => b'A',
        b'C' => b'G',
        b'G' => b'C',
        other => other,
    }
}

#[pyfunction]
fn reverse_complement(seq: &str) -> String {
    let bytes: Vec<u8> = seq.bytes().rev().map(complement_byte).collect();
    // Input is guaranteed ASCII/UTF-8 DNA alphabet on every strider call path.
    String::from_utf8(bytes).unwrap_or_else(|_| seq.to_uppercase())
}

fn is_self_complementary_bytes(seq: &[u8]) -> bool {
    // Public Python API: seq.upper() == reverse_complement(seq),  U passes through.
    seq.iter()
        .zip(seq.iter().rev())
        .all(|(a, b)| a.to_ascii_uppercase() == complement_byte(*b))
}

/// Duplex/Tm-path variant: Python normalizes seq to U→T BEFORE calling
/// is_self_complementary inside duplex_dh_ds / melting_temperature / duplex_dg,
/// so 'U' behaves as 'T' here.
fn is_self_complementary_norm_bytes(seq: &[u8]) -> bool {
    seq.iter().zip(seq.iter().rev()).all(|(a, b)| {
        let ca = match norm_upper(*b) {
            b'A' => b'T',
            b'T' => b'A',
            b'C' => b'G',
            b'G' => b'C',
            other => other,
        };
        norm_upper(*a) == ca
    })
}

#[pyfunction]
fn is_self_complementary(seq: &str) -> bool {
    is_self_complementary_bytes(seq.as_bytes())
}

// ────────────────────────────────────────────────────────────────────────────
// Core NN walk
// ────────────────────────────────────────────────────────────────────────────

#[inline]
fn nn_lookup(a: u8, b: u8) -> (f64, f64) {
    // Python _sum_nn: direct dict hit, else complement-pair, else (-8.0,-22.0).
    if a != 255 && b != 255 {
        return NN[((a << 2) | b) as usize];
    }
    // Complement fallback: reverse_complement(dinuc)
    let ca = complement_byte(b);
    let cb = complement_byte(a);
    let ca_code = BASE_CODE[ca as usize];
    let cb_code = BASE_CODE[cb as usize];
    if ca_code != 255 && cb_code != 255 {
        return NN[((ca_code << 2) | cb_code) as usize];
    }
    (-8.0, -22.0) // average
}

/// (ΔH kcal/mol, ΔS cal/mol/K) from `_sum_nn` + `_initiation` + symmetry term,
/// mirroring `nn_dna.duplex_dh_ds`. Input: raw UTF-8 sequence bytes.
fn duplex_dh_ds_bytes(seq: &[u8]) -> (f64, f64) {
    let mut dh = 0.0;
    let mut ds = 0.0;
    let n = seq.len();
    if n == 0 {
        return (0.0, 0.0); // Python raises IndexError; empty input is invalid
    }
    for i in 0..n.saturating_sub(1) {
        let a = BASE_CODE[seq[i] as usize];
        let b = BASE_CODE[seq[i + 1] as usize];
        let (h, s) = nn_lookup(a, b);
        dh += h;
        ds += s;
    }
    // Initiation per terminal base (Python: for end_base in (seq[0], seq[-1])).
    // Note: for a 1-base sequence Python adds BOTH endpoints = the same base twice.
    for idx in [0usize, n - 1] {
        let (h, s) = match norm_upper(seq[idx]) {
            b'G' | b'C' => INIT_GC,
            _ => INIT_AT,
        };
        dh += h;
        ds += s;
    }
    if is_self_complementary_norm_bytes(seq) {
        ds += SYMMETRY_DS;
    }
    (dh, ds)
}

#[pyfunction]
fn duplex_dh_ds(seq: &str) -> (f64, f64) {
    duplex_dh_ds_bytes(seq.as_bytes())
}

// ────────────────────────────────────────────────────────────────────────────
// Salt corrections (strider.thermo.salt) — scalar ports, bit-identical formulas
// ────────────────────────────────────────────────────────────────────────────

const T_REF_K: f64 = 310.15; // 37 °C reference

#[inline]
fn fgc(seq: &[u8]) -> f64 {
    // Python: seq.upper() letters in "GC"; length is full seq length.
    if seq.is_empty() {
        return 0.5;
    }
    let gc = seq
        .iter()
        .filter(|b| matches!(b.to_ascii_uppercase(), b'G' | b'C'))
        .count();
    gc as f64 / seq.len() as f64
}

#[inline]
fn na_tm_correction(fgc: f64, sodium_M: f64) -> f64 {
    // Owczarzy 2004 linearized, Tm_ref = 340 K (strider _na_correction)
    let ln_na = sodium_M.ln();
    let inv_tm_correction = (4.29 * fgc - 3.95) * 1e-5 * ln_na + 9.40e-6 * ln_na * ln_na;
    -inv_tm_correction * 340.0 * 340.0
}

#[inline]
fn mg_tm_correction(fgc: f64, mg_m: f64, n_bp: usize) -> f64 {
    // Owczarzy 2008 Eq. 16 (strider _mg_correction)
    let ln_mg = mg_m.ln();
    let (a, b, c, d, e, f, g) = (
        3.92e-5, -9.11e-6, 6.26e-5, 1.42e-5, -4.82e-4, 5.25e-4, 8.31e-5,
    );
    let length_factor = 1.0 / (2.0 * (n_bp.max(2) - 1) as f64);
    let inv_tm_corr =
        a + b * ln_mg + fgc * (c + d * ln_mg) + length_factor * (e + f * ln_mg + g * ln_mg * ln_mg);
    -inv_tm_corr * 340.0 * 340.0
}

#[pyfunction]
#[pyo3(signature = (seq, sodium_M, magnesium_M=0.0))]
fn owczarzy_tm_correction(seq: &str, sodium_M: f64, magnesium_M: f64) -> f64 {
    let bytes = seq.as_bytes();
    let f = fgc(bytes);
    let n_bp = bytes.len();

    if magnesium_M > 0.0 && sodium_M > 0.0 {
        let ratio = magnesium_M.sqrt() / sodium_M;
        if ratio < 0.22 {
            na_tm_correction(f, sodium_M)
        } else if ratio < 6.0 {
            // _mixed_correction (v1.2.1, issue #10): von Ahsen sodium-equivalent
            // recipe — [Na+]_eq = [Na+] + 120·√[Mg²⁺]_free (concs in mM),
            // then evaluate the monovalent Owczarzy 2004 correction at na_eq.
            let na_eq = sodium_M + 0.120 * (magnesium_M * 1000.0).sqrt();
            na_tm_correction(f, na_eq)
        } else {
            mg_tm_correction(f, magnesium_M, n_bp)
        }
    } else if magnesium_M > 0.0 {
        mg_tm_correction(f, magnesium_M, n_bp)
    } else {
        na_tm_correction(f, sodium_M)
    }
}

#[pyfunction]
#[pyo3(signature = (seq, sodium_M, celsius=37.0))]
fn na_correction_dg(seq: &str, sodium_M: f64, celsius: f64) -> f64 {
    let n = seq.len() as i64 - 1; // number of phosphates
    if n <= 0 || sodium_M <= 0.0 {
        return 0.0;
    }
    let dg_correction = 0.368 * (n as f64) * sodium_M.ln() * 1.987e-3 * (celsius + 273.15) / 1000.0;
    -dg_correction
}

#[pyfunction]
#[pyo3(signature = (sodium_M, magnesium_M=0.0, celsius=37.0, material="dna"))]
fn dg_per_bp_salt(sodium_M: f64, magnesium_M: f64, celsius: f64, material: &str) -> f64 {
    let effective_na = sodium_M + 3.4 * magnesium_M.max(0.0).sqrt();
    if effective_na <= 0.0 {
        return 0.0;
    }
    const DG_PER_BP_NA: f64 = -0.114;
    const RNA_SALT_FACTOR: f64 = 1.06;
    let coeff = if material.eq_ignore_ascii_case("rna") {
        DG_PER_BP_NA * RNA_SALT_FACTOR
    } else {
        DG_PER_BP_NA
    };
    let frac = (celsius + 273.15) / T_REF_K;
    coeff * effective_na.ln() * frac
}

#[pyfunction]
#[pyo3(signature = (seq, sodium_M, magnesium_M=0.0, celsius=37.0, material="dna"))]
fn duplex_salt_dg(seq: &str, sodium_M: f64, magnesium_M: f64, celsius: f64, material: &str) -> f64 {
    seq.len() as f64 * dg_per_bp_salt(sodium_M, magnesium_M, celsius, material)
}

#[pyfunction]
#[pyo3(signature = (n_pairs, sodium_M, magnesium_M=0.0, material="dna"))]
fn tan_chen_helix_dg(
    n_pairs: i64,
    sodium_M: f64,
    magnesium_M: f64,
    material: &str,
) -> PyResult<f64> {
    const MIN_BP: i64 = 6;
    let n = n_pairs;
    if n < MIN_BP {
        return Err(PyValueError::new_err(format!(
            "Tan-Chen helix salt model is fit for stems >= {} bp; got N={}. Use the per-base-pair model for short stems.",
            MIN_BP, n
        )));
    }
    let mat = material.to_ascii_lowercase();
    let is_rna = match mat.as_str() {
        "dna" => false,
        "rna" => true,
        _ => return Err(PyValueError::new_err("material must be 'dna' or 'rna'")),
    };

    let ln_na = if sodium_M > 0.0 { sodium_M.ln() } else { 0.0 };
    let (a1, b1) = if !is_rna {
        (-0.07 * ln_na + 0.012 * ln_na * ln_na, 0.013 * ln_na * ln_na)
    } else {
        (-0.075 * ln_na + 0.012 * ln_na * ln_na, 0.018 * ln_na * ln_na)
    };
    let dg1 = a1 + b1 / n as f64;
    if magnesium_M <= 0.0 {
        return Ok((n as f64 - 1.0) * dg1);
    }

    let ln_mg = magnesium_M.ln();
    let nf = n as f64;
    let (a2, b2) = if !is_rna {
        (
            0.02 * ln_mg + 0.0068 * ln_mg * ln_mg,
            1.18 * ln_mg + 0.344 * ln_mg * ln_mg,
        )
    } else {
        (
            -0.6 / nf + 0.025 * ln_mg + 0.0068 * ln_mg * ln_mg,
            ln_mg + 0.38 * ln_mg * ln_mg,
        )
    };
    let dg2 = a2 + b2 / (nf * nf);
    if sodium_M <= 0.0 {
        return Ok((nf - 1.0) * dg2);
    }

    let x1 =
        sodium_M / (sodium_M + (8.1 - 32.4 / nf) * (5.2 - ln_na) * magnesium_M);
    let x2 = 1.0 - x1;
    let arg = (1.0 / x1 - 1.0) * sodium_M;
    let dg12 = if arg > 0.0 {
        -0.6 * x1 * x2 * ln_na * arg.ln() / nf
    } else {
        0.0
    };
    Ok((nf - 1.0) * (x1 * dg1 + x2 * dg2) + dg12)
}

// ────────────────────────────────────────────────────────────────────────────
// Duplex thermodynamics (strider.thermo.nn_dna)
// ────────────────────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (seq, complement=None, celsius=37.0, sodium_M=1.0, magnesium_M=0.0))]
fn duplex_dg(
    seq: &str,
    complement: Option<&str>,
    celsius: f64,
    sodium_M: f64,
    magnesium_M: f64,
) -> f64 {
    let _ = complement; // API parity: upstream accepts but never reads it
    let t = celsius + 273.15;
    let (dh, ds) = duplex_dh_ds_bytes(seq.as_bytes());
    let mut dg = dh - t * (ds / 1000.0);
    if sodium_M != 1.0 || magnesium_M > 0.0 {
        dg += duplex_salt_dg(seq, sodium_M, magnesium_M, celsius, "dna");
    }
    dg
}

#[pyfunction]
#[pyo3(signature = (seq, strand_conc_M=250e-9, sodium_M=0.137, magnesium_M=0.0))]
fn melting_temperature(seq: &str, strand_conc_M: f64, sodium_M: f64, magnesium_M: f64) -> f64 {
    let (dh, ds) = duplex_dh_ds_bytes(seq.as_bytes());
    let self_comp = is_self_complementary_norm_bytes(seq.as_bytes());
    let ln_ct = if self_comp {
        strand_conc_M.ln()
    } else {
        (strand_conc_M / 4.0).ln()
    };
    let mut tm = (dh * 1000.0) / (ds + R * 1000.0 * ln_ct) - 273.15;
    if sodium_M != 1.0 || magnesium_M > 0.0 {
        tm += owczarzy_tm_correction(seq, sodium_M, magnesium_M);
    }
    tm
}

#[pyfunction]
#[pyo3(signature = (seq, sodium_M=0.05, magnesium_M=0.003, dntp_M=0.0008, oligo_conc_M=0.25e-6))]
fn duplex_tm(seq: &str, sodium_M: f64, magnesium_M: f64, dntp_M: f64, oligo_conc_M: f64) -> f64 {
    let free_mg = (magnesium_M - dntp_M).max(0.0);
    melting_temperature(seq, oligo_conc_M, sodium_M, free_mg)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reverse_complement, m)?)?;
    m.add_function(wrap_pyfunction!(is_self_complementary, m)?)?;
    m.add_function(wrap_pyfunction!(duplex_dh_ds, m)?)?;
    m.add_function(wrap_pyfunction!(duplex_dg, m)?)?;
    m.add_function(wrap_pyfunction!(melting_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(duplex_tm, m)?)?;
    m.add_function(wrap_pyfunction!(owczarzy_tm_correction, m)?)?;
    m.add_function(wrap_pyfunction!(na_correction_dg, m)?)?;
    m.add_function(wrap_pyfunction!(dg_per_bp_salt, m)?)?;
    m.add_function(wrap_pyfunction!(duplex_salt_dg, m)?)?;
    m.add_function(wrap_pyfunction!(tan_chen_helix_dg, m)?)?;
    Ok(())
}
