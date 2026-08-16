//! Packed-key lookup helpers for the generated DNA energy tables.
//!
//! Keys in `parameters_dna` are short ACGT strings (2–8 bases).  We pack each
//! base as 2 bits (A=0, C=1, G=2, T=3, U treated as T) LSB-first, exactly
//! mirroring `native/codegen_tables.py::pack`, and binary-search the sorted
//! generated arrays.  This avoids building any map at runtime: lookups are
//! ≤ log2(N) comparisons on static data (≤ 14 for the 9216-entry 2×2 table).

#[allow(dead_code)] // some tables (hairpin set) land in a later DP port
pub mod dna {
    include!("tables_dna.rs");
}

/// 256-entry ASCII base → 2-bit code table (U folded to T).
///
/// UNKNOWN bytes map to `u32::MAX`, NOT to 3 (T): a base like 'N' must produce
/// a packed key that *misses* every generated table entry (same as Python's
/// `dict.get(key, default)` returning the default), never a false T-lookup.
/// All valid table keys are ≤ 8 ACGT bases → packed codes ≤ 65535, so any
/// code containing the MAX sentinel bit is guaranteed to miss.
pub static CODE_TABLE: [u32; 256] = {
    let mut t = [u32::MAX; 256];
    t[b'A' as usize] = 0;
    t[b'a' as usize] = 0;
    t[b'C' as usize] = 1;
    t[b'c' as usize] = 1;
    t[b'G' as usize] = 2;
    t[b'g' as usize] = 2;
    t[b'T' as usize] = 3;
    t[b't' as usize] = 3;
    t[b'U' as usize] = 3;
    t[b'u' as usize] = 3;
    t
};

/// Pack a byte slice as 2-bit codes LSB-first. Identical to the generator.
#[inline(always)]
pub fn pack(bytes: &[u8]) -> u32 {
    let mut code = 0u32;
    for (i, &b) in bytes.iter().enumerate() {
        code |= CODE_TABLE[b as usize] << (2 * i);
    }
    code
}

/// Binary-search a generated, code-sorted table. `None` when absent — the
/// Python fallback dicts affect flow exactly like this (`.get(key, default)`),
/// so result ordering must be replicated at the call site.
#[inline(always)]
pub fn lookup(table: &[(u32, f64)], code: u32) -> Option<f64> {
    table
        .binary_search_by_key(&code, |&(c, _)| c)
        .ok()
        .map(|i| table[i].1)
}
