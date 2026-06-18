"""
Command-line interface for strider.

Subcommands:
    fold              MFE secondary structure of a single sequence
    pfunc             Ensemble partition function ΔG and base-pair probabilities
    duplex            Duplex ΔG and melting temperature for two strands
    melt              Melting temperature of a self-complementary or specified duplex
    cotranscriptional Co-transcriptional folding trajectory (prefix-by-prefix)
    verify            Run a design verification report against a JSON sequence spec
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def _make_engine(args):
    from strider.thermo.engine import ThermoEngine
    return ThermoEngine(
        material=args.material,
        celsius=args.celsius,
        sodium=args.sodium,
        magnesium=args.magnesium,
        backend=args.backend,
    )


def _read_sequence(arg: str) -> str:
    """Treat ``arg`` as a sequence string, '-' for stdin, or '@path' for a file."""
    if arg == "-":
        return sys.stdin.read().strip().upper().replace("\n", "").replace(" ", "")
    if arg.startswith("@"):
        with open(arg[1:]) as f:
            return f.read().strip().upper().replace("\n", "").replace(" ", "")
    return arg.strip().upper()


# ─── fold ────────────────────────────────────────────────────────────────────

def cmd_fold(args) -> int:
    from strider.structure.mfe import fold_mfe
    seq = _read_sequence(args.sequence)
    structure, energy, pairs = fold_mfe(seq, celsius=args.celsius, material=args.material)
    if args.json:
        print(json.dumps({
            "sequence": seq, "structure": structure,
            "energy_kcal_per_mol": float(energy),
            "pairs": [list(p) for p in pairs],
        }, indent=2))
    else:
        print(seq)
        print(structure)
        print(f"ΔG = {energy:+.3f} kcal/mol  ({len(pairs)} bp)")
    return 0


# ─── pfunc ───────────────────────────────────────────────────────────────────

def cmd_pfunc(args) -> int:
    eng = _make_engine(args)
    sequences = [_read_sequence(s) for s in args.sequences]
    res = eng.pfunc(*sequences)
    if args.json:
        out = {
            "sequences": sequences,
            "free_energy_kcal_per_mol": float(res.free_energy),
            "partition_function": float(res.partition_function),
        }
        if args.pair_probs:
            out["pair_probs"] = res.pair_probs.tolist()
        print(json.dumps(out, indent=2))
    else:
        for s in sequences:
            print(s)
        print(f"ΔG_ens = {res.free_energy:+.3f} kcal/mol  (Z = {res.partition_function:.4g}, "
              f"backend={eng.backend_name})")
    return 0


# ─── duplex / melt ───────────────────────────────────────────────────────────

def cmd_duplex(args) -> int:
    from strider.thermo.nn_dna import duplex_dg, melting_temperature, reverse_complement
    seq1 = _read_sequence(args.seq1)
    seq2 = _read_sequence(args.seq2) if args.seq2 else reverse_complement(seq1)
    dg = duplex_dg(seq1, seq2, celsius=args.celsius, sodium_M=args.sodium)
    tm = melting_temperature(
        seq1, strand_conc_M=args.strand_conc, sodium_M=args.sodium, magnesium_M=args.magnesium,
    )
    if args.json:
        print(json.dumps({
            "seq1": seq1, "seq2": seq2,
            "duplex_dg_kcal_per_mol": float(dg),
            "melting_temperature_celsius": float(tm),
        }, indent=2))
    else:
        print(f"5'-{seq1}-3'")
        print(f"3'-{seq2[::-1]}-5'")
        print(f"ΔG_duplex = {dg:+.3f} kcal/mol   Tm = {tm:.2f} °C")
    return 0


def cmd_melt(args) -> int:
    from strider.thermo.nn_dna import melting_temperature
    seq = _read_sequence(args.sequence)
    tm = melting_temperature(
        seq, strand_conc_M=args.strand_conc, sodium_M=args.sodium, magnesium_M=args.magnesium,
    )
    if args.json:
        print(json.dumps({"sequence": seq, "tm_celsius": float(tm)}, indent=2))
    else:
        print(f"Tm = {tm:.2f} °C  ([Na+]={args.sodium} M, [Mg2+]={args.magnesium} M, "
              f"[strand]={args.strand_conc} M)")
    return 0


# ─── cotranscriptional ───────────────────────────────────────────────────────

def cmd_cotranscriptional(args) -> int:
    from strider.structure.cotranscriptional import fold_cotranscriptional
    seq = _read_sequence(args.sequence)
    traj = fold_cotranscriptional(
        seq, celsius=args.celsius, material=args.material,
        min_length=args.min_length, step=args.step,
    )
    if args.json:
        print(json.dumps({
            "sequence": seq,
            "prefixes": [
                {"length": p.length, "structure": p.structure,
                 "energy_kcal_per_mol": p.energy, "pairs": [list(pp) for pp in p.pairs]}
                for p in traj.prefixes
            ],
            "rearrangements": traj.rearrangements(),
        }, indent=2))
    else:
        for p in traj.prefixes:
            print(f"{p.length:4d}  {p.structure}  ΔG={p.energy:+.3f}")
        rearr = traj.rearrangements()
        if rearr:
            print(f"\nrearrangements at prefix lengths: {rearr}")
    return 0


# ─── verify ──────────────────────────────────────────────────────────────────

def cmd_verify(args) -> int:
    """Run a CHA-style design verification from a JSON sequence spec.

    JSON spec format:
        {"mirna": "...", "H1": "...", "H2": "...", "CP": "..."}
    """
    from strider.bridge.cha import CHABridge
    with open(args.spec) as f:
        sequences = json.load(f)
    eng = _make_engine(args)
    bridge = CHABridge(sequences=sequences, engine=eng)
    report = bridge.verify()
    if args.json:
        print(json.dumps(report.to_dict() if hasattr(report, "to_dict") else str(report),
                         indent=2, default=str))
    else:
        print(report)
    return 0


# ─── draw ────────────────────────────────────────────────────────────────────

def _draw_engine(args):
    import matplotlib
    matplotlib.use("Agg")
    return _make_engine(args)


def cmd_draw_structure(args) -> int:
    eng = _draw_engine(args)
    from strider.viz.structure2d import draw_structure
    seq = _read_sequence(args.sequence) if "&" not in args.sequence else args.sequence.upper()
    ax = draw_structure(
        seq, args.structure, engine=eng, color=args.color,
        relax_iters=0 if args.no_relax else 60, title=args.title,
    )
    return _savefig(ax, args)


def cmd_draw_complex(args) -> int:
    eng = _draw_engine(args)
    from strider.viz.structure2d import draw_complex
    seqs = [_read_sequence(s) for s in args.sequences]
    ax = draw_complex(seqs, args.structure, engine=eng, names=args.names, title=args.title)
    return _savefig(ax, args)


def cmd_draw_accessibility(args) -> int:
    eng = _draw_engine(args)
    from strider.viz.annotate import draw_accessibility_track
    seq = _read_sequence(args.sequence)
    domains = None
    if args.domains:
        if args.domains.startswith("@"):
            with open(args.domains[1:]) as f:
                spec = json.load(f)
        else:
            spec = json.loads(args.domains)
        # 2-element lists are interpreted as [start, end] spans
        domains = {k: (tuple(v) if isinstance(v, list) and len(v) == 2 else v)
                   for k, v in spec.items()}
    ax = draw_accessibility_track(seq, engine=eng, domains=domains, title=args.title)
    return _savefig(ax, args)


def cmd_draw_arc(args) -> int:
    eng = _draw_engine(args)
    from strider.viz.arc import arc_diagram
    seq = _read_sequence(args.sequence) if "&" not in args.sequence else args.sequence.upper()
    ax = arc_diagram(seq, args.structure, engine=eng, color_by=args.color_by,
                     min_prob=args.min_prob, title=args.title)
    return _savefig(ax, args)


def cmd_draw_reaction(args) -> int:
    eng = _draw_engine(args)
    from strider.bridge.mantis_bridge import CHABridge
    from strider.viz.reaction import draw_cascade
    with open(args.spec) as f:
        sequences = json.load(f)
    bridge = CHABridge(sequences=sequences, engine=eng)
    fig = draw_cascade(bridge, engine=eng, show_rates=args.rates,
                       title=args.title or "Reaction cascade")
    return _savefig(fig, args)


# ─── argparse wiring ─────────────────────────────────────────────────────────

def _add_thermo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--celsius", type=float, default=37.0, help="temperature (°C, default 37)")
    p.add_argument("--material", choices=["dna", "rna"], default="dna")
    p.add_argument("--sodium", type=float, default=0.137, help="[Na+] in M (default 0.137)")
    p.add_argument("--magnesium", type=float, default=0.01, help="[Mg2+] in M (default 0.01)")
    p.add_argument("--backend", choices=["auto", "native", "vienna"],
                   default="native", help="thermo backend (default native)")


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    _add_thermo_args(p)
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def _add_fig_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", "-o", required=True,
                   help="output figure path (format from extension: png/svg/pdf)")
    p.add_argument("--dpi", type=int, default=150, help="raster DPI (default 150)")
    p.add_argument("--title", default=None, help="figure title")


def _savefig(obj, args) -> int:
    """Save a returned Axes or Figure to ``args.out`` and report."""
    fig = obj.figure if hasattr(obj, "figure") else obj
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strider",
        description="Nucleic acid thermodynamics, kinetics, and circuit design.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # fold
    p_fold = sub.add_parser("fold", help="MFE secondary structure")
    p_fold.add_argument("sequence", help="sequence (or '-' for stdin, '@file' for path)")
    p_fold.add_argument("--celsius", type=float, default=37.0)
    p_fold.add_argument("--material", choices=["dna", "rna"], default="dna")
    p_fold.add_argument("--json", action="store_true")
    p_fold.set_defaults(func=cmd_fold)

    # pfunc
    p_pf = sub.add_parser("pfunc", help="ensemble partition function ΔG")
    p_pf.add_argument("sequences", nargs="+", help="one or more sequences (multi-strand complex)")
    p_pf.add_argument("--pair-probs", action="store_true",
                      help="include pair-probability matrix in JSON output")
    _add_engine_args(p_pf)
    p_pf.set_defaults(func=cmd_pfunc)

    # duplex
    p_dx = sub.add_parser("duplex", help="duplex ΔG and Tm")
    p_dx.add_argument("seq1", help="first strand (5'→3')")
    p_dx.add_argument("seq2", nargs="?", help="second strand (default: reverse complement of seq1)")
    p_dx.add_argument("--celsius", type=float, default=37.0)
    p_dx.add_argument("--sodium", type=float, default=0.137)
    p_dx.add_argument("--magnesium", type=float, default=0.01)
    p_dx.add_argument("--strand-conc", type=float, default=1e-6,
                      help="strand concentration for Tm (default 1 µM)")
    p_dx.add_argument("--json", action="store_true")
    p_dx.set_defaults(func=cmd_duplex)

    # melt
    p_mt = sub.add_parser("melt", help="melting temperature of a duplex")
    p_mt.add_argument("sequence", help="sequence (Tm against its reverse complement)")
    p_mt.add_argument("--sodium", type=float, default=0.137)
    p_mt.add_argument("--magnesium", type=float, default=0.01)
    p_mt.add_argument("--strand-conc", type=float, default=1e-6)
    p_mt.add_argument("--json", action="store_true")
    p_mt.set_defaults(func=cmd_melt)

    # cotranscriptional
    p_ct = sub.add_parser("cotranscriptional", aliases=["cotx"],
                          help="co-transcriptional folding trajectory")
    p_ct.add_argument("sequence", help="full sequence")
    p_ct.add_argument("--celsius", type=float, default=37.0)
    p_ct.add_argument("--material", choices=["dna", "rna"], default="rna")
    p_ct.add_argument("--min-length", type=int, default=5,
                      help="skip prefixes shorter than this (default 5)")
    p_ct.add_argument("--step", type=int, default=1, help="sample every N nucleotides")
    p_ct.add_argument("--json", action="store_true")
    p_ct.set_defaults(func=cmd_cotranscriptional)

    # verify
    p_v = sub.add_parser("verify", help="verify a sequence design from a JSON spec")
    p_v.add_argument("spec", help="path to JSON sequence spec")
    _add_engine_args(p_v)
    p_v.set_defaults(func=cmd_verify)

    # draw (nested subcommands)
    _add_draw_subparser(sub)

    return parser


def _add_draw_subparser(sub) -> None:
    p_draw = sub.add_parser("draw", help="render visualization figures")
    draw_sub = p_draw.add_subparsers(dest="draw_command", required=True)

    # draw structure
    ps = draw_sub.add_parser("structure", help="2D secondary structure")
    ps.add_argument("sequence", help="sequence ('&'/'+' for multi-strand, '@file', '-' stdin)")
    ps.add_argument("--structure", default=None, help="dot-bracket (folded if omitted)")
    ps.add_argument("--color", choices=["nt", "strand", "accessibility"], default="nt")
    ps.add_argument("--no-relax", action="store_true", help="disable layout relaxation")
    _add_thermo_args(ps)
    _add_fig_args(ps)
    ps.set_defaults(func=cmd_draw_structure)

    # draw complex
    pc = draw_sub.add_parser("complex", help="2D multi-strand complex")
    pc.add_argument("sequences", nargs="+", help="two or more strand sequences")
    pc.add_argument("--names", nargs="+", default=None, help="per-strand labels")
    pc.add_argument("--structure", default=None, help="dot-bracket (folded if omitted)")
    _add_thermo_args(pc)
    _add_fig_args(pc)
    pc.set_defaults(func=cmd_draw_complex)

    # draw accessibility
    pa = draw_sub.add_parser("accessibility", help="1D toehold-accessibility track")
    pa.add_argument("sequence", help="sequence ('@file', '-' stdin)")
    pa.add_argument("--domains", default=None,
                    help='JSON or @file of {"name":[start,end]} domain spans')
    _add_thermo_args(pa)
    _add_fig_args(pa)
    pa.set_defaults(func=cmd_draw_accessibility)

    # draw arc
    par = draw_sub.add_parser("arc", help="arc diagram")
    par.add_argument("sequence", help="sequence ('&'/'+' for multi-strand)")
    par.add_argument("--structure", default=None, help="dot-bracket (folded if omitted)")
    par.add_argument("--color-by", choices=["type", "probability", "strand"], default="type")
    par.add_argument("--min-prob", type=float, default=0.1)
    _add_thermo_args(par)
    _add_fig_args(par)
    par.set_defaults(func=cmd_draw_arc)

    # draw reaction
    pr = draw_sub.add_parser("reaction", help="CHA reaction cascade from a JSON spec")
    pr.add_argument("--spec", required=True, help="JSON spec {mirna,H1,H2,CP}")
    pr.add_argument("--rates", action="store_true", help="annotate kinetic rates")
    _add_thermo_args(pr)
    _add_fig_args(pr)
    pr.set_defaults(func=cmd_draw_reaction)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
