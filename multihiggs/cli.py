from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import load_config
from .fit import fit_results_csv, format_polynomial_report
from .grid import generate_scan_points
from .histograms import write_histogram_csv
from .madgraph import run_madevent, run_mg5, write_madevent_card, write_process_card
from .results import discover_completed_runs, write_results_csv
from .term_inference import format_inferred_terms, infer_terms_from_process_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MadGraph multi-Higgs scan and fitting pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    grid_parser = add_config_command(subparsers, "grid", "Write the configured scan grid")
    grid_parser.add_argument("-o", "--output", type=Path)

    process_parser = add_config_command(subparsers, "generate-process", "Write or run an MG5 process card")
    process_parser.add_argument("-o", "--output", type=Path)
    process_parser.add_argument("--force-output", action="store_true", help="Append -f to the MG5 output command")
    process_parser.add_argument("--run", action="store_true", help="Run mg5_aMC after writing the card")

    scan_parser = add_config_command(subparsers, "scan", "Write or run a madevent scan command file")
    scan_parser.add_argument("-o", "--output", type=Path)
    scan_parser.add_argument("--max-runs", type=int)
    scan_parser.add_argument("--force", action="store_true", help="Do not skip existing run directories")
    scan_parser.add_argument("--run", action="store_true", help="Run madevent after writing the card")

    collect_parser = add_config_command(subparsers, "collect", "Collect completed run cross sections")
    collect_parser.add_argument("-o", "--output", type=Path)
    collect_parser.add_argument("--run-number", action="append", dest="run_numbers")
    collect_parser.add_argument("--exclude-run-number", action="append", dest="exclude_run_numbers")
    collect_parser.add_argument(
        "--min-events",
        type=int,
        help="Override the configured event-count minimum; use 0 to disable the filter",
    )

    hist_parser = add_config_command(subparsers, "hist", "Histogram configured LHE observables")
    hist_parser.add_argument("-o", "--output", type=Path)
    hist_parser.add_argument("--observable", action="append", dest="observables")
    hist_parser.add_argument("--run-number", action="append", dest="run_numbers")
    hist_parser.add_argument(
        "--min-events",
        type=int,
        help="Override the configured event-count minimum; use 0 to disable the filter",
    )

    fit_parser = add_config_command(subparsers, "fit", "Fit cross sections or histogram bins")
    fit_parser.add_argument("-i", "--input", type=Path)
    fit_parser.add_argument("-o", "--output", type=Path)
    fit_parser.add_argument(
        "--min-events",
        type=int,
        help="Only fit CSV rows with event_count greater than or equal to this value",
    )
    fit_parser.add_argument(
        "--print-polynomial",
        action="store_true",
        help="Print old-style polynomial coefficient blocks after fitting",
    )

    infer_parser = add_config_command(
        subparsers,
        "infer-terms",
        "Infer fit terms from a generated MG5 process directory",
    )
    infer_parser.add_argument(
        "--process-dir",
        type=Path,
        help="Override the generated process directory; defaults to [process].mg5_path / output",
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "grid":
        return command_grid(config, args.output)
    if args.command == "generate-process":
        return command_generate_process(config, args.output, args.force_output, args.run)
    if args.command == "scan":
        return command_scan(config, args.output, args.max_runs, args.force, args.run)
    if args.command == "collect":
        return command_collect(
            config,
            args.output,
            args.run_numbers,
            args.exclude_run_numbers,
            args.min_events,
        )
    if args.command == "hist":
        return command_hist(config, args.output, args.observables, args.run_numbers, args.min_events)
    if args.command == "fit":
        return command_fit(config, args.input, args.output, args.print_polynomial, args.min_events)
    if args.command == "infer-terms":
        return command_infer_terms(config, args.process_dir)
    raise AssertionError(args.command)


def add_config_command(subparsers, name: str, help_text: str):
    command = subparsers.add_parser(name, help=help_text)
    command.add_argument("config", type=Path)
    return command


def default_path(config, filename: str) -> Path:
    return config.output_dir / filename


def command_grid(config, output: Path | None) -> int:
    points = generate_scan_points(config)
    output = output or default_path(config, "scan_points.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [coupling.name for coupling in config.couplings] + [
        coupling.name + "_text" for coupling in config.couplings
    ] + ["run_name"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            row = point.as_dict(config)
            row["run_name"] = point.run_name(config)
            writer.writerow(row)
    print(f"Wrote {len(points)} scan points to {output}")
    return 0


def command_generate_process(config, output: Path | None, force_output: bool, run: bool) -> int:
    output = output or default_path(config, config.name + "_process.mg5")
    write_process_card(config, output, force_output=force_output)
    print(f"Wrote MG5 process card to {output}")
    if not run:
        print(f"Dry run only. Start MG5 with: multihiggs generate-process {config.path} --run")
        return 0
    return run_mg5(config, output)


def command_scan(
    config,
    output: Path | None,
    max_runs: int | None,
    force: bool,
    run: bool,
) -> int:
    points = generate_scan_points(config)
    output = output or default_path(config, config.name + "_scan.dcmd")
    _, selected = write_madevent_card(config, points, output, max_runs=max_runs, force=force)
    print(f"Configured scan points: {len(points)}")
    print(f"Events per selected run: {config.scan.nevents}")
    print(f"Existing-run event minimum: {config.scan.event_minimum}")
    print(f"Selected points to run: {len(selected)}")
    print(f"Wrote madevent command file to {output}")
    if selected:
        print("First selected points:")
        for point in selected[:10]:
            print("  " + ", ".join(f"{c.name}={t}" for c, t in zip(config.couplings, point.texts)))
    if not run:
        print(f"Dry run only. Start madevent with: multihiggs scan {config.path} --run")
        return 0
    if not selected:
        return 0
    return run_madevent(config, output)


def command_collect(
    config,
    output: Path | None,
    run_numbers: list[str] | None,
    exclude_run_numbers: list[str] | None,
    min_events: int | None,
) -> int:
    results = discover_completed_runs(
        config,
        run_numbers=None if run_numbers is None else set(run_numbers),
        exclude_run_numbers=None if exclude_run_numbers is None else set(exclude_run_numbers),
        min_events=min_events,
    )
    output = output or default_path(config, "xsecs.csv")
    write_results_csv(config, results, output)
    print(f"Wrote {len(results)} completed runs to {output}")
    return 0


def command_hist(
    config,
    output: Path | None,
    observables: list[str] | None,
    run_numbers: list[str] | None,
    min_events: int | None,
) -> int:
    output = output or default_path(config, "histograms.csv")
    write_histogram_csv(
        config,
        output,
        observable_names=None if observables is None else set(observables),
        run_numbers=None if run_numbers is None else set(run_numbers),
        min_events=min_events,
    )
    print(f"Wrote histograms to {output}")
    return 0


def command_fit(
    config,
    input_path: Path | None,
    output: Path | None,
    print_polynomial: bool,
    min_events: int | None,
) -> int:
    input_path = input_path or default_path(config, "xsecs.csv")
    output = output or default_path(config, "fit.json")
    results = fit_results_csv(config, input_path, output, min_events=min_events)
    print(f"Wrote {len(results)} fit result(s) to {output}")
    for result in results[:10]:
        print(
            f"{result.label}: sigma_SM={result.sigma_sm:.6g} pb, "
            f"rank={result.rank}/{len(config.fit.terms)}, chi2/dof={result.chi2_dof:.4g}"
        )
    if len(results) > 10:
        print(f"... {len(results) - 10} additional bins")
    if print_polynomial:
        for result in results:
            print()
            print(format_polynomial_report(config, result))
    return 0


def command_infer_terms(config, process_dir: Path | None) -> int:
    process_dir = process_dir or config.process_dir
    result = infer_terms_from_process_dir(
        process_dir,
        tuple(coupling.parameter for coupling in config.couplings),
    )
    print(format_inferred_terms(result))
    return 0
