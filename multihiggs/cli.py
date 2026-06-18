from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .contours import (
    build_contour_data,
    build_variation_data,
    load_fit_json,
    parse_fixed_values,
    safe_filename_piece,
    select_fit_record,
    write_contour_plot,
    write_variation_plot,
)
from .distributions import (
    build_distribution_data,
    default_distribution_output,
    parse_parameter_points,
    write_distribution_plot,
)
from .fit import fit_results_csv, format_polynomial_report
from .grid import generate_scan_points
from .histograms import write_histogram_csv
from .madgraph import run_madevent_points, run_mg5, write_madevent_card, write_process_card
from .mheft import restricted_mheft_squared_order_cap
from .results import discover_completed_runs, write_results_csv
from .term_inference import format_inferred_terms, infer_terms_from_process_dir, update_config_fit_terms
from .term_maps import resolve_term_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MadGraph multi-Higgs scan and fitting pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    grid_parser = add_config_command(subparsers, "grid", "Write the configured scan grid")
    grid_parser.add_argument("-o", "--output", type=Path)
    grid_parser.add_argument(
        "--run-number",
        help="Override [scan].run_number for generated run names without editing the config",
    )

    process_parser = add_config_command(subparsers, "generate-process", "Write or run an MG5 process card")
    process_parser.add_argument("-o", "--output", type=Path)
    process_parser.add_argument("--force-output", action="store_true", help="Append -f to the MG5 output command")
    process_parser.add_argument("--run", action="store_true", help="Run mg5_aMC after writing the card")

    scan_parser = add_config_command(subparsers, "scan", "Write or run a madevent scan command file")
    scan_parser.add_argument("-o", "--output", type=Path)
    scan_parser.add_argument("--max-runs", type=int)
    scan_parser.add_argument("--force", action="store_true", help="Do not skip existing run directories")
    scan_parser.add_argument("--run", action="store_true", help="Run madevent after writing the card")
    scan_parser.add_argument(
        "--run-number",
        help="Override [scan].run_number for generated run names without editing the config",
    )

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
    fit_parser.add_argument(
        "--term-map",
        help="Term map for --physical-basis or additional polynomial output with --print-polynomial",
    )
    fit_parser.add_argument(
        "--expand-term-map",
        action="store_true",
        help="Also print expanded polynomial coefficients for the selected --term-map",
    )
    fit_parser.add_argument(
        "--physical-basis",
        action="store_true",
        help="Infer a physical_monomial fit basis from the generated process before fitting",
    )
    fit_parser.add_argument(
        "--amplitude-basis",
        choices=("sm_like_hhh",),
        help="Project inferred physical amplitudes onto a named compact amplitude basis before fitting",
    )

    contour_parser = add_config_command(subparsers, "contour", "Plot a two-variable normalized fit contour")
    contour_parser.add_argument("-i", "--input", type=Path, help="Fit JSON to plot; defaults to fit.json")
    contour_parser.add_argument("-o", "--output", type=Path, help="Output image path")
    contour_parser.add_argument("--pdf-output", type=Path, help="Optional PDF output path")
    contour_parser.add_argument("--label", help="Fit label to plot when the JSON contains multiple fits")
    contour_parser.add_argument("--x", required=True, help="Configured coupling to use on the x axis")
    contour_parser.add_argument("--y", required=True, help="Configured coupling to use on the y axis")
    contour_parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Fix a non-axis coupling to a scan-variable value; may be repeated",
    )
    contour_parser.add_argument(
        "--x-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Override the x-axis scan-variable range",
    )
    contour_parser.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Override the y-axis scan-variable range",
    )
    contour_parser.add_argument("--points", type=int, default=201, help="Default grid points per axis")
    contour_parser.add_argument("--x-points", type=int, help="Grid points on the x axis")
    contour_parser.add_argument("--y-points", type=int, help="Grid points on the y axis")
    contour_parser.add_argument("--linear", action="store_true", help="Use a linear color scale")

    variation_parser = add_config_command(
        subparsers,
        "variation",
        "Plot a one-variable normalized cross-section variation",
    )
    variation_parser.add_argument("-i", "--input", type=Path, help="Fit JSON to plot; defaults to fit.json")
    variation_parser.add_argument("-o", "--output", type=Path, help="Output image path")
    variation_parser.add_argument("--pdf-output", type=Path, help="Optional PDF output path")
    variation_parser.add_argument("--label", help="Fit label to plot when the JSON contains multiple fits")
    variation_parser.add_argument("--x", required=True, help="Configured coupling to scan on the x axis")
    variation_parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Fix a non-axis coupling to a scan-variable value; may be repeated",
    )
    variation_parser.add_argument(
        "--x-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Override the x-axis scan-variable range",
    )
    variation_parser.add_argument("--points", type=int, default=201, help="Number of points on the x axis")
    variation_parser.add_argument("--log-y", action="store_true", help="Use a logarithmic y axis")

    distribution_parser = add_config_command(
        subparsers,
        "distribution",
        "Plot fitted histogram distributions at selected parameter points",
    )
    distribution_parser.add_argument("-i", "--input", type=Path, help="Histogram fit JSON; defaults to hist_fit.json")
    distribution_parser.add_argument("-o", "--output", type=Path, help="Output image path")
    distribution_parser.add_argument("--pdf-output", type=Path, help="Optional PDF output path")
    distribution_parser.add_argument("--observable", required=True, help="Configured observable to plot")
    distribution_parser.add_argument(
        "--point",
        action="append",
        default=[],
        metavar="NAME=VALUE[,NAME=VALUE...]",
        help="Parameter point to plot; may be repeated. Unspecified couplings use sm_value.",
    )
    distribution_parser.add_argument("--no-sm", action="store_true", help="Do not add the SM distribution")
    distribution_parser.add_argument("--density", action="store_true", help="Plot bin contents divided by bin width")
    distribution_parser.add_argument("--log-y", action="store_true", help="Use a logarithmic y axis")

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
    infer_parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Only print inferred terms; do not rewrite [fit].terms in the config",
    )
    infer_parser.add_argument(
        "--term-map",
        help="Also print inferred polynomial powers in the named term map",
    )
    infer_parser.add_argument(
        "--expand-term-map",
        action="store_true",
        help="Also print expanded polynomial powers for the selected --term-map",
    )
    infer_parser.add_argument(
        "--physical-basis",
        action="store_true",
        help="Infer/update terms in the selected --term-map as the physical fit basis",
    )
    infer_parser.add_argument(
        "--amplitude-basis",
        choices=("sm_like_hhh",),
        help="Project inferred physical amplitudes onto a named compact amplitude basis",
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "grid":
        config = with_run_number(config, args.run_number)
        return command_grid(config, args.output)
    if args.command == "generate-process":
        return command_generate_process(config, args.output, args.force_output, args.run)
    if args.command == "scan":
        config = with_run_number(config, args.run_number)
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
        return command_fit(
            config,
            args.input,
            args.output,
            args.print_polynomial,
            args.min_events,
            args.term_map,
            args.expand_term_map,
            args.physical_basis,
            args.amplitude_basis,
        )
    if args.command == "contour":
        return command_contour(
            config,
            args.input,
            args.output,
            args.pdf_output,
            args.label,
            args.x,
            args.y,
            args.fix,
            args.x_range,
            args.y_range,
            args.points,
            args.x_points,
            args.y_points,
            not args.linear,
        )
    if args.command == "variation":
        return command_variation(
            config,
            args.input,
            args.output,
            args.pdf_output,
            args.label,
            args.x,
            args.fix,
            args.x_range,
            args.points,
            args.log_y,
        )
    if args.command == "distribution":
        return command_distribution(
            config,
            args.input,
            args.output,
            args.pdf_output,
            args.observable,
            args.point,
            not args.no_sm,
            args.density,
            args.log_y,
        )
    if args.command == "infer-terms":
        return command_infer_terms(
            config,
            args.process_dir,
            not args.no_update_config,
            args.term_map,
            args.expand_term_map,
            args.physical_basis,
            args.amplitude_basis,
        )
    raise AssertionError(args.command)


def add_config_command(subparsers, name: str, help_text: str):
    command = subparsers.add_parser(name, help=help_text)
    command.add_argument("config", type=Path)
    return command


def with_run_number(config, run_number: str | None):
    if run_number is None:
        return config
    return replace(config, scan=replace(config.scan, run_number=str(run_number)))


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
    return run_madevent_points(config, selected, output)


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
    term_map_name: str | None,
    expand_term_map: bool,
    physical_basis: bool,
    amplitude_basis: str | None,
) -> int:
    input_path = input_path or default_path(config, "xsecs.csv")
    output = output or default_path(config, "fit.json")
    fit_config = config
    if physical_basis:
        if term_map_name is None:
            raise ValueError("--physical-basis requires --term-map")
        source_names = tuple(coupling.parameter for coupling in config.couplings)
        term_map = resolve_term_map(config, term_map_name, source_names=source_names)
        inferred_terms = infer_terms_from_process_dir(
            config.process_dir,
            source_names,
            term_map=term_map,
            physical_basis=True,
            mheft_squared_order_cap=restricted_mheft_squared_order_cap(config),
            amplitude_basis=amplitude_basis,
        )
        fit_config = replace(
            config,
            fit=replace(
                config.fit,
                basis="physical_monomial",
                terms=inferred_terms.cross_section_terms,
                term_map=term_map_name,
            ),
        )
        print(f"Using inferred physical fit basis: {len(fit_config.fit.terms)} term(s)")
        print(f"Physical-basis fit variables: {','.join(inferred_terms.coupling_names)}")
        if amplitude_basis is not None:
            print(f"Applied amplitude basis: {amplitude_basis}")
    elif amplitude_basis is not None:
        raise ValueError("--amplitude-basis requires --physical-basis")

    results = fit_results_csv(fit_config, input_path, output, min_events=min_events)
    print(f"Wrote {len(results)} fit result(s) to {output}")
    for result in results[:10]:
        print(
            f"{result.label}: sigma_SM={result.sigma_sm:.6g} pb, "
            f"rank={result.rank}/{len(fit_config.fit.terms)}, chi2/dof={result.chi2_dof:.4g}"
        )
    if len(results) > 10:
        print(f"... {len(results) - 10} additional bins")
    if print_polynomial:
        for result in results:
            print()
            print(
                format_polynomial_report(
                    fit_config,
                    result,
                    term_map_name=term_map_name,
                    expand_term_map=expand_term_map,
                )
            )
    return 0


def command_contour(
    config,
    input_path: Path | None,
    output: Path | None,
    pdf_output: Path | None,
    label: str | None,
    x_name: str,
    y_name: str,
    raw_fixed_values: list[str] | None,
    x_range: list[float] | None,
    y_range: list[float] | None,
    points: int,
    x_points: int | None,
    y_points: int | None,
    log_scale: bool,
) -> int:
    input_path = input_path or default_path(config, "fit.json")
    payload = load_fit_json(input_path)
    fit_record = select_fit_record(payload, label)
    fixed_values = parse_fixed_values(config, raw_fixed_values, axis_names={x_name, y_name})
    contour = build_contour_data(
        config,
        fit_record,
        x_name=x_name,
        y_name=y_name,
        fixed_values=fixed_values,
        x_range=None if x_range is None else (x_range[0], x_range[1]),
        y_range=None if y_range is None else (y_range[0], y_range[1]),
        x_points=x_points or points,
        y_points=y_points or points,
    )
    label_piece = safe_filename_piece(contour.fit_label)
    output = output or default_path(config, f"{label_piece}_{contour.x_name}_{contour.y_name}_contour.png")
    write_contour_plot(config, contour, output, log_scale=log_scale)
    print(f"Wrote contour plot to {output}")
    if pdf_output is not None:
        write_contour_plot(config, contour, pdf_output, log_scale=log_scale)
        print(f"Wrote contour plot to {pdf_output}")
    print(
        f"Plotted {contour.fit_label}: {contour.x_name} vs {contour.y_name}, "
        f"ratio range {float(contour.ratio.min()):.6g} to {float(contour.ratio.max()):.6g}"
    )
    if contour.fixed_values:
        fixed_text = ", ".join(f"{name}={value:g}" for name, value in sorted(contour.fixed_values.items()))
        print(f"Fixed: {fixed_text}")
    return 0


def command_variation(
    config,
    input_path: Path | None,
    output: Path | None,
    pdf_output: Path | None,
    label: str | None,
    x_name: str,
    raw_fixed_values: list[str] | None,
    x_range: list[float] | None,
    points: int,
    log_y: bool,
) -> int:
    input_path = input_path or default_path(config, "fit.json")
    payload = load_fit_json(input_path)
    fit_record = select_fit_record(payload, label)
    fixed_values = parse_fixed_values(config, raw_fixed_values, axis_names={x_name})
    variation = build_variation_data(
        config,
        fit_record,
        x_name=x_name,
        fixed_values=fixed_values,
        x_range=None if x_range is None else (x_range[0], x_range[1]),
        points=points,
    )
    label_piece = safe_filename_piece(variation.fit_label)
    output = output or default_path(config, f"{label_piece}_{variation.x_name}_variation.png")
    write_variation_plot(config, variation, output, log_y=log_y)
    print(f"Wrote variation plot to {output}")
    if pdf_output is not None:
        write_variation_plot(config, variation, pdf_output, log_y=log_y)
        print(f"Wrote variation plot to {pdf_output}")
    print(
        f"Plotted {variation.fit_label}: {variation.x_name}, "
        f"ratio range {float(variation.ratio.min()):.6g} to {float(variation.ratio.max()):.6g}"
    )
    if variation.fixed_values:
        fixed_text = ", ".join(f"{name}={value:g}" for name, value in sorted(variation.fixed_values.items()))
        print(f"Fixed: {fixed_text}")
    return 0


def command_distribution(
    config,
    input_path: Path | None,
    output: Path | None,
    pdf_output: Path | None,
    observable_name: str,
    raw_points: list[str] | None,
    include_sm: bool,
    density: bool,
    log_y: bool,
) -> int:
    input_path = input_path or default_path(config, "hist_fit.json")
    payload = load_fit_json(input_path)
    parameter_points = parse_parameter_points(config, raw_points)
    distribution = build_distribution_data(
        config,
        payload,
        observable_name,
        parameter_points,
        include_sm=include_sm,
        density=density,
    )
    output = output or default_distribution_output(config, observable_name)
    write_distribution_plot(config, distribution, output, log_y=log_y)
    print(f"Wrote distribution plot to {output}")
    if pdf_output is not None:
        write_distribution_plot(config, distribution, pdf_output, log_y=log_y)
        print(f"Wrote distribution plot to {pdf_output}")
    print(
        f"Plotted {observable_name} for "
        + ", ".join(series.label for series in distribution.series)
    )
    return 0


def command_infer_terms(
    config,
    process_dir: Path | None,
    update_config: bool,
    term_map_name: str | None,
    expand_term_map: bool,
    physical_basis: bool,
    amplitude_basis: str | None,
) -> int:
    process_dir = process_dir or config.process_dir
    if physical_basis and term_map_name is None:
        raise ValueError("--physical-basis requires --term-map")
    if amplitude_basis is not None and not physical_basis:
        raise ValueError("--amplitude-basis requires --physical-basis")
    source_names = tuple(coupling.parameter for coupling in config.couplings)
    term_map = (
        None
        if term_map_name is None
        else resolve_term_map(config, term_map_name, source_names=source_names)
    )
    result = infer_terms_from_process_dir(
        process_dir,
        source_names,
        term_map=term_map,
        physical_basis=physical_basis,
        mheft_squared_order_cap=restricted_mheft_squared_order_cap(config),
        amplitude_basis=amplitude_basis,
    )
    print(format_inferred_terms(result, term_map=term_map, expand_term_map=expand_term_map))
    if update_config:
        update_config_fit_terms(
            config.path,
            result.cross_section_terms,
            basis="physical_monomial" if physical_basis else None,
            term_map_name=term_map_name if physical_basis else None,
        )
        print()
        print(f"WARNING: Updated [fit].terms in {config.path}.")
        print("WARNING: Subsequent `fit` commands will use these inferred terms from the config file.")
        npoints = len(generate_scan_points(config))
        nterms = len(result.cross_section_terms)
        if npoints < nterms:
            print(
                "WARNING: "
                f"The configured scan has {npoints} point(s), but the inferred fit uses {nterms} term(s). "
                "Increase the scan grid before fitting, or the fit will be underdetermined."
            )
        print("WARNING: CHECK the inferred terms before using them for production fits.")
    return 0
