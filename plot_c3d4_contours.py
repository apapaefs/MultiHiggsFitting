import gzip
import glob
import math
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
MG_LOCATION = os.path.join(ROOT, 'MG5_aMC_v3_5_15')

C3_PLOT_RANGE = (-30.0, 30.0)
D4_PLOT_RANGE = (-700.0, 700.0)
K3_FIT_RANGE = (-29.0, 31.0)
K4_FIT_RANGE = (-699.0, 701.0)

PLOT_NC3 = 601
PLOT_ND4 = 701
REQUIRE_WIDE_RUNS = True
ENERGY_LABEL = '14 TeV'

UNITARITY_MH = 125.0
UNITARITY_V = 246.0
UNITARITY_LEVEL = 0.5
UNITARITY_NC3 = 301
UNITARITY_ND4 = 401
UNITARITY_SQRTS = np.arange(200.0, 5000.0, 10.0)
_UNITARITY_GRID = None


PROCESSES = [
    {
        'name': 'gg_hhh',
        'title': r'$gg \to hhh$',
        'directory': 'gg_hhh_c3d4',
        'runnumbers': None,
        # Run 1 is the older HHH grid and is inconsistent with the
        # regenerated precision grid at duplicated points.
        'exclude_runnumbers': ['1'],
        'wide_runnum': '3',
        'terms': [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (0, 2),
        ],
        'expected_wide_runs': 11,
        'outfile_base': 'hhh_c3d4_normalized_contour',
    },
    {
        'name': 'gg_4h',
        'title': r'$gg \to hhhh$',
        'directory': 'gg_4h_c3d4',
        'runnumbers': None,
        'wide_runnum': '3',
        'terms': (
            [(i, 0) for i in range(0, 7)]
            + [(i, 1) for i in range(0, 5)]
            + [(i, 2) for i in range(0, 3)]
        ),
        'expected_wide_runs': 17,
        'outfile_base': 'hhhh_c3d4_normalized_contour',
    },
]


def round_sig(x, sig=4):
    if x == 0.0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def scale_to_chebyshev(value, value_range):
    xmin, xmax = value_range
    return (2.0 * value - xmin - xmax) / (xmax - xmin)


def chebyshev_t(order, x):
    if order == 0:
        return np.ones_like(x, dtype=float)
    if order == 1:
        return np.asarray(x, dtype=float)

    t_prev = np.ones_like(x, dtype=float)
    t_curr = np.asarray(x, dtype=float)
    for _ in range(2, order + 1):
        t_prev, t_curr = t_curr, 2.0 * x * t_curr - t_prev
    return t_curr


def chebyshev_t_coefficients(order):
    if order == 0:
        return np.asarray([1.0], dtype=float)
    if order == 1:
        return np.asarray([0.0, 1.0], dtype=float)

    t_prev = np.asarray([1.0], dtype=float)
    t_curr = np.asarray([0.0, 1.0], dtype=float)
    for _ in range(2, order + 1):
        doubled_x_t = np.concatenate(([0.0], 2.0 * t_curr))
        if len(t_prev) < len(doubled_x_t):
            t_prev = np.pad(t_prev, (0, len(doubled_x_t) - len(t_prev)))
        t_prev, t_curr = t_curr, doubled_x_t - t_prev
    return t_curr


def chebyshev_t_coupling_coefficients(order, value_range):
    xmin, xmax = value_range
    scale = 2.0 / (xmax - xmin)
    offset = (2.0 - xmin - xmax) / (xmax - xmin)

    x_coeffs = chebyshev_t_coefficients(order)
    coupling_coeffs = np.zeros(order + 1, dtype=float)
    for power, coeff in enumerate(x_coeffs):
        for out_power in range(power + 1):
            coupling_coeffs[out_power] += (
                coeff
                * math.comb(power, out_power)
                * (offset ** (power - out_power))
                * (scale ** out_power)
            )
    return coupling_coeffs


def chebyshev_row(c3, d4, terms):
    k3 = 1.0 + float(c3)
    k4 = 1.0 + float(d4)
    x3 = scale_to_chebyshev(k3, K3_FIT_RANGE)
    x4 = scale_to_chebyshev(k4, K4_FIT_RANGE)
    return np.asarray([chebyshev_t(i, x3) * chebyshev_t(j, x4) for i, j in terms], dtype=float)


def read_xsec(lhefile):
    with gzip.open(lhefile, 'rt', errors='ignore') as stream:
        text = stream.read()
    match = re.search(r'Integrated weight \(pb\)\s*:\s*([0-9.eE+-]+)', text)
    if match is None:
        raise RuntimeError('No Integrated weight found in ' + lhefile)
    return float(match.group(1))


def read_integration_error(proc_dir, run_name, xsec):
    html_file = os.path.join(proc_dir, 'HTML', run_name, 'results.html')
    if os.path.exists(html_file):
        with open(html_file) as stream:
            html = stream.read()
        match = re.search(
            r'<b>s=\s*([0-9.eE+-]+)\s*(?:&#177|&plusmn;|±)\s*([0-9.eE+-]+)\s*\(pb\)</b>',
            html,
        )
        if match is not None:
            return float(match.group(2))
    print('Warning: using 1% fallback integration error for', run_name)
    return max(abs(float(xsec)) * 0.01, 1e-30)


def read_runs(config):
    proc_dir = os.path.join(MG_LOCATION, config['directory'])
    event_dir = os.path.join(proc_dir, 'Events')

    points = []
    counts = {}
    runnumbers = config.get('runnumbers')
    runnumber_filter = None if runnumbers is None else set(str(runnum) for runnum in runnumbers)
    excluded_runnumbers = set(str(runnum) for runnum in config.get('exclude_runnumbers', []))
    prefix = 'run_' + config['name'] + '_'
    pattern = os.path.join(event_dir, prefix + '*')

    for rundir in sorted(glob.glob(pattern)):
        if not os.path.isdir(rundir):
            continue
        run_name = os.path.basename(rundir)
        rest = run_name[len(prefix):]
        parts = rest.split('_')
        if len(parts) != 3:
            continue
        runnum, c3_text, d4_text = parts
        if runnumber_filter is not None and runnum not in runnumber_filter:
            continue
        if runnum in excluded_runnumbers:
            continue
        try:
            c3 = float(c3_text)
            d4 = float(d4_text)
        except ValueError:
            continue

        lhefile = os.path.join(rundir, 'unweighted_events.lhe.gz')
        if not os.path.exists(lhefile):
            continue
        xsec = read_xsec(lhefile)
        xerr = read_integration_error(proc_dir, run_name, xsec)
        points.append((c3, d4, xsec, xerr))
        counts[runnum] = counts.get(runnum, 0) + 1

    if len(points) == 0:
        raise RuntimeError('No completed runs found for ' + config['name'])
    points.sort()
    return points, dict(sorted(counts.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]))


def build_design_matrix(points, terms):
    return np.asarray([chebyshev_row(c3, d4, terms) for c3, d4, _, _ in points], dtype=float)


def fit_weighted(points, terms):
    design = build_design_matrix(points, terms)
    values = np.asarray([point[2] for point in points], dtype=float)
    errors = np.asarray([point[3] for point in points], dtype=float)
    errors = np.where(errors > 0.0, errors, np.maximum(np.abs(values) * 0.01, 1e-30))

    fit_design = design / errors[:, None]
    fit_values = values / errors
    coeffs, _, rank, _ = np.linalg.lstsq(fit_design, fit_values, rcond=None)

    residuals = values - np.dot(design, coeffs)
    dof = max(len(values) - len(coeffs), 1)
    chi2_dof = float(np.dot(residuals / errors, residuals / errors) / dof)
    cov = max(chi2_dof, 1.0) * np.linalg.pinv(np.dot(fit_design.T, fit_design))
    cond = np.linalg.cond(design)
    sigma_sm = float(np.dot(chebyshev_row(0.0, 0.0, terms), coeffs))
    return coeffs, cov, rank, cond, chi2_dof, sigma_sm


def chebyshev_to_coupling_transform(terms):
    term_index = {term: i for i, term in enumerate(terms)}
    transform = np.zeros((len(terms), len(terms)), dtype=float)

    for col, (i3, i4) in enumerate(terms):
        c3_poly = chebyshev_t_coupling_coefficients(i3, K3_FIT_RANGE)
        d4_poly = chebyshev_t_coupling_coefficients(i4, K4_FIT_RANGE)
        for p3, c3_coeff in enumerate(c3_poly):
            for p4, d4_coeff in enumerate(d4_poly):
                row = term_index.get((p3, p4))
                if row is not None:
                    transform[row, col] += c3_coeff * d4_coeff
    return transform


def format_coupling_term(term):
    p3, p4 = term
    pieces = []
    if p3 == 1:
        pieces.append('c3')
    elif p3 > 1:
        pieces.append('c3**' + str(p3))
    if p4 == 1:
        pieces.append('d4')
    elif p4 > 1:
        pieces.append('d4**' + str(p4))
    if len(pieces) == 0:
        return '1'
    return '*'.join(pieces)


def print_normalized_fit(config, coeffs, cov):
    terms = config['terms']
    transform = chebyshev_to_coupling_transform(terms)
    coupling_coeffs = np.dot(transform, coeffs)

    sigma_zero = coupling_coeffs[0]
    jac = transform / sigma_zero - np.outer(coupling_coeffs, transform[0, :]) / (sigma_zero ** 2)
    norm_coeffs = coupling_coeffs / sigma_zero
    norm_cov = np.dot(jac, np.dot(cov, jac.T))
    norm_errors = np.sqrt(np.maximum(np.diag(norm_cov), 0.0))

    print('  Full normalized c3,d4 fit:')
    for term, coeff, err in zip(terms, norm_coeffs, norm_errors):
        print('    ' + format_coupling_term(term) + ' ' + str(round_sig(coeff, 6)) + ' +- ' + str(round_sig(err, 3)))


def evaluate_ratio(coeffs, terms):
    c3_values = np.linspace(C3_PLOT_RANGE[0], C3_PLOT_RANGE[1], PLOT_NC3)
    d4_values = np.linspace(D4_PLOT_RANGE[0], D4_PLOT_RANGE[1], PLOT_ND4)
    c3_grid, d4_grid = np.meshgrid(c3_values, d4_values)

    k3_grid = 1.0 + c3_grid
    k4_grid = 1.0 + d4_grid
    x3 = scale_to_chebyshev(k3_grid, K3_FIT_RANGE)
    x4 = scale_to_chebyshev(k4_grid, K4_FIT_RANGE)

    max_i = max(i for i, _ in terms)
    max_j = max(j for _, j in terms)
    t3 = [chebyshev_t(i, x3) for i in range(max_i + 1)]
    t4 = [chebyshev_t(j, x4) for j in range(max_j + 1)]

    sigma = np.zeros_like(c3_grid, dtype=float)
    for coeff, (i, j) in zip(coeffs, terms):
        sigma += coeff * t3[i] * t4[j]

    sigma_sm = float(np.dot(chebyshev_row(0.0, 0.0, terms), coeffs))
    return c3_grid, d4_grid, sigma / sigma_sm


def re_a0_partialwave(s, c3_grid, d4_grid):
    k3_squared = (1.0 + c3_grid) ** 2
    k4 = 1.0 + d4_grid
    mh2 = UNITARITY_MH ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        prefactor = (
            3.0
            * mh2
            * np.sqrt(s ** 2 - 4.0 * mh2 * s)
            / (32.0 * np.pi * s * (s - mh2) * UNITARITY_V ** 2)
        )
        bracket = (
            -k4 * (s - mh2)
            - 3.0 * k3_squared * mh2
            + (
                6.0
                * k3_squared
                * mh2
                * (s - mh2)
                / (s - 4.0 * mh2)
                * np.log(s / mh2 - 3.0)
            )
        )
        value = np.abs(prefactor * bracket)
    return np.where(np.isfinite(value), value, 0.0)


def get_unitarity_grid():
    global _UNITARITY_GRID
    if _UNITARITY_GRID is not None:
        return _UNITARITY_GRID

    c3_values = np.linspace(C3_PLOT_RANGE[0], C3_PLOT_RANGE[1], UNITARITY_NC3)
    d4_values = np.linspace(D4_PLOT_RANGE[0], D4_PLOT_RANGE[1], UNITARITY_ND4)
    c3_grid, d4_grid = np.meshgrid(c3_values, d4_values)
    max_partial_wave = np.zeros_like(c3_grid, dtype=float)

    for sqrt_s in UNITARITY_SQRTS:
        current = re_a0_partialwave(sqrt_s ** 2, c3_grid, d4_grid)
        max_partial_wave = np.maximum(max_partial_wave, current)

    _UNITARITY_GRID = c3_grid, d4_grid, max_partial_wave
    return _UNITARITY_GRID


def overlay_unitarity_contour(ax):
    c3_grid, d4_grid, partial_wave = get_unitarity_grid()
    return ax.contour(
        c3_grid,
        d4_grid,
        partial_wave,
        levels=[UNITARITY_LEVEL],
        colors='black',
        linestyles='--',
        linewidths=1.7,
    )


def make_log_levels(ratio):
    positive = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    if len(positive) == 0:
        raise RuntimeError('No positive normalized cross section values found')
    min_value = float(np.min(positive))
    max_value = float(np.max(positive))
    lo = math.floor(math.log10(min_value))
    hi = math.ceil(math.log10(max_value))
    nlevels = min(max((hi - lo) * 4 + 1, 12), 80)
    return np.logspace(lo, hi, nlevels)


def make_line_levels(filled_levels):
    lo = math.floor(math.log10(filled_levels[0]))
    hi = math.ceil(math.log10(filled_levels[-1]))
    levels = []
    for power in range(lo, hi + 1):
        value = 10.0 ** power
        if filled_levels[0] <= value <= filled_levels[-1]:
            levels.append(value)
    return levels


def format_level(value):
    if value >= 1000.0 or value < 0.01:
        return '%.0e' % value
    if value >= 10.0:
        return '%.0f' % value
    return '%.2g' % value


def plot_contour(config, coeffs, rank, cond, chi2_dof, sigma_sm, npoints, with_unitarity=False):
    c3_grid, d4_grid, ratio = evaluate_ratio(coeffs, config['terms'])
    ratio_positive = np.ma.masked_where(ratio <= 0.0, ratio)
    levels = make_log_levels(ratio)
    line_levels = make_line_levels(levels)

    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    contour = ax.contourf(
        c3_grid,
        d4_grid,
        ratio_positive,
        levels=levels,
        norm=colors.LogNorm(vmin=levels[0], vmax=levels[-1]),
        cmap='viridis',
        extend='both',
    )
    if np.any(ratio <= 0.0):
        ax.contourf(c3_grid, d4_grid, ratio <= 0.0, levels=[0.5, 1.5], colors=['0.75'], alpha=0.8)

    lines = ax.contour(c3_grid, d4_grid, ratio_positive, levels=line_levels, colors='white', linewidths=0.55)
    ax.clabel(lines, fmt=format_level, inline=True, fontsize=10)
    if with_unitarity:
        overlay_unitarity_contour(ax)

    ax.plot([0.0], [0.0], marker='o', color='white', markeredgecolor='black', markersize=5)

    ax.set_xlim(C3_PLOT_RANGE)
    ax.set_ylim(D4_PLOT_RANGE)
    ax.set_xlabel(r'$c_3$', fontsize=18)
    ax.set_ylabel(r'$d_4$', fontsize=18)
    ax.set_title(config['title'] + ' at ' + ENERGY_LABEL + r': $\sigma(c_3,d_4)/\sigma(0,0)$', fontsize=20)
    ax.tick_params(axis='both', labelsize=15)

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r'$\sigma(c_3,d_4)/\sigma(0,0)$', fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    suffix = '_unitarity' if with_unitarity else ''
    png = os.path.join(ROOT, config['outfile_base'] + suffix + '.png')
    pdf = os.path.join(ROOT, config['outfile_base'] + suffix + '.pdf')
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf, float(np.nanmin(ratio)), float(np.nanmax(ratio))


def main():
    for config in PROCESSES:
        points, counts = read_runs(config)
        wide_runnum = config.get('wide_runnum', '3')
        wide_count = counts.get(wide_runnum, 0)
        if wide_count < config['expected_wide_runs']:
            message = (
                config['name']
                + ' has only '
                + str(wide_count)
                + ' completed wide run-'
                + wide_runnum
                + ' points; expected '
                + str(config['expected_wide_runs'])
            )
            if REQUIRE_WIDE_RUNS:
                raise RuntimeError(message)
            print('Warning:', message)

        coeffs, cov, rank, cond, chi2_dof, sigma_sm = fit_weighted(points, config['terms'])
        png, pdf, rmin, rmax = plot_contour(config, coeffs, rank, cond, chi2_dof, sigma_sm, len(points))
        unit_png, unit_pdf, _, _ = plot_contour(
            config,
            coeffs,
            rank,
            cond,
            chi2_dof,
            sigma_sm,
            len(points),
            with_unitarity=True,
        )

        print(config['name'])
        print('  fit points:', len(points))
        print('  run counts:', counts)
        print('  rank:', str(rank) + '/' + str(len(config['terms'])))
        print('  condition:', round_sig(cond, 4))
        print('  chi2/dof:', round_sig(chi2_dof, 4))
        print('  sigma(c3=0,d4=0) [pb]:', round_sig(sigma_sm, 6))
        print_normalized_fit(config, coeffs, cov)
        print('  plotted ratio range:', round_sig(rmin, 4), 'to', round_sig(rmax, 4))
        print('  wrote:', png)
        print('  wrote:', pdf)
        print('  wrote:', unit_png)
        print('  wrote:', unit_pdf)


if __name__ == '__main__':
    main()
