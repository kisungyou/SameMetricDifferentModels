"""Portable manuscript figures; no LaTeX installation is needed.

Data, alignment, panel dimensions and scores follow the manuscript. DejaVu Sans
replaces its platform-specific Arial/TeX rendering. The original figures are
preserved in reference/manuscript_figures for comparison.
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
WIDTH_MM, HEIGHT_MM = 141.0, 124.0
EXACT_PANEL_SCALE = 0.8
EXACT_ORIGINAL_PANEL_MM = (WIDTH_MM - (0.52 + 0.10 + 0.25) * 25.4) / 2
EXACT_HEIGHT_MM = 80.0 - (1 - EXACT_PANEL_SCALE) * EXACT_ORIGINAL_PANEL_MM
SCALES = (0.5, 1.0, 2.0)


def configure():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "text.usetex": False, "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": None,
    })


def align_positions(parameters, n, m, party):
    """Apply the original common rotations; no rescaling or translation."""
    original = {q: parameters[f"q{q}_x"][n+m:].reshape(n+m, 2) for q in (1, 2)}
    _, _, vt = np.linalg.svd(original[1][:n] - original[1][:n].mean(axis=0), full_matrices=False)
    rotation1 = vt.T
    reference = original[1] @ rotation1
    if reference[:n][party == 200, 0].mean() < reference[:n][party == 100, 0].mean():
        rotation1[:, 0] *= -1
    reference = original[1] @ rotation1
    idx = np.argmax(np.abs(reference[:n, 1]))
    if reference[idx, 1] < 0:
        rotation1[:, 1] *= -1
    reference = original[1] @ rotation1
    u, _, vt = np.linalg.svd(original[2].T @ reference)
    rotations = {1: rotation1, 2: u @ vt}
    aligned = {q: original[q] @ rotations[q] for q in (1, 2)}
    for q in (1, 2):
        if not np.allclose(rotations[q].T @ rotations[q], np.eye(2), atol=1e-12):
            raise ValueError("Display transformation is not orthogonal")
        before = np.linalg.norm(original[q][:n, None] - original[q][None, n:], axis=2)
        after = np.linalg.norm(aligned[q][:n, None] - aligned[q][None, n:], axis=2)
        if not np.allclose(before, after, atol=1e-11, rtol=1e-11):
            raise ValueError("Display changed actor-item distances")
    return original, aligned, rotations


def load_inputs(results_dir, data_dir):
    results_dir, data_dir = Path(results_dir), Path(data_dir)
    with np.load(results_dir / "fit_comparison_results.npz", allow_pickle=False) as f:
        parameters = {key: f[key].copy() for key in f.files}
    with np.load(data_dir / "fit_comparison_split.npz", allow_pickle=False) as f:
        data = {key: f[key].copy() for key in f.files}
    members = pd.read_csv(data_dir / "fit_comparison_members.csv")
    if not np.array_equal(members.icpsr, data["member_ids"]):
        raise ValueError("Senator IDs are out of order")
    party = members.party_code.to_numpy()
    n, m = data["y"].shape
    original, aligned, rotations = align_positions(parameters, n, m, party)
    position_file = results_dir / "figure2_positions.npz"
    if position_file.exists():
        with np.load(position_file, allow_pickle=False) as f:
            for q in (1, 2):
                if not np.array_equal(f[f"q{q}_original"], original[q]):
                    raise ValueError("Archived display does not match selected fit")
                if not np.allclose(f[f"q{q}_displayed"], aligned[q], atol=1e-11, rtol=1e-11):
                    raise ValueError("Recomputed display differs from archived display")
                aligned[q] = f[f"q{q}_displayed"].copy()
                rotations[q] = f[f"q{q}_rotation"].copy()
    rows = []
    for sd in SCALES:
        for q in (2, 1):
            prefix = f"q{q}" if sd == 1 else f"q{q}_sd{sd}"
            x = parameters[prefix + "_x"]
            a, b = x[:n], x[n:n+m]
            zw = x[n+m:].reshape(n+m, 2)
            distance = np.linalg.norm(zw[:n, None] - zw[None, n:], axis=2)
            eta = a[:, None] + b[None, :] - distance**q
            loss = np.logaddexp(0., eta) - data["y"] * eta
            rows.append({"coordinate_sd": sd, "q": q,
                         "training_log_loss": float(loss[data["train"]].mean()),
                         "held_out_log_loss": float(loss[data["test"]].mean()),
                         "held_out_brier": float(((expit(eta)-data["y"])**2)[data["test"]].mean())})
    table = pd.DataFrame(rows)
    losses = {q: table.loc[table.q.eq(q), "held_out_log_loss"].tolist() for q in (1, 2)}
    return aligned, losses, party, n, table


def draw(aligned, losses, party, n):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    width, height = WIDTH_MM / 25.4, HEIGHT_MM / 25.4
    fig = plt.figure(figsize=(width, height))
    left, right_margin, gap = 0.52, 0.10, 0.25
    side = (width - left - right_margin - gap) / 2
    map_bottom = 2.33
    points = np.r_[aligned[1], aligned[2]]
    lower, upper = points.min(axis=0), points.max(axis=0)
    center, half = (lower + upper) / 2, 0.56 * np.max(upper - lower)
    xlim = (center[0] - half, center[0] + half)
    ylim = (center[1] - half, center[1] + half)
    colors = {100: "#2378b5", 200: "#c9524c", 328: "#a87924"}
    marks = {100: "o", 200: "s", 328: "D"}
    for column, (q, label, title) in enumerate(((2, "A", "Quadratic"), (1, "B", "Linear"))):
        ax = fig.add_axes([(left + column * (side + gap)) / width,
                           map_bottom / height, side / width, side / height])
        xy = aligned[q]
        ax.scatter(xy[n:, 0], xy[n:, 1], s=8, c="#a9adb3", marker="x",
                   linewidths=0.45, alpha=0.55, zorder=1)
        for pc in (100, 200, 328):
            selected = party == pc
            ax.scatter(xy[:n][selected, 0], xy[:n][selected, 1],
                       s=18 if pc != 328 else 23, c=colors[pc], marker=marks[pc],
                       edgecolors="white", linewidths=0.35, alpha=0.95, zorder=3)
        ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="Latent coordinate 1")
        ax.set_xticks([-2, 0, 2], labels=["−2", "0", "2"])
        ax.set_yticks([-2, 0, 2], labels=["−2", "0", "2"])
        ax.set_title(rf"({label}) {title} ($q={q}$)", loc="left", pad=6)
        ax.grid(color="#e9ebee", linewidth=0.45)
        ax.set_axisbelow(True)
        if column == 0:
            ax.set_ylabel("Latent coordinate 2")
        else:
            ax.tick_params(labelleft=False)
    party_handles = [Line2D([0], [0], marker=marks[pc], color="none",
                           markerfacecolor=colors[pc], markeredgecolor="white",
                           markersize=4.8, label=label)
                     for pc, label in ((100, "Democrat"), (200, "Republican"), (328, "Independent"))]
    party_handles.append(Line2D([0], [0], marker="x", color="#a9adb3", linewidth=0,
                                markersize=4.5, label="Roll call"))
    fig.legend(handles=party_handles, loc="center",
               bbox_to_anchor=((left + (width - left - right_margin) / 2) / width, 1.83 / height),
               ncol=4, frameon=False, handletextpad=0.3, columnspacing=1.1, handlelength=0.8)

    ax = fig.add_axes([left / width, 0.48 / height,
                       (width - left - right_margin) / width, 1.0 / height])
    model_colors = {2: "#424851", 1: "#178571"}
    for q, marker in ((2, "s"), (1, "o")):
        ax.plot(SCALES, losses[q], marker=marker, markersize=4.3,
                color=model_colors[q], linewidth=1.2,
                label=rf"$q={q}$")
        for sd, value in zip(SCALES, losses[q]):
            ax.annotate(f"{value:.3f}", (sd, value), textcoords="offset points",
                        xytext=(0, 7 if q == 1 else -12), ha="center",
                        fontsize=8.5, color=model_colors[q])
    ax.set_xscale("log", base=2)
    ax.set_xticks(SCALES, labels=["0.5", "1", "2"])
    ax.set_yticks([0, 0.1, 0.2, 0.3], labels=["0.0", "0.1", "0.2", "0.3"])
    ax.set(xlim=(0.43, 2.32), ylim=(0, 0.38), xlabel=r"Coordinate prior SD, $s$")
    ax.set_title("(C) Held-out log loss", loc="left", pad=5)
    ax.grid(axis="y", color="#e9ebee", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.25,
              columnspacing=1.2, handletextpad=0.4, borderaxespad=0.25)
    return fig


def draw_supplement(aligned, losses, party, n, table):
    """Four-panel supplement: maps, log loss, and Brier scores."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(5.85, 6.4))
    points = np.r_[aligned[1], aligned[2]]
    lower, upper = points.min(axis=0), points.max(axis=0)
    center, half = (lower + upper) / 2, 0.56 * np.max(upper-lower)
    colors = {100: "#2378b5", 200: "#c9524c", 328: "#a87924"}
    marks = {100: "o", 200: "s", 328: "D"}
    for col, q in enumerate((2, 1)):
        ax = fig.add_axes([.12+.46*col, 3.65/6.4, .38, .38*5.85/6.4])
        xy = aligned[q]
        ax.scatter(xy[n:, 0], xy[n:, 1], s=9, c="#a9adb3", marker="x",
                   linewidths=.5, alpha=.55)
        for pc in (100, 200, 328):
            use = party == pc
            ax.scatter(xy[:n][use, 0], xy[:n][use, 1], s=22 if pc != 328 else 27,
                       c=colors[pc], marker=marks[pc], edgecolors="white", linewidths=.35)
        ax.set(xlim=(center[0]-half, center[0]+half),
               ylim=(center[1]-half, center[1]+half), aspect="equal",
               xlabel="Latent coordinate 1")
        ax.set_title(f"({'AB'[col]}) {'Quadratic' if q == 2 else 'Linear'}", loc="left", pad=8)
        ax.grid(color="#e9ebee", lw=.5)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("Latent coordinate 2")
        else:
            ax.tick_params(labelleft=False)
    handles = [Line2D([0], [0], marker=marks[p], color="none", markerfacecolor=colors[p],
                      markeredgecolor="white", markersize=5.5, label=label)
               for p, label in ((100, "Democrat"), (200, "Republican"), (328, "Independent"))]
    handles.append(Line2D([0], [0], marker="x", color="#a9adb3", lw=0, markersize=5, label="Roll call"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.53, 2.85/6.4),
               frameon=False, ncol=4, handletextpad=.35, columnspacing=.9, handlelength=.8)
    model_colors = {2: "#424851", 1: "#178571"}
    for col, key, title in ((0, "held_out_log_loss", "(C) Log loss"),
                            (1, "held_out_brier", "(D) Brier score")):
        ax = fig.add_axes([.12+.46*col, .65/6.4, .38, 1.63/6.4])
        max_score = 0
        for q, marker in ((2, "s"), (1, "o")):
            values = table.loc[table.q.eq(q), key].to_numpy()
            ax.plot(SCALES, values, marker=marker, color=model_colors[q],
                    markersize=4.3, lw=1.25, label=rf"$q={q}$")
            max_score = max(max_score, max(values))
            for x, value in zip(SCALES, values):
                ax.annotate(f"{value:.3f}", (x, value), textcoords="offset points",
                            xytext=(0, 8 if q == 1 else -16), ha="center", color=model_colors[q])
        ax.set_xscale("log", base=2)
        ax.set_xticks(SCALES, labels=["0.5", "1", "2"])
        ax.set(xlim=(.43, 2.32), ylim=(0, max_score*1.32), xlabel="Coordinate prior SD  $s$")
        ax.set_title(title, loc="left", pad=9)
        ax.grid(axis="y", color="#e9ebee", lw=.6)
        ax.set_axisbelow(True)
        if col == 0:
            ax.legend(loc="lower left", frameon=False, ncol=2,
                      handlelength=1.2, columnspacing=.8, handletextpad=.4)
    return fig


def render_all(results_dir, data_dir, output_dir):
    """Verify fitted inputs and export Figures 1, 2, S1 and Table S1.

    Returns {figure_name: {'png': Path, 'pdf': Path}}. The reference inputs
    are never written; all exports are confined to the requested output folder.
    """
    from .checks import verify_results, exact_checks
    results_dir, data_dir, output_dir = map(lambda p: Path(p).resolve(), (results_dir, data_dir, output_dir))
    for protected in (ROOT / "reference", ROOT / "data", data_dir):
        if output_dir == protected or protected in output_dir.parents:
            raise ValueError("Figure output must not overwrite reference inputs or data")
    report = verify_results(results_dir, data_dir, compare_reference=False)
    if not report["passed"]:
        raise ValueError("Saved fit verification failed; inspect idealpoint.checks")
    configure()
    import matplotlib.pyplot as plt
    aligned, losses, party, n, table = load_inputs(results_dir, data_dir)
    exact = exact_checks()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "centered_logits": draw_exact(exact),
        "senate_illustration": draw(aligned, losses, party, n),
        "fit_comparison": draw_supplement(aligned, losses, party, n, table),
    }
    outputs = {}
    for name, figure in figures.items():
        outputs[name] = {}
        for ext in ("png", "pdf"):
            path = output_dir / f"{name}.{ext}"
            figure.savefig(path, dpi=220)
            outputs[name][ext] = path
        plt.close(figure)
    table.to_csv(output_dir / "table_s1.csv", index=False)
    rows = ["s & q & Training log loss & Held-out log loss & Held-out Brier \\\\"]
    rows += [f"{r.coordinate_sd:.1f} & {r.q:.0f} & {r.training_log_loss:.6f} & "
             f"{r.held_out_log_loss:.6f} & {r.held_out_brier:.6f} \\\\" for r in table.itertuples()]
    (output_dir / "table_s1.tex").write_text("\n".join(rows) + "\n")
    (output_dir / "figure_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return outputs


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reference")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "reference-figures")
    args = parser.parse_args()
    outputs = render_all(args.source, args.data, args.output)
    print(json.dumps(outputs, default=str, indent=2))


def draw_exact(report):
    """Display the saved rational matrices without recomputing the analysis."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    matrices = [[[Fraction(value) for value in row] for row in matrix]
                for matrix in report["centered_logit_matrices"]]
    assert report["ranks"] == [2, 1]
    assert len(matrices) == 2 and all(len(matrix) == 3 for matrix in matrices)
    width, height = WIDTH_MM / 25.4, EXACT_HEIGHT_MM / 25.4
    fig = plt.figure(figsize=(width, height))
    left, right_margin, gap = 0.52, 0.10, 0.25
    original_side = (width - left - right_margin - gap) / 2
    side = EXACT_PANEL_SCALE * original_side
    # Keep panel centers and all text sizes fixed on the 141 mm canvas.
    # Remove only the lost panel height from the canvas, preserving margins.
    left += (original_side - side) / 2
    gap += original_side - side
    cmap = LinearSegmentedColormap.from_list("signed", ["#28516b", "#fafafa", "#aa5d3f"])
    norm = Normalize(-2, 2)
    for index, (matrix, q, name) in enumerate(zip(matrices, (1, 2), ("Linear", "Quadratic"))):
        ax = fig.add_axes([(left + index * (side + gap)) / width,
                           0.48 / height, side / width, side / height])
        for i, row in enumerate(matrix):
            assert len(row) == 3
            for j, value in enumerate(row):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=cmap(norm(float(value))),
                                       edgecolor="white", linewidth=1.5))
                label = str(value) if value >= 0 else "−" + str(-value)
                ax.text(j, i, label, ha="center", va="center", fontsize=10,
                        color="white" if abs(value) > Fraction(3, 2) else "#172938")
        ax.set(xlim=(-0.5, 2.5), ylim=(2.5, -0.5), aspect="equal",
               xlabel="Item location")
        ax.set_xticks(range(3), labels=["−1", "0", "1"])
        ax.set_yticks(range(3), labels=["−1", "0", "1"])
        ax.set_title(rf"({'AB'[index]}) {name} ($q={q}$)", loc="left", pad=6)
        if index == 0:
            ax.set_ylabel("Actor location")
        for spine in ax.spines.values():
            spine.set_visible(False)
    return fig


if __name__ == "__main__":
    main()


