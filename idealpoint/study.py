"""Run the Senate study without modifying the archived manuscript results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]


def run(mode="smoke", output_dir=None):
    """Fit all six model/scale combinations and save scores and displays.

    ``full`` reproduces the manuscript's starts, tolerances and refinements.
    ``smoke`` uses a shortened optimization schedule only to check execution.
    Existing fit outputs are not silently overwritten.
    """
    from . import fit
    from .checks import verify_results
    from .figures import render_all
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    output = Path(output_dir or ROOT / "outputs" / f"{mode}-refit").resolve()
    if ROOT / "outputs" not in output.parents:
        raise ValueError("Choose a fresh directory within this repository's outputs/")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}. Choose a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    fit.configure_paths(ROOT / "data", output)
    design = fit.prepare_data()
    started = time.perf_counter()
    results = fit.fit_models(design, mode=mode)
    if mode == "full":
        results = fit.focused_refinement(results)
    fit.write_summary(results)
    report = verify_results(output, ROOT / "data", compare_reference=(mode == "full"))
    (output / "fit_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError(f"Generated results failed internal checks; see {output / 'fit_verification.json'}")
    render_all(output, ROOT / "data", output / "figures")
    record = {"mode": mode, "elapsed_seconds": time.perf_counter() - started,
              "internal_checks_passed": report["passed"],
              "manuscript_design": mode == "full"}
    (output / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    return results


def rebuild_data(output_dir=None):
    """Reconstruct processed inputs from the supplied raw snapshot in a new folder."""
    from . import fit
    output = Path(output_dir or ROOT / "outputs" / "preprocessed").resolve()
    if ROOT / "outputs" not in output.parents:
        raise ValueError("Rebuilt data must be written under outputs/")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("S114_votes.csv", "S114_members.csv", "S114_rollcalls.csv", "fit_comparison_design.json"):
        shutil.copy2(ROOT / "data" / name, output / name)
    fit.configure_paths(output, ROOT / "outputs" / "unused-fit")
    return fit.prepare_data(rebuild=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        design = rebuild_data(args.output)
        print(json.dumps({key: design[key] for key in ("n_members", "n_rollcalls", "n_train", "n_test")}, indent=2))
    else:
        run(args.mode, args.output)


if __name__ == "__main__":
    main()
