"""Standalone matched q=1 and q=2 approximate MAP comparison.

Only true Yea/Nay votes (cast codes 1/6) enter the analysis.  The fixed
preprocessing, split, seeds, priors, objective, and full fitting algorithm are
those used for the Senate illustration.  NOMINATE estimates are never used;
party labels enter only subsequent display and descriptive checks.

Call configure_paths(data_dir, output_dir), prepare_data(), fit_models(design),
focused_refinement(results), and write_summary(results).  Rendering is separate.
All inputs are bundled: this module never downloads data.  A smoke run is an
explicitly labeled software check and is not a replication of the estimates.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS_DIR = ROOT / "outputs" / "results"
RESULTS = RESULTS_DIR / "fit_comparison_results.json"
FIT_FILE = RESULTS_DIR / "fit_comparison_results.npz"
SPLIT_FILE = DATA / "fit_comparison_split.npz"
DESIGN_FILE = DATA / "fit_comparison_design.json"
MEMBERS_FILE = DATA / "fit_comparison_members.csv"
ROLLS_FILE = DATA / "fit_comparison_rollcalls.csv"

K = 2
GAMMA = 1.0
INTERCEPT_SD = 3.0
COORDINATE_SD = 1.0
SENSITIVITY_SDS = [0.5, 1.0, 2.0]
SPLIT_SEED = 20261023
HOLDOUT_FRACTION = 0.20
MIN_MEMBER_PARTICIPATION = 0.80
MIN_ROLL_PARTICIPATION = 0.80
MIN_TRAIN_MINOR_SHARE = 0.10
MIN_TRAIN_VOTES = 50
START_SEEDS = [104729, 130363, 155921, 181081, 206369, 231779, 257053, 282407]
MAXITER = 2500
GTOL = 1.0e-5
FTOL = 1.0e-11
SOURCES = {
    "S114_votes.csv": "https://voteview.com/static/data/out/votes/S114_votes.csv",
    "S114_members.csv": "https://voteview.com/static/data/out/members/S114_members.csv",
    "S114_rollcalls.csv": "https://voteview.com/static/data/out/rollcalls/S114_rollcalls.csv",
}



# Pin the bundled snapshot independently of mutable processed-data metadata.
SOURCE_HASHES = {
    "S114_votes.csv": "cf8fb57d99ecc32dc0a35ddb62fc578926479a80092e72ca70caad8e2fe435b1",
    "S114_members.csv": "aa575f1930125b3ca0e5e7bd068d09005a6895722631fcae1d69a562da2a100e",
    "S114_rollcalls.csv": "ef06eccacacf21aede6f91bdb5eb81d4426645fbb2aecb7e0bb760b17dd1c1c2",
}


def _local_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    candidate = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError(f"Path must be inside this replication repository: {candidate}")
    return candidate


def configure_paths(data_dir: str | Path | None = None,
                    output_dir: str | Path | None = None) -> None:
    """Select input and direct numerical-output directories within the repository.

    Relative paths are resolved from this repository's root.  No directories are
    created until a write is requested.  This module-level configuration is for
    sequential runs; concurrent runs should use separate Python processes.
    """
    global DATA, RESULTS_DIR, RESULTS, FIT_FILE, SPLIT_FILE, DESIGN_FILE
    global MEMBERS_FILE, ROLLS_FILE
    data_path = _local_path(data_dir if data_dir is not None else ROOT / "data")
    result_path = _local_path(output_dir if output_dir is not None else ROOT / "outputs" / "results")
    for protected in (ROOT / "data", ROOT / "reference", data_path):
        if result_path == protected or result_path.is_relative_to(protected):
            raise ValueError(f"Numerical outputs cannot overwrite inputs or reference files: {result_path}")
    DATA, RESULTS_DIR = data_path, result_path
    RESULTS = RESULTS_DIR / "fit_comparison_results.json"
    FIT_FILE = RESULTS_DIR / "fit_comparison_results.npz"
    SPLIT_FILE = DATA / "fit_comparison_split.npz"
    DESIGN_FILE = DATA / "fit_comparison_design.json"
    MEMBERS_FILE = DATA / "fit_comparison_members.csv"
    ROLLS_FILE = DATA / "fit_comparison_rollcalls.csv"


def _validate_raw_inputs() -> None:
    for name, expected in SOURCE_HASHES.items():
        path = DATA / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled input {path}; no network download is attempted.")
        if sha256(path) != expected:
            raise ValueError(f"Bundled raw input hash mismatch: {name}")



def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")

def _preprocess_raw():
    """Reconstruct the frozen array design using only the bundled raw votes."""
    votes = pd.read_csv(DATA / "S114_votes.csv")
    members = pd.read_csv(DATA / "S114_members.csv")
    rolls = pd.read_csv(DATA / "S114_rollcalls.csv")
    members = members.loc[members.chamber.eq("Senate")].copy()
    members.icpsr = members.icpsr.astype(int)
    members = members.drop_duplicates("icpsr").sort_values("icpsr")
    votes = votes.loc[votes.chamber.eq("Senate") & votes.cast_code.isin([1, 6])].copy()
    votes.icpsr = votes.icpsr.astype(int)
    assert not votes.duplicated(["icpsr", "rollnumber"]).any()
    all_ids = members.icpsr.to_numpy()
    all_rolls = np.sort(rolls.loc[rolls.chamber.eq("Senate"), "rollnumber"].unique())
    observed_table = votes.assign(y=(votes.cast_code == 1).astype(float)).pivot(
        index="icpsr", columns="rollnumber", values="y"
    ).reindex(index=all_ids, columns=all_rolls)
    y_all = observed_table.to_numpy()
    observed_all = np.isfinite(y_all)
    # Participation depends on the existence of a cast vote, not its Yea/Nay value.
    keep_members = observed_all.mean(axis=1) >= MIN_MEMBER_PARTICIPATION
    y = y_all[keep_members]
    observed = np.isfinite(y)
    ids = all_ids[keep_members]
    rng = np.random.default_rng(SPLIT_SEED)
    # A fixed Bernoulli mask is drawn once on the full retained-member grid.
    train = observed & (rng.random(y.shape) >= HOLDOUT_FRACTION)
    test = observed & ~train
    train_count = train.sum(axis=0)
    train_yes = (np.nan_to_num(y) * train).sum(axis=0)
    train_yes_share = np.divide(train_yes, train_count, out=np.zeros_like(train_yes),
                                where=train_count > 0)
    # Outcome-based roll filtering uses TRAINING votes only.
    keep_rolls = (
        (observed.mean(axis=0) >= MIN_ROLL_PARTICIPATION)
        & (train_count >= MIN_TRAIN_VOTES)
        & (train_yes_share >= MIN_TRAIN_MINOR_SHARE)
        & (train_yes_share <= 1 - MIN_TRAIN_MINOR_SHARE)
    )
    roll_ids = all_rolls[keep_rolls]
    y, train, test = y[:, keep_rolls], train[:, keep_rolls], test[:, keep_rolls]
    observed = np.isfinite(y)
    assert np.all(train.sum(axis=0) > 0) and np.all(train.sum(axis=1) > 0)
    assert not np.any(train & test)
    assert np.array_equal(train | test, observed)
    y = np.nan_to_num(y, nan=0.0)
    return y, train, test, ids, roll_ids, observed, members, rolls, all_ids, all_rolls


def _write_preprocessed(data) -> dict:
    y, train, test, ids, roll_ids, observed, members, rolls, all_ids, all_rolls = data
    np.savez_compressed(SPLIT_FILE, y=y, train=train, test=test, member_ids=ids,
                        roll_ids=roll_ids)
    # Export ONLY descriptors, excluding any externally estimated ideal points.
    md = members.set_index("icpsr").loc[ids, ["bioname", "state_abbrev", "party_code"]]
    md.to_csv(MEMBERS_FILE)
    rd = rolls.set_index("rollnumber").loc[roll_ids]
    allowed = [c for c in ["date", "bill_number", "vote_desc", "vote_question", "vote_result"]
               if c in rd.columns]
    rd[allowed].to_csv(ROLLS_FILE)
    long_i, long_j = np.where(observed)
    pd.DataFrame({
        "icpsr": ids[long_i], "rollnumber": roll_ids[long_j],
        "yea": y[long_i, long_j].astype(int),
        "role": np.where(train[long_i, long_j], "train", "test"),
    }).to_csv(DATA / "fit_comparison_votes.csv", index=False)
    design = {
        "description": "Frozen matched MAP comparison; not original-paper replication",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_citation": "Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and Luke Sonnet (2026). Voteview: Congressional Roll-Call Votes Database. https://voteview.com/",
        "source_documentation": ["https://voteview.com/data",
                                 "https://voteview.com/articles/data_help_votes"],
        "sources": {name: {"url": url, "sha256": sha256(DATA / name)}
                    for name, url in SOURCES.items()},
        "congress": 114, "chamber": "Senate", "original_members": int(len(all_ids)),
        "original_rollcalls": int(len(all_rolls)), "n_members": int(len(ids)),
        "n_rollcalls": int(len(roll_ids)), "n_observed": int(observed.sum()),
        "n_missing": int((~observed).sum()), "n_train": int(train.sum()),
        "n_test": int(test.sum()), "test_fraction_of_observed": float(test.sum()/observed.sum()),
        "date_min": str(rd.date.min()), "date_max": str(rd.date.max()),
        "party_counts": {str(int(k)): int(v) for k, v in md.party_code.value_counts().items()},
        "preprocessing": {
            "cast_code_1": 1, "cast_code_6": 0,
            "other_cast_codes": "excluded, not imputed as Nay",
            "president_records": "excluded by chamber == Senate",
            "member_min_participation_all_rolls": MIN_MEMBER_PARTICIPATION,
            "roll_min_participation_retained_members": MIN_ROLL_PARTICIPATION,
            "roll_min_training_votes": MIN_TRAIN_VOTES,
            "roll_min_training_minority_share": MIN_TRAIN_MINOR_SHARE,
            "roll_filter_outcomes": "training outcomes only; held-out Yea/Nay unused",
            "external_ideal_points_or_probabilities": "never used",
            "party_labels": "display and orientation only; not used in estimation",
        },
        "split": {"seed": SPLIT_SEED, "test_probability": HOLDOUT_FRACTION,
                  "rule": "independent uniform draw on sorted member-by-roll grid before training-only roll filtering"},
        "model": {
            "log_odds": "a_i + b_j - gamma * ||z_i - w_j||_2 ** q",
            "q": [1, 2], "K": K, "gamma_fixed": GAMMA,
            "intercept_prior": {"distribution": "independent Normal", "mean": 0, "sd": INTERCEPT_SD},
            "coordinate_prior": {"distribution": "independent Normal coordinates", "mean": 0, "sd": COORDINATE_SD},
            "objective": "sum_train(log(1+exp(eta_ij))-y_ij*eta_ij) + sum(a_i^2+b_j^2)/(2*3^2) + sum(||z_i||^2+||w_j||^2)/2",
            "prior_warning": "Identical parameter priors do not induce identical probability priors across q.",
            "q1_numerics": "exact Euclidean norm; no smoothing; zero vector chosen as subgradient at exact coincidence",
        },
        "optimization_plan": {"method": "L-BFGS-B", "starts": len(START_SEEDS),
                              "seeds": START_SEEDS, "maxiter": MAXITER, "gtol": GTOL, "ftol": FTOL,
                              "selection": "lowest training penalized negative log likelihood only",
                              "initialization": "start 0 training-residual SVD, start 1 perturbed SVD, remaining starts independent Gaussian coordinates; shared seed and coordinates across q"},
        "sensitivity_plan": {"coordinate_sd": SENSITIVITY_SDS,
                             "starts_at_each_other_sd": "selected s=1 fit plus shared random-coordinate start 2",
                             "selection": "lowest training penalized objective at each scale; all three scales reported; no holdout-based selection"},
        "primary_metrics": ["held-out mean negative log likelihood", "held-out Brier score"],
        "interpretation": "One data set and one common split; descriptive comparison only, no significance or general superiority claim.",
        "split_sha256": sha256(SPLIT_FILE),
    }
    save_json(DESIGN_FILE, design)
    return design

def _validate_processed(design: dict, rebuilt) -> None:
    """Check hashes and reconstruct every split array and exported descriptor."""
    y, train, test, ids, roll_ids, observed, members, rolls, all_ids, all_rolls = rebuilt
    for name, digest in SOURCE_HASHES.items():
        if design["sources"][name]["sha256"] != digest:
            raise ValueError(f"Design source hash does not match the bundled snapshot: {name}")
    if sha256(SPLIT_FILE) != design["split_sha256"]:
        raise ValueError("Frozen split hash does not match its design metadata.")
    expected_arrays = dict(y=y, train=train, test=test, member_ids=ids, roll_ids=roll_ids)
    with np.load(SPLIT_FILE, allow_pickle=False) as saved:
        if set(saved.files) != set(expected_arrays):
            raise ValueError("Frozen split has unexpected or missing arrays.")
        for key, expected in expected_arrays.items():
            if saved[key].dtype != expected.dtype or not np.array_equal(saved[key], expected):
                raise ValueError(f"Frozen split array differs from raw-data reconstruction: {key}")
    md = members.set_index("icpsr").loc[ids, ["bioname", "state_abbrev", "party_code"]]
    rd = rolls.set_index("rollnumber").loc[roll_ids]
    allowed = [c for c in ["date", "bill_number", "vote_desc", "vote_question", "vote_result"]
               if c in rd.columns]
    pd.testing.assert_frame_equal(pd.read_csv(MEMBERS_FILE).set_index("icpsr"), md,
                                  check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_csv(ROLLS_FILE).set_index("rollnumber"), rd[allowed],
                                  check_dtype=False)
    long_i, long_j = np.where(observed)
    expected_votes = pd.DataFrame({
        "icpsr": ids[long_i], "rollnumber": roll_ids[long_j],
        "yea": y[long_i, long_j].astype(int),
        "role": np.where(train[long_i, long_j], "train", "test"),
    })
    pd.testing.assert_frame_equal(pd.read_csv(DATA / "fit_comparison_votes.csv"), expected_votes,
                                  check_dtype=False)
    expected_counts = {
        "congress": 114, "chamber": "Senate", "original_members": len(all_ids),
        "original_rollcalls": len(all_rolls), "n_members": len(ids),
        "n_rollcalls": len(roll_ids), "n_observed": int(observed.sum()),
        "n_missing": int((~observed).sum()), "n_train": int(train.sum()),
        "n_test": int(test.sum()), "date_min": str(rd.date.min()), "date_max": str(rd.date.max()),
        "party_counts": {str(int(k)): int(v) for k, v in md.party_code.value_counts().items()},
        "test_fraction_of_observed": float(test.sum() / observed.sum()),
    }
    for key, expected in expected_counts.items():
        if design[key] != expected:
            raise ValueError(f"Design metadata disagrees with reconstructed input: {key}")
    expected_choices = [
        (design["split"]["seed"], SPLIT_SEED),
        (design["split"]["test_probability"], HOLDOUT_FRACTION),
        (design["preprocessing"]["member_min_participation_all_rolls"], MIN_MEMBER_PARTICIPATION),
        (design["preprocessing"]["roll_min_participation_retained_members"], MIN_ROLL_PARTICIPATION),
        (design["preprocessing"]["roll_min_training_votes"], MIN_TRAIN_VOTES),
        (design["preprocessing"]["roll_min_training_minority_share"], MIN_TRAIN_MINOR_SHARE),
        (design["model"]["K"], K), (design["model"]["gamma_fixed"], GAMMA),
        (design["model"]["intercept_prior"]["sd"], INTERCEPT_SD),
        (design["model"]["coordinate_prior"]["sd"], COORDINATE_SD),
        (design["optimization_plan"]["seeds"], START_SEEDS),
        (design["optimization_plan"]["maxiter"], MAXITER),
        (design["optimization_plan"]["gtol"], GTOL),
        (design["optimization_plan"]["ftol"], FTOL),
        (design["sensitivity_plan"]["coordinate_sd"], SENSITIVITY_SDS),
    ]
    if any(actual != expected for actual, expected in expected_choices):
        raise ValueError("Design choices differ from this replication's fixed analysis plan.")


def prepare_data(rebuild: bool = False) -> dict:
    """Validate a frozen split, or explicitly rebuild it from bundled raw inputs.

    Rebuilding is allowed only in a generated-data directory: copy the bundled
    raw inputs into outputs/preprocessed and configure that directory first.
    The original data/ and reference/ snapshots are protected.  Rebuilding
    preserves the numerical split, while recording a new processing timestamp
    and the digest of the newly written NPZ container.
    """
    _validate_raw_inputs()
    if rebuild:
        if any(DATA == p or DATA.is_relative_to(p) for p in (ROOT / "data", ROOT / "reference")):
            raise ValueError("Rebuild in a generated directory such as outputs/preprocessed, not bundled data/.")
    required = (DESIGN_FILE, SPLIT_FILE, MEMBERS_FILE, ROLLS_FILE,
                DATA / "fit_comparison_votes.csv")
    if not rebuild:
        missing = [p.name for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing processed inputs: {', '.join(missing)}; explicitly rebuild from raw inputs.")
    rebuilt = _preprocess_raw()
    design = _write_preprocessed(rebuilt) if rebuild else json.loads(DESIGN_FILE.read_text())
    _validate_processed(design, rebuilt)
    return design


def unpack(x: np.ndarray, n: int, m: int):
    a, b = x[:n], x[n:n+m]
    z = x[n+m:n+m+n*K].reshape(n, K)
    w = x[n+m+n*K:].reshape(m, K)
    return a, b, z, w

def logits(x: np.ndarray, n: int, m: int, q: int) -> np.ndarray:
    a, b, z, w = unpack(x, n, m)
    delta = z[:, None, :] - w[None, :, :]
    distance2 = np.einsum("ijk,ijk->ij", delta, delta)
    distance_q = np.sqrt(distance2) if q == 1 else distance2
    return a[:, None] + b[None, :] - GAMMA * distance_q

def objective_gradient(x: np.ndarray, y: np.ndarray, train: np.ndarray, q: int,
                       coordinate_sd: float = COORDINATE_SD):
    n, m = y.shape
    a, b, z, w = unpack(x, n, m)
    delta = z[:, None, :] - w[None, :, :]
    distance2 = np.einsum("ijk,ijk->ij", delta, delta)
    if q == 1:
        distance = np.sqrt(distance2)
        # No epsilon is added to the distance or objective.  The zero subgradient
        # is explicitly selected at exact coincidences (the norm is nonsmooth).
        direction = np.divide(delta, distance[:, :, None], out=np.zeros_like(delta),
                              where=distance[:, :, None] > 0)
    else:
        distance = distance2
        direction = 2.0 * delta
    eta = a[:, None] + b[None, :] - GAMMA * distance
    terms = np.logaddexp(0.0, eta) - y * eta
    value = terms[train].sum() + (a@a + b@b) / (2 * INTERCEPT_SD**2)
    value += ((z*z).sum() + (w*w).sum()) / (2 * coordinate_sd**2)
    error = (expit(eta) - y) * train
    pair_gradient = GAMMA * error[:, :, None] * direction
    ga = error.sum(axis=1) + a / INTERCEPT_SD**2
    gb = error.sum(axis=0) + b / INTERCEPT_SD**2
    gz = -pair_gradient.sum(axis=1) + z / coordinate_sd**2
    gw = pair_gradient.sum(axis=0) + w / coordinate_sd**2
    return float(value), np.concatenate([ga, gb, gz.ravel(), gw.ravel()])

def intercept_fit(y: np.ndarray, train: np.ndarray):
    n, m = y.shape
    def fun(ab):
        a, b = ab[:n], ab[n:]
        eta = a[:, None] + b[None, :]
        e = (expit(eta)-y) * train
        value = (np.logaddexp(0, eta)-y*eta)[train].sum()
        value += (ab@ab)/(2*INTERCEPT_SD**2)
        gradient = np.r_[e.sum(axis=1), e.sum(axis=0)] + ab/INTERCEPT_SD**2
        return value, gradient
    r = minimize(fun, np.zeros(n+m), jac=True, method="L-BFGS-B",
                 options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-6})
    return r.x[:n], r.x[n:]

def initializations(y: np.ndarray, train: np.ndarray):
    n, m = y.shape
    a, b = intercept_fit(y, train)
    residual = (y-expit(a[:,None]+b[None,:])) * train
    u, s, vt = np.linalg.svd(residual, full_matrices=False)
    z_svd = u[:, :K]*np.sqrt(s[:K])
    w_svd = vt[:K].T*np.sqrt(s[:K])
    factor = 0.65/np.sqrt(np.mean(np.r_[z_svd.ravel(), w_svd.ravel()]**2))
    z_svd, w_svd = factor*z_svd, factor*w_svd
    starts = []
    for k, seed in enumerate(START_SEEDS):
        rng = np.random.default_rng(seed)
        if k == 0:
            z, w = z_svd.copy(), w_svd.copy()
        elif k == 1:
            z, w = z_svd + rng.normal(0, 0.15, (n,K)), w_svd + rng.normal(0, 0.15, (m,K))
        else:
            z, w = rng.normal(0, 0.65, (n,K)), rng.normal(0, 0.65, (m,K))
        starts.append((a.copy(), b.copy(), z, w))
    return starts

def check_gradient(y, train, q, x):
    _, analytic = objective_gradient(x, y, train, q)
    rng = np.random.default_rng(3991)
    indices = rng.choice(x.size, size=min(32, x.size), replace=False)
    errors = []
    for k in indices:
        h = 1e-5
        plus, minus = x.copy(), x.copy()
        plus[k] += h
        minus[k] -= h
        fd = (objective_gradient(plus,y,train,q)[0]-objective_gradient(minus,y,train,q)[0])/(2*h)
        errors.append(abs(fd-analytic[k]) / max(1.0, abs(fd), abs(analytic[k])))
    maximum = float(max(errors))
    if maximum > 5e-5:
        raise RuntimeError(f"Gradient check failed for q={q}: {maximum}")
    return maximum

def scores(eta, y, mask):
    p = expit(eta)
    return {
        "n": int(mask.sum()),
        "negative_log_likelihood": float((np.logaddexp(0,eta)-y*eta)[mask].mean()),
        "brier": float(((p-y)**2)[mask].mean()),
        "accuracy": float(((p>=0.5)==y)[mask].mean()),
    }

def write_scores(results: dict) -> None:
    """Write the common six-row score table for full or smoke results."""
    rows = []
    for sd in SENSITIVITY_SDS:
        for q in [2, 1]:
            r = results["sensitivity"][str(sd)][str(q)]
            rows.append({"coordinate_sd": sd, "q": q, **r["held_out"],
                         "objective": r["selected_start"]["objective"],
                         "optimizer_success_flag": r["selected_start"]["success"],
                         "max_abs_selected_gradient": r["selected_start"]["max_abs_gradient"]})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "fit_comparison_results_scores.csv", index=False)


@threadpool_limits.wrap(limits=1)
def fit_models(design: dict, mode: str = "full"):
    """Fit all six model/scale pairs; full mode preserves the published procedure.

    Smoke mode uses one baseline start per exponent and two starts at each
    nonbaseline scale, with at most 30 iterations per fit.  It uses the full
    frozen dataset and is only a software check.  Call focused_refinement after
    full fitting, and render figures separately.
    """
    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'")
    current_design = prepare_data()
    if design != current_design:
        raise ValueError("Supplied design differs from the validated configured inputs.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    maxiter = MAXITER if mode == "full" else 30
    n_baseline_starts = len(START_SEEDS) if mode == "full" else 1
    data = np.load(SPLIT_FILE)
    y, train, test = data["y"], data["train"], data["test"]
    n, m = y.shape
    starts = initializations(y, train)
    all_runs, summaries, saved = [], {}, {}
    for q in [2, 1]:
        best = None
        for start_id, (a0,b0,z0,w0) in enumerate(starts[:n_baseline_starts]):
            distance = np.linalg.norm(z0[:,None,:]-w0[None,:,:], axis=2)**q
            # Offset the additive projection of the initial spatial penalty.
            a = a0 + distance.mean(axis=1) - distance.mean()/2
            b = b0 + distance.mean(axis=0) - distance.mean()/2
            x0 = np.r_[a,b,z0.ravel(),w0.ravel()]
            gradient_error = check_gradient(y,train,q,x0) if start_id == 0 else None
            started = time.perf_counter()
            result = minimize(objective_gradient, x0, args=(y,train,q), jac=True,
                              method="L-BFGS-B", options={"maxiter": maxiter, "ftol": FTOL,
                                                        "gtol": GTOL, "maxls": 60, "maxcor": 20})
            row = {
                "q": q, "coordinate_sd": COORDINATE_SD, "start_id": start_id, "seed": START_SEEDS[start_id],
                "objective": float(result.fun), "success": bool(result.success),
                "status": int(result.status), "message": str(result.message),
                "iterations": int(result.nit), "function_evaluations": int(result.nfev),
                "max_abs_gradient": float(np.max(np.abs(result.jac))),
                "gradient_l2_norm": float(np.linalg.norm(result.jac)),
                "seconds": time.perf_counter()-started,
                "gradient_check_max_relative_error": gradient_error,
            }
            all_runs.append(row)
            print(json.dumps({k: row[k] for k in ["q","start_id","objective","success","iterations","max_abs_gradient","seconds"]}), flush=True)
            # Held-out responses play NO role in selecting a start.
            if best is None or result.fun < best[0].fun:
                best = (result, row)
            # Save progress after every start, without requiring a full rerun.
            pd.DataFrame(all_runs).to_csv(RESULTS_DIR / "fit_comparison_results_starts.csv", index=False)
        result, selected = best
        eta = logits(result.x,n,m,q)
        a,b,z,w = unpack(result.x,n,m)
        distances = np.linalg.norm(z[:,None,:]-w[None,:,:],axis=2)
        summaries[str(q)] = {
            "selected_start": selected, "training": scores(eta,y,train),
            "held_out": scores(eta,y,test),
            "min_cross_mode_distance": float(distances.min()),
            "n_distances_below_1e_minus8": int((distances<1e-8).sum()),
            "coordinate_rms": float(np.sqrt(np.mean(np.r_[z.ravel(),w.ravel()]**2))),
            "interpretation": "best objective among finite starts; no guarantee of global optimum",
        }
        saved[f"q{q}_x"] = result.x
        saved[f"q{q}_eta"] = eta
        # Persist each selected model immediately.
        np.savez_compressed(FIT_FILE, **saved)
    sensitivity = {"1.0": summaries.copy()}
    # These scales and starts are specified before examining any comparative
    # score.  Every scale is reported; the holdout is never used to choose one.
    for sd in [s for s in SENSITIVITY_SDS if s != COORDINATE_SD]:
        sensitivity[str(sd)] = {}
        for q in [2,1]:
            best = None
            a0,b0,z0,w0 = starts[2]
            distance = np.linalg.norm(z0[:,None,:]-w0[None,:,:],axis=2)**q
            random_x = np.r_[a0+distance.mean(axis=1)-distance.mean()/2,
                              b0+distance.mean(axis=0)-distance.mean()/2,
                              z0.ravel(),w0.ravel()]
            for start_id,x0 in enumerate([saved[f"q{q}_x"],random_x]):
                started=time.perf_counter()
                result=minimize(objective_gradient,x0,args=(y,train,q,sd),jac=True,
                                method="L-BFGS-B",options={"maxiter":maxiter,"ftol":FTOL,
                                                           "gtol":GTOL,"maxls":60,"maxcor":20})
                row={"q":q,"coordinate_sd":sd,"start_id":start_id,
                     "seed":None if start_id==0 else START_SEEDS[2],
                     "initialization":"baseline MAP" if start_id==0 else "shared random start 2",
                     "objective":float(result.fun),"success":bool(result.success),
                     "status":int(result.status),"message":str(result.message),
                     "iterations":int(result.nit),"function_evaluations":int(result.nfev),
                     "max_abs_gradient":float(np.max(np.abs(result.jac))),
                     "gradient_l2_norm":float(np.linalg.norm(result.jac)),
                     "seconds":time.perf_counter()-started}
                all_runs.append(row)
                print(json.dumps({k:row[k] for k in ["q","coordinate_sd","start_id","objective","success","iterations","max_abs_gradient","seconds"]}),flush=True)
                if best is None or result.fun<best[0].fun: best=(result,row)
                pd.DataFrame(all_runs).to_csv(RESULTS_DIR/"fit_comparison_results_starts.csv",index=False)
            result,selected=best
            eta=logits(result.x,n,m,q)
            sensitivity[str(sd)][str(q)]={"selected_start":selected,
                                         "training":scores(eta,y,train),
                                         "held_out":scores(eta,y,test)}
            saved[f"q{q}_sd{sd}_x"]=result.x
            saved[f"q{q}_sd{sd}_eta"]=eta
            np.savez_compressed(FIT_FILE,**saved)
    heldout_difference = expit(saved["q1_eta"])[test] - expit(saved["q2_eta"])[test]
    results = {
        "design": design, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                    "pandas": pd.__version__, "platform": platform.platform(), "blas_threads": 1},
        "execution": {
            "mode": mode, "baseline_starts_per_exponent": n_baseline_starts,
            "sensitivity_starts_per_model": 2, "maxiter_per_initial_fit": maxiter,
            "focused_refinement_performed": False,
            "purpose": "full estimation procedure" if mode == "full" else "software check only; not inferential results",
        },
        "focused_refinements": [],
        "optimization_caution": "Finite-start local optimization and stopping flags do not establish global optimality or nonsmooth stationarity.",
        "models": summaries, "sensitivity": sensitivity, "all_starts": all_runs,
        "held_out_probability_difference_q1_minus_q2": {
            "mean_absolute": float(np.mean(np.abs(heldout_difference))),
            "rms": float(np.sqrt(np.mean(heldout_difference**2))),
            "max_absolute": float(np.max(np.abs(heldout_difference))),
        },
        "interpretation_limits": [
            "Matched MAP fits of likelihood forms, not replications of the original Bayesian analyses.",
            "Common parameter priors need not give common induced probability priors.",
            "One Congress and one random split do not establish performance dominance.",
            "Finite-start local optimization does not establish global MAP optimality.",
            "Plotted units are fixed by gamma and coordinate priors, not validated cardinal ideology.",
        ],
    }
    write_scores(results)
    save_json(RESULTS, results)
    return results

@threadpool_limits.wrap(limits=1)
def focused_refinement(results: dict):
    """Check the concrete near-coincidence issue without changing the model.

    Only q=1 scales with a selected gradient larger than 0.1 are reconsidered.
    Two continuations use the selected fit and an almost-identical coordinate
    perturbation, each with a larger iteration budget.  The exact same objective
    and training-only selection rule apply.  All before/after changes are saved.
    A small change is an optimization stability check, not a stationarity proof.
    """
    if results.get("execution", {}).get("mode") == "smoke":
        raise ValueError("Smoke mode does not run focused refinement; use full mode.")
    if results["design"] != prepare_data():
        raise ValueError("Results design differs from the validated configured inputs.")
    data=np.load(SPLIT_FILE)
    saved=dict(np.load(FIT_FILE))
    y,train,test=data["y"],data["train"],data["test"]
    n,m=y.shape
    refinements=[]
    for sd in SENSITIVITY_SDS:
        r=results["sensitivity"][str(sd)]["1"]
        if r["selected_start"]["max_abs_gradient"]<=0.1:
            continue
        key="q1_x" if sd==1 else f"q1_sd{sd}_x"
        eta_key="q1_eta" if sd==1 else f"q1_sd{sd}_eta"
        x_original=saved[key].copy()
        eta_original=logits(x_original,n,m,1)
        objective_original=objective_gradient(x_original,y,train,1,sd)[0]
        before={"objective":objective_original,"held_out":scores(eta_original,y,test),
                "selected_start":r["selected_start"]}
        best_x=x_original.copy(); best_value=objective_original; best_row=r["selected_start"]
        seed=40039+int(sd*1000)
        rng=np.random.default_rng(seed)
        perturbed=x_original.copy()
        perturbed[n+m:]+=rng.normal(0,1e-4,len(perturbed)-n-m)
        for continuation_id,x0 in enumerate([x_original,perturbed]):
            started=time.perf_counter()
            fit=minimize(objective_gradient,x0,args=(y,train,1,sd),jac=True,method="L-BFGS-B",
                         options={"maxiter":10000,"maxfun":100000,"maxls":80,
                                  "maxcor":30,"ftol":1e-12,"gtol":1e-5})
            row={"q":1,"coordinate_sd":sd,"start_id":f"refine_{continuation_id}",
                 "seed":None if continuation_id==0 else seed,
                 "initialization":"selected fit" if continuation_id==0 else "selected fit plus coordinate Normal(mean=0, sd=1e-4; variance=1e-8) perturbation",
                 "objective":float(fit.fun),"success":bool(fit.success),"status":int(fit.status),
                 "message":str(fit.message),"iterations":int(fit.nit),"function_evaluations":int(fit.nfev),
                 "max_abs_gradient":float(np.max(np.abs(fit.jac))),
                 "gradient_l2_norm":float(np.linalg.norm(fit.jac)),"seconds":time.perf_counter()-started,
                 "maxiter":10000,"ftol":1e-12,"maxcor":30}
            results["all_starts"].append(row)
            print(json.dumps({k:row[k] for k in ["q","coordinate_sd","start_id","objective","success","iterations","max_abs_gradient","seconds"]}),flush=True)
            if fit.fun<best_value: best_x,best_value,best_row=fit.x,float(fit.fun),row
        eta=logits(best_x,n,m,1)
        saved[key],saved[eta_key]=best_x,eta
        old_probability=expit(eta_original)[test]
        new_probability=expit(eta)[test]
        updated={"selected_start":best_row,"training":scores(eta,y,train),
                 "held_out":scores(eta,y,test)}
        if sd==1:
            _,_,z,w=unpack(best_x,n,m)
            distances=np.linalg.norm(z[:,None]-w[None,:],axis=2)
            updated.update({"min_cross_mode_distance":float(distances.min()),
                            "n_distances_below_1e_minus8":int((distances<1e-8).sum()),
                            "coordinate_rms":float(np.sqrt(np.mean(np.r_[z.ravel(),w.ravel()]**2))),
                            "interpretation":"best observed training objective; no global or nonsmooth stationarity guarantee"})
            results["models"]["1"]=updated
        results["sensitivity"][str(sd)]["1"]=updated
        refinements.append({"coordinate_sd":sd,"before":before,
                            "after":{"objective":best_value,"held_out":scores(eta,y,test),"selected_start":best_row},
                            "objective_change":best_value-objective_original,
                            "objective_change_per_training_vote":(best_value-objective_original)/train.sum(),
                            "held_out_probability_mean_abs_change":float(np.mean(abs(new_probability-old_probability))),
                            "held_out_probability_max_abs_change":float(np.max(abs(new_probability-old_probability))),
                            "held_out_log_loss_change":scores(eta,y,test)["negative_log_likelihood"]-before["held_out"]["negative_log_likelihood"],
                            "interpretation":"stability check only; does not establish exact/global MAP or a nonsmooth stationarity certificate"})
    results["focused_refinements"]=refinements
    if "execution" in results:
        results["execution"]["focused_refinement_performed"] = True
    results["completed_utc"]=datetime.now(timezone.utc).isoformat()
    diff=expit(saved["q1_eta"])[test]-expit(saved["q2_eta"])[test]
    results["held_out_probability_difference_q1_minus_q2"]={
        "mean_absolute":float(np.mean(np.abs(diff))),
        "rms":float(np.sqrt(np.mean(diff**2))),
        "max_absolute":float(np.max(np.abs(diff))),
    }
    results["optimization_caution"]="Exact q=1 norm creates near-coincident configurations. L-BFGS stopping flags and selected gradient norms are reported verbatim; small objective/prediction changes on continuation are numerical stability evidence, not a proof of stationarity or global MAP optimality."
    np.savez_compressed(FIT_FILE,**saved)
    pd.DataFrame(results["all_starts"]).to_csv(RESULTS_DIR/"fit_comparison_results_starts.csv",index=False)
    write_scores(results)
    save_json(RESULTS,results)
    return results

def write_summary(results: dict):
    """Keep the readable summary synchronized with the selected fitted results."""
    design = results["design"]
    lines = [
        "MATCHED DISTANCE-MODEL FIT COMPARISON",
        "",
        "Generated from fit_comparison_results.json; no manually copied scores.",
        "Scope: approximate MAP fits, not original-paper Bayesian replications.",
        f"Completed (UTC): {results['completed_utc']}",
        f"Runtime: {json.dumps(results['runtime'], ensure_ascii=False)}",
        f"Execution: {json.dumps(results.get('execution', {'mode': 'archived full run'}))}",
        "",
        f"Data: {design['n_members']} senators, {design['n_rollcalls']} roll calls, "
        f"{design['n_train']} training votes, {design['n_test']} held-out votes.",
        f"Preprocessing and source hashes: {DESIGN_FILE.relative_to(ROOT)}.",
        "Models: q in {1,2}, K=2, gamma=1; intercept SD=3; coordinate SD in {0.5,1,2}.",
        "Selection uses the penalized training objective only; all scales are reported.",
        "Full-run refinement perturbation: Normal(mean=0, SD=1e-4), variance=1e-8.",
        "",
        "SCORES",
        "s     q   Training log loss   Held-out log loss   Held-out Brier",
    ]
    for sd in SENSITIVITY_SDS:
        for q in [2, 1]:
            result = results["sensitivity"][str(sd)][str(q)]
            lines.append(f"{sd:<5} {q}   {result['training']['negative_log_likelihood']:.9f}"
                         f"         {result['held_out']['negative_log_likelihood']:.9f}"
                         f"         {result['held_out']['brier']:.9f}")
    lines.extend(["", "NUMERICAL DIAGNOSTICS"])
    for q in [2, 1]:
        initial = [r for r in results["all_starts"]
                   if r["q"] == q and r["coordinate_sd"] == 1 and isinstance(r["start_id"], int)]
        gradient = next(r["gradient_check_max_relative_error"] for r in initial if r["start_id"] == 0)
        spread = max(r["objective"] for r in initial) - min(r["objective"] for r in initial)
        lines.append(f"q={q}: initial gradient-check relative error={gradient:.9g}; "
                     f"baseline-start objective spread={spread:.9g}.")
    for sd in SENSITIVITY_SDS:
        for q in [2, 1]:
            row = results["sensitivity"][str(sd)][str(q)]["selected_start"]
            lines.append(f"s={sd}, q={q}: objective={row['objective']:.12g}; "
                         f"gradient infinity norm={row['max_abs_gradient']:.9g}; "
                         f"start={row['start_id']}; stopping flag={row['message']}.")
    for item in results.get("focused_refinements", []):
        lines.append(f"Refinement s={item['coordinate_sd']}: "
                     f"objective change={item['objective_change']:.12g}; "
                     f"mean/max held-out probability change="
                     f"{item['held_out_probability_mean_abs_change']:.9g}/"
                     f"{item['held_out_probability_max_abs_change']:.9g}; "
                     f"held-out log-loss change={item['held_out_log_loss_change']:.9g}.")
    lines.extend(["", "INTERPRETATION", results["optimization_caution"],
                  *results["interpretation_limits"],
                  "", "Figure rendering is a separate step.",
                  f"Numerical outputs: {RESULTS_DIR.relative_to(ROOT)}."])
    if results.get("execution", {}).get("mode") == "smoke":
        lines.extend(["", "SMOKE RUN: software check only; do not use these estimates for inference."])
    (RESULTS_DIR / "fit_comparison_results.txt").write_text("\n".join(lines) + "\n")

