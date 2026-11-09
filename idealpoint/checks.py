"""Independent, read-only checks for the exact example and Senate analysis.

This module does not import the optimizer or inspect manuscript sources. Raw
CSV records reconstruct the fixed evaluation split, and saved parameter vectors
reconstruct every logit, score, and posterior objective. Numerical consistency
does not certify convergence or a global optimum. All returned objects can be
serialized with ``json.dumps(report, allow_nan=False)``.
"""
from __future__ import annotations

import csv
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np


REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"
SCALES = (0.5, 1.0, 2.0)
RAW_SHA256 = {
    "S114_votes.csv": "cf8fb57d99ecc32dc0a35ddb62fc578926479a80092e72ca70caad8e2fe435b1",
    "S114_members.csv": "aa575f1930125b3ca0e5e7bd068d09005a6895722631fcae1d69a562da2a100e",
    "S114_rollcalls.csv": "ef06eccacacf21aede6f91bdb5eb81d4426645fbb2aecb7e0bb760b17dd1c1c2",
}
SPLIT_SHA256 = "bd20e470af1a160e1120c84d827fabeafcc90812d601adabc88e624a2f275889"
EXPECTED_COUNTS = {
    "original_members": 100, "original_rollcalls": 502,
    "n_members": 96, "n_rollcalls": 373, "n_train": 28164,
    "n_test": 7018, "n_observed": 35182, "n_missing": 626,
}
SCORE_NAMES = ("training_log_loss", "held_out_log_loss", "held_out_brier")
# Published Table S1 precision. These values are never used to compute scores.
PUBLISHED_SCORES = {
    (0.5, 2): ("0.180557", "0.208056", "0.063118"),
    (0.5, 1): ("0.297424", "0.310438", "0.086802"),
    (1.0, 2): ("0.166961", "0.203353", "0.062489"),
    (1.0, 1): ("0.232936", "0.250925", "0.072520"),
    (2.0, 2): ("0.163273", "0.204449", "0.062606"),
    (2.0, 1): ("0.195810", "0.223432", "0.066511"),
}


def _multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def _rank(matrix):
    a = [list(map(F, row)) for row in matrix]
    rank = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(rank, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][col]
        a[rank] = [x / scale for x in a[rank]]
        for i in range(len(a)):
            if i != rank:
                scale = a[i][col]
                a[i] = [x - scale * y for x, y in zip(a[i], a[rank])]
        rank += 1
    return rank


def _contrast(matrix, i, k, j, l):
    return matrix[i][j] + matrix[k][l] - matrix[i][l] - matrix[k][j]


def exact_checks():
    """Return exact rational checks; finite checks do not replace the proofs.

    Checks the centered logits, ranks, eigenpairs, positive contrasts, and
    signed interval-overlap identity on all 625 ordered endpoint choices from
    a five-point grid. Also checks concrete invariance and rejection examples.
    No files are read or written.
    """
    x = [F(-1), F(0), F(1)]
    h = [[F(i == j) - F(1, 3) for j in range(3)] for i in range(3)]
    logits = [[[-abs(a - b) ** q for b in x] for a in x] for q in (1, 2)]
    centered = [_multiply(_multiply(h, eta), h) for eta in logits]
    expected = [
        [[F(10, 9), F(-2, 9), F(-8, 9)],
         [F(-2, 9), F(4, 9), F(-2, 9)],
         [F(-8, 9), F(-2, 9), F(10, 9)]],
        [[F(2), F(0), F(-2)], [F(0), F(0), F(0)], [F(-2), F(0), F(2)]],
    ]
    ranks = [_rank(a) for a in centered]
    checks = {"centered_matrices": centered == expected, "centered_ranks": ranks == [2, 1]}
    vectors = [[[F(1)], [F(1)], [F(1)]],
               [[F(1)], [F(0)], [F(-1)]],
               [[F(1)], [F(-2)], [F(1)]]]
    eigenvalues = [(F(0), F(2), F(2, 3)), (F(0), F(4), F(0))]
    checks["eigenpairs"] = all(
        _multiply(a, v) == [[ev * row[0]] for row in v]
        for a, values in zip(centered, eigenvalues) for v, ev in zip(vectors, values))
    contrasts = {(i, k, j, l): _contrast(logits[1], i, k, j, l)
                 for i in range(3) for k in range(i + 1, 3)
                 for j in range(3) for l in range(j + 1, 3)}
    checks["quadratic_ordered_contrasts_positive"] = all(v > 0 for v in contrasts.values())
    checks["contradictory_middle_interval_contrasts"] = (
        contrasts[0, 1, 1, 2] == contrasts[1, 2, 0, 1] == 2)
    overlap_ok = True
    for u, up, v, vp in product(map(F, range(-2, 3)), repeat=4):
        lhs = -abs(u - v) - abs(up - vp) + abs(u - vp) + abs(up - v)
        sign = ((up > u) - (up < u)) * ((vp > v) - (vp < v))
        overlap = max(F(0), min(max(u, up), max(v, vp)) - max(min(u, up), min(v, vp)))
        overlap_ok &= lhs == 2 * sign * overlap
    checks["interval_overlap_625_grid_cases"] = overlap_ok
    # A = diag(2, 1), with its inverse transpose acting on the items.
    actors = [(F(1), F(2)), (F(-1), F(3))]
    items = [(F(2), F(-1)), (F(0), F(1))]
    actor_effects, item_effects = [F(2), F(-1)], [F(3), F(0)]
    moved_actors = [(2 * z[0], z[1]) for z in actors]
    moved_items = [(w[0] / 2, w[1]) for w in items]
    norm2 = lambda z: sum(t * t for t in z)
    squared_distance = lambda z, w: sum((a - b) ** 2 for a, b in zip(z, w))
    original = [[actor_effects[i] + item_effects[j] - squared_distance(z, w)
                 for j, w in enumerate(items)] for i, z in enumerate(actors)]
    adjusted_a = [a + norm2(zp) - norm2(z) for a, z, zp in zip(actor_effects, actors, moved_actors)]
    adjusted_b = [b + norm2(wp) - norm2(w) for b, w, wp in zip(item_effects, items, moved_items)]
    moved = [[adjusted_a[i] + adjusted_b[j] - squared_distance(z, w)
              for j, w in enumerate(moved_items)] for i, z in enumerate(moved_actors)]
    checks["quadratic_inverse_linear_invariance_example"] = original == moved
    unsquared_before = [[-abs(u - v) for v in (F(1), F(2))] for u in (F(1), F(2))]
    unsquared_after = [[-abs(u - v) for v in (F(1, 2), F(1))] for u in (F(2), F(4))]
    checks["unsquared_inverse_linear_counterexample"] = (
        _contrast(unsquared_before, 0, 1, 0, 1) == 2
        and _contrast(unsquared_after, 0, 1, 0, 1) == 0)
    checks["nonadditive_rejection_example"] = all(
        _contrast([[F(abs(i - j) ** q) for j in range(2)] for i in range(2)], 0, 1, 0, 1) == -2
        for q in (1, 2))
    return {
        "passed": all(checks.values()), "checks": checks,
        "coordinates": list(map(str, x)),
        "centered_logit_matrices": [[[str(v) for v in row] for row in a] for a in centered],
        "ranks": ranks, "eigenvalues": [["0", "2/3", "2"], ["0", "0", "4"]],
        "quadratic_ordered_rectangles": {",".join(map(str, k)): str(v) for k, v in contrasts.items()},
        "verification": "Exact rational arithmetic. Finite arithmetic checks support, but do not replace, the general proofs.",
    }


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _rebuild_data(data_dir):
    """Use the stated protocol and raw CSV records, independently of fit code."""
    members = {}
    for row in _rows(data_dir / "S114_members.csv"):
        if row["chamber"] == "Senate":
            members.setdefault(int(row["icpsr"]), row)
    rolls = {int(row["rollnumber"]): row for row in _rows(data_dir / "S114_rollcalls.csv")
             if row["chamber"] == "Senate"}
    ids, roll_ids = np.array(sorted(members)), np.array(sorted(rolls))
    original_counts = {"original_members": len(ids), "original_rollcalls": len(roll_ids)}
    ii = {int(v): i for i, v in enumerate(ids)}
    jj = {int(v): j for j, v in enumerate(roll_ids)}
    y = np.full((len(ids), len(roll_ids)), np.nan)
    seen = set()
    for row in _rows(data_dir / "S114_votes.csv"):
        if row["chamber"] != "Senate" or int(row["cast_code"]) not in (1, 6):
            continue
        member, roll = int(float(row["icpsr"])), int(row["rollnumber"])
        # Presidential records may say Senate in the vote file. The member
        # roster, not the vote's chamber alone, determines Senate membership.
        if member not in ii:
            continue
        if roll not in jj or (member, roll) in seen:
            raise ValueError(f"Unknown roll call or duplicate raw vote: {(member, roll)}")
        seen.add((member, roll))
        y[ii[member], jj[roll]] = int(row["cast_code"]) == 1
    keep = np.isfinite(y).sum(axis=1) >= np.ceil(0.8 * len(roll_ids))
    y, ids = y[keep], ids[keep]
    observed = np.isfinite(y)
    draws = np.random.default_rng(20261023).random(y.shape)
    train, test = observed & (draws >= 0.2), observed & (draws < 0.2)
    count = train.sum(axis=0)
    yes = np.where(train, y, 0).sum(axis=0)
    share = np.divide(yes, count, out=np.zeros_like(yes), where=count > 0)
    keep = ((observed.sum(axis=0) >= np.ceil(0.8 * len(ids))) & (count >= 50)
            & (share >= 0.1) & (share <= 0.9))
    y, train, test, roll_ids = y[:, keep], train[:, keep], test[:, keep], roll_ids[keep]
    arrays = {"y": np.nan_to_num(y, nan=0.0), "train": train, "test": test,
              "member_ids": ids, "roll_ids": roll_ids}
    counts = {**original_counts, "n_members": len(ids), "n_rollcalls": len(roll_ids),
              "n_train": int(train.sum()), "n_test": int(test.sum()),
              "n_observed": int((train | test).sum()), "n_missing": int((~(train | test)).sum())}
    parties = np.array([int(members[int(i)]["party_code"]) for i in ids])
    metadata = {
        "date_min": min(rolls[int(i)]["date"] for i in roll_ids),
        "date_max": max(rolls[int(i)]["date"] for i in roll_ids),
        "party_counts": {str(int(p)): int((parties == p).sum()) for p in np.unique(parties)},
    }
    return arrays, counts, parties, metadata, members, rolls


def _processed_checks(data_dir, arrays, members, rolls):
    ids, roll_ids = arrays["member_ids"], arrays["roll_ids"]
    member_rows = _rows(data_dir / "fit_comparison_members.csv")
    roll_rows = _rows(data_dir / "fit_comparison_rollcalls.csv")
    checks = {
        "processed_member_order": [int(r["icpsr"]) for r in member_rows] == ids.tolist(),
        "processed_roll_call_order": [int(r["rollnumber"]) for r in roll_rows] == roll_ids.tolist(),
        "processed_member_descriptors": all(
            int(r["icpsr"]) in members and all(r[k] == members[int(r["icpsr"])][k]
                for k in ("bioname", "state_abbrev", "party_code")) for r in member_rows),
        "processed_roll_call_descriptors": all(
            int(r["rollnumber"]) in rolls and all(r[k] == rolls[int(r["rollnumber"])][k]
                for k in ("date", "bill_number", "vote_desc", "vote_question", "vote_result")) for r in roll_rows),
    }
    ii, jj = {int(v): i for i, v in enumerate(ids)}, {int(v): j for j, v in enumerate(roll_ids)}
    long_y = np.zeros_like(arrays["y"])
    long_train, long_test = np.zeros_like(arrays["train"]), np.zeros_like(arrays["test"])
    seen, valid = set(), True
    for row in _rows(data_dir / "fit_comparison_votes.csv"):
        member, roll = int(row["icpsr"]), int(row["rollnumber"])
        pair = (member, roll)
        if (member not in ii or roll not in jj or pair in seen
                or row["role"] not in ("train", "test") or row["yea"] not in ("0", "1")):
            valid = False
            continue
        seen.add(pair)
        i, j = ii[member], jj[roll]
        long_y[i, j] = int(row["yea"])
        (long_train if row["role"] == "train" else long_test)[i, j] = True
    checks["processed_votes_reconstruct_raw_split"] = bool(
        valid and np.array_equal(long_y, arrays["y"])
        and np.array_equal(long_train, arrays["train"]) and np.array_equal(long_test, arrays["test"]))
    return checks


def _reconstruct(x, arrays, q, sd):
    y, train, test = arrays["y"], arrays["train"], arrays["test"]
    n, m = y.shape
    if x.shape != ((n + m) * 3,) or not np.isfinite(x).all():
        raise ValueError(f"Invalid saved parameter vector for s={sd}, q={q}")
    a, b = x[:n], x[n:n + m]
    z = x[n + m:n + m + 2 * n].reshape(n, 2)
    w = x[n + m + 2 * n:].reshape(m, 2)
    delta = z[:, None] - w[None, :]
    distance2 = np.square(delta).sum(axis=2)
    distance = np.sqrt(distance2)
    eta = a[:, None] + b[None, :] - (distance if q == 1 else distance2)
    probability = np.exp(-np.logaddexp(0, -eta))
    loss = np.logaddexp(0, eta) - y * eta
    intercept_penalty = float((np.square(a).sum() + np.square(b).sum()) / 18)
    coordinate_penalty = float((np.square(z).sum() + np.square(w).sum()) / (2 * sd * sd))
    values = {
        "training_log_loss": float(loss[train].mean()),
        "training_brier": float(np.square(probability - y)[train].mean()),
        "training_accuracy": float(((probability >= 0.5) == y)[train].mean()),
        "held_out_log_loss": float(loss[test].mean()),
        "held_out_brier": float(np.square(probability - y)[test].mean()),
        "held_out_accuracy": float(((probability >= 0.5) == y)[test].mean()),
        "objective": float(loss[train].sum() + intercept_penalty + coordinate_penalty),
    }
    error = (probability - y) * train
    direction = (np.divide(delta, distance[:, :, None], out=np.zeros_like(delta),
                           where=distance[:, :, None] > 0) if q == 1 else 2 * delta)
    pair = error[:, :, None] * direction
    gradient = np.r_[error.sum(axis=1) + a / 9, error.sum(axis=0) + b / 9,
                     (-pair.sum(axis=1) + z / sd ** 2).ravel(),
                     (pair.sum(axis=0) + w / sd ** 2).ravel()]
    diagnostics = {
        "training_negative_log_likelihood": float(loss[train].sum()),
        "intercept_prior_penalty": intercept_penalty, "coordinate_prior_penalty": coordinate_penalty,
        "gradient_infinity_norm": float(np.abs(gradient).max()),
        "gradient_l2_norm": float(np.linalg.norm(gradient)),
        "minimum_actor_item_distance": float(distance.min()),
        "near_actor_item_pairs": int((distance < 1e-8).sum()),
        "exactly_coincident_training_pairs": int(((distance == 0) & train).sum()),
    }
    if not np.isfinite(eta).all() or not all(np.isfinite(v) for v in (*values.values(), *diagnostics.values())):
        raise ValueError(f"Nonfinite reconstructed values for s={sd}, q={q}")
    return eta, values, diagnostics, z, w


def _geometry(z, w, parties, arrays):
    distance = np.linalg.norm(z[:, None] - z[None, :], axis=2)
    major = np.isin(parties, (100, 200))
    silhouettes = []
    for i in np.flatnonzero(major):
        own = parties == parties[i]
        own[i] = False
        a = float(distance[i, own].mean())
        b = float(distance[i, major & (parties != parties[i])].mean())
        silhouettes.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    near = np.triu(distance < 1e-8, k=1)
    democratic = parties == 100
    cross = np.linalg.norm(z[:, None] - w[None, :], axis=2) < 1e-8
    return {
        "mean_two_party_silhouette": float(np.mean(silhouettes)),
        "near_unordered_senator_pairs": int(near.sum()),
        "near_democratic_senator_pairs": int(near[np.ix_(democratic, democratic)].sum()),
        "near_senator_roll_call_pairs": int(cross.sum()),
        "near_training_senator_roll_call_pairs": int((cross & arrays["train"]).sum()),
        "near_held_out_senator_roll_call_pairs": int((cross & arrays["test"]).sum()),
        "near_missing_senator_roll_call_pairs": int((cross & ~(arrays["train"] | arrays["test"])).sum()),
    }


def _figure_checks(figure, parameters, arrays):
    n, m = arrays["y"].shape
    checks = {"figure_coordinate_sd": float(figure["coordinate_sd"]) == 1.0}
    for field in ("member_ids", "roll_ids"):
        checks[f"figure_{field}"] = bool(np.array_equal(figure[field], arrays[field]))
    errors = {}
    for q in (1, 2):
        original = parameters[f"q{q}_x"][n + m:].reshape(n + m, 2)
        displayed, rotation = figure[f"q{q}_displayed"], figure[f"q{q}_rotation"]
        if displayed.shape != original.shape or rotation.shape != (2, 2):
            raise ValueError(f"Invalid Figure 2 array shapes for q={q}")
        checks[f"figure_q{q}_source_coordinates"] = bool(np.array_equal(original, figure[f"q{q}_original"]))
        orthogonality = float(np.abs(rotation.T @ rotation - np.eye(2)).max())
        coordinate_error = float(np.abs(displayed - original @ rotation).max())
        transformed = parameters[f"q{q}_x"].copy()
        transformed[n + m:] = displayed.ravel()
        old_eta, old_scores, _, _, _ = _reconstruct(parameters[f"q{q}_x"], arrays, q, 1.0)
        new_eta, new_scores, _, _, _ = _reconstruct(transformed, arrays, q, 1.0)
        errors[str(q)] = {
            "orthogonality_error": orthogonality, "coordinate_reconstruction_error": coordinate_error,
            "maximum_logit_change": float(np.abs(old_eta - new_eta).max()),
            "posterior_objective_change": new_scores["objective"] - old_scores["objective"],
        }
        checks[f"figure_q{q}_orthogonal_alignment"] = (
            orthogonality <= 1e-12 and coordinate_error <= 1e-12
            and errors[str(q)]["maximum_logit_change"] <= 1e-10
            and abs(errors[str(q)]["posterior_objective_change"]) <= 1e-8)
    return checks, errors


def _model_key(sd, q):
    return f"s={sd},q={q}"


def _prefix(sd, q):
    return f"q{q}" if sd == 1.0 else f"q{q}_sd{sd}"


def _published_matches(calculated, geometry):
    score_matches = {
        _model_key(sd, q): all(f"{calculated[_model_key(sd, q)][metric]:.6f}" == reported
                             for metric, reported in zip(SCORE_NAMES, values))
        for (sd, q), values in PUBLISHED_SCORES.items()}
    geometry_matches = {}
    for q, silhouette, pairs, cross in ((1, "0.858", 214, 589), (2, "0.661", 0, 0)):
        row = geometry["fits"][str(q)]
        geometry_matches[str(q)] = (
            f"{row['mean_two_party_silhouette']:.3f}" == silhouette
            and row["near_democratic_senator_pairs"] == pairs
            and row["near_unordered_senator_pairs"] == pairs
            and row["near_senator_roll_call_pairs"] == cross)
    return {"scores": score_matches, "baseline_geometry": geometry_matches,
            "all_match": all(score_matches.values()) and all(geometry_matches.values())}


def verify_results(results_dir, data_dir, compare_reference=True):
    """Independently validate the six saved Senate fits without modifying files.

    Parameters are directories containing the original basenames. Raw data,
    the frozen split, processed CSVs, fit JSON/NPZ, and score/start CSVs are
    required. ``figure2_positions.npz`` and ``verification.json`` are checked
    when present. Missing or structurally malformed input raises an exception;
    ordinary numerical/integrity mismatches appear as failed Boolean checks.

    ``passed`` concerns internal consistency and frozen-input integrity. With
    ``compare_reference=True``, the bundled reference archive is also audited.
    New fit differences are reported separately and do not themselves fail the
    audit. A successful audit does not establish optimizer convergence.
    """
    results_dir, data_dir = Path(results_dir).resolve(), Path(data_dir).resolve()
    design = _json(data_dir / "fit_comparison_design.json")
    result = _json(results_dir / "fit_comparison_results.json")
    parameters = _npz(results_dir / "fit_comparison_results.npz")
    saved = _npz(data_dir / "fit_comparison_split.npz")
    arrays, counts, parties, metadata, members, rolls = _rebuild_data(data_dir)
    for field, expected in arrays.items():
        if field not in saved or saved[field].shape != expected.shape:
            raise ValueError(f"Invalid frozen split shape or missing array: {field}")
    checks = {f"reconstructed_{field}": bool(np.array_equal(value, saved[field]))
              for field, value in arrays.items()}
    checks.update({
        "published_counts": counts == EXPECTED_COUNTS,
        "design_counts": all(design.get(k) == v for k, v in counts.items()),
        "design_dates_and_parties": all(design.get(k) == v for k, v in metadata.items()),
        "results_design": result["design"] == design,
        "mask_boolean_dtype": saved["train"].dtype == bool and saved["test"].dtype == bool,
        "mask_partition_disjoint": not bool(np.any(saved["train"] & saved["test"])),
        "mask_partition_complete": bool(np.array_equal(saved["train"] | saved["test"], arrays["train"] | arrays["test"])),
        "training_covers_every_actor_and_item": bool(np.all(saved["train"].sum(axis=0) > 0)
                                                     and np.all(saved["train"].sum(axis=1) > 0)),
        "binary_finite_outcomes": bool(np.isfinite(saved["y"]).all() and np.isin(saved["y"], (0, 1)).all()),
    })
    model = design["model"]
    checks["stated_model_and_split"] = (
        model["K"] == 2 and model["gamma_fixed"] == 1 and sorted(model["q"]) == [1, 2]
        and model["intercept_prior"]["mean"] == 0 and model["intercept_prior"]["sd"] == 3
        and model["coordinate_prior"]["mean"] == 0 and model["coordinate_prior"]["sd"] == 1
        and design["split"]["seed"] == 20261023 and design["split"]["test_probability"] == 0.2
        and design["sensitivity_plan"]["coordinate_sd"] == list(SCALES))
    hashes = {"data": {}, "results": {}}
    for name in (*RAW_SHA256, "fit_comparison_design.json", "fit_comparison_split.npz",
                 "fit_comparison_members.csv", "fit_comparison_rollcalls.csv", "fit_comparison_votes.csv"):
        hashes["data"][name] = _sha256(data_dir / name)
    for name, expected in RAW_SHA256.items():
        checks[f"sha256_{name}"] = hashes["data"][name] == expected == design["sources"][name]["sha256"]
    # An explicit rebuild may change ZIP serialization without changing any
    # array. Require its active checksum and independent array reconstruction;
    # report original byte identity separately from scientific equivalence.
    checks["sha256_frozen_split"] = (
        hashes["data"]["fit_comparison_split.npz"] == design["split_sha256"])
    checks.update(_processed_checks(data_dir, arrays, members, rolls))
    score_rows = _rows(results_dir / "fit_comparison_results_scores.csv")
    rows_by_model = {(float(r["coordinate_sd"]), int(r["q"])): r for r in score_rows}
    expected_models = {(sd, q) for sd in SCALES for q in (1, 2)}
    actual_models = {(float(sd), int(q)) for sd, fits in result["sensitivity"].items() for q in fits}
    checks["six_score_rows"] = len(score_rows) == 6 and set(rows_by_model) == expected_models
    checks["six_model_summaries"] = actual_models == expected_models
    if not expected_models <= set(rows_by_model) or not expected_models <= actual_models:
        raise ValueError("All six prespecified (coordinate_sd, q) fits are required")
    all_starts = result["all_starts"]
    start_rows = _rows(results_dir / "fit_comparison_results_starts.csv")
    checks["start_table_row_count"] = len(start_rows) == len(all_starts)
    checks["start_table_matches_json"] = len(start_rows) == len(all_starts) and all(
        int(row["q"]) == run["q"] and float(row["coordinate_sd"]) == run["coordinate_sd"]
        and str(row["start_id"]) == str(run["start_id"])
        and abs(float(row["objective"]) - run["objective"]) <= 1e-8
        and abs(float(row["max_abs_gradient"]) - run["max_abs_gradient"]) <= 1e-7
        for row, run in zip(start_rows, all_starts))
    calculated, diagnostics, errors, baseline_geometry = {}, {}, {}, {}
    for sd in SCALES:
        for q in (2, 1):
            key, prefix = _model_key(sd, q), _prefix(sd, q)
            eta, values, diagnostic, z, w = _reconstruct(parameters[prefix + "_x"], arrays, q, sd)
            saved_eta = parameters[prefix + "_eta"]
            if saved_eta.shape != eta.shape or not np.isfinite(saved_eta).all():
                raise ValueError(f"Invalid saved logits: {prefix}")
            fit = result["sensitivity"][str(sd)][str(q)]
            selected = fit["selected_start"]
            expected = {"objective": selected["objective"]}
            for group, stem in (("training", "training"), ("held_out", "held_out")):
                expected.update({f"{stem}_log_loss": fit[group]["negative_log_likelihood"],
                                 f"{stem}_brier": fit[group]["brier"],
                                 f"{stem}_accuracy": fit[group]["accuracy"]})
            model_errors = {k: abs(values[k] - float(v)) for k, v in expected.items()}
            model_errors["saved_logits"] = float(np.abs(eta - saved_eta).max())
            row = rows_by_model[sd, q]
            for name, field in (("held_out_log_loss", "negative_log_likelihood"),
                                ("held_out_brier", "brier"), ("held_out_accuracy", "accuracy"),
                                ("objective", "objective")):
                model_errors["csv_" + name] = abs(values[name] - float(row[field]))
            model_errors["gradient_infinity_norm"] = abs(diagnostic["gradient_infinity_norm"] - selected["max_abs_gradient"])
            model_errors["gradient_l2_norm"] = abs(diagnostic["gradient_l2_norm"] - selected["gradient_l2_norm"])
            model_errors["csv_gradient_infinity_norm"] = abs(diagnostic["gradient_infinity_norm"] - float(row["max_abs_selected_gradient"]))
            checks[key + "_reconstruction"] = all(
                error <= (1e-7 if "gradient" in name else 1e-8 if "objective" in name else 1e-10)
                for name, error in model_errors.items())
            checks[key + "_score_counts"] = (fit["training"]["n"] == counts["n_train"]
                and fit["held_out"]["n"] == counts["n_test"] and int(row["n"]) == counts["n_test"])
            candidates = [r for r in all_starts if r["q"] == q and r["coordinate_sd"] == sd]
            checks[key + "_lowest_recorded_objective"] = bool(candidates) and abs(
                selected["objective"] - min(r["objective"] for r in candidates)) <= 1e-8
            checks[key + "_selected_start_record"] = any(r == selected for r in candidates)
            diagnostic.update({"selected_start": selected["start_id"],
                               "optimizer_success_flag": bool(selected["success"]),
                               "optimizer_stopping_message": selected["message"],
                               "certified_global_map": False,
                               "nonsmooth_stationarity_certified": False if q == 1 else None})
            calculated[key], diagnostics[key], errors[key] = values, diagnostic, model_errors
            if sd == 1:
                baseline_geometry[str(q)] = _geometry(z, w, parties, arrays)
    geometry = {
        "coordinate_sd": 1.0, "strict_distance_threshold": 1e-8,
        "silhouette_party_counts": {"Democrat": int((parties == 100).sum()),
                                    "Republican": int((parties == 200).sum())},
        "silhouette_excluded_senators": int((~np.isin(parties, (100, 200))).sum()),
        "silhouette_interpretation": "Descriptive separation of known parties after fitting; not recovery of unknown factions.",
        "fits": baseline_geometry,
    }
    exact = exact_checks()
    checks["exact_arithmetic"] = exact["passed"]
    figure_report = {"checked": False, "reason": "No saved display coordinates in this result directory."}
    figure_path = results_dir / "figure2_positions.npz"
    if figure_path.exists():
        figure_checks, figure_errors = _figure_checks(_npz(figure_path), parameters, arrays)
        checks.update(figure_checks)
        figure_report = {"checked": True, "absolute_errors": figure_errors}
    exact_path = results_dir / "verification.json"
    if exact_path.exists():
        recorded_exact = _json(exact_path)
        checks["saved_exact_report"] = all(recorded_exact.get(k) == exact[k]
            for k in ("centered_logit_matrices", "ranks", "eigenvalues"))
    for name in ("fit_comparison_results.json", "fit_comparison_results.npz",
                 "fit_comparison_results_scores.csv", "fit_comparison_results_starts.csv",
                 "figure2_positions.npz", "verification.json"):
        if (results_dir / name).exists():
            hashes["results"][name] = _sha256(results_dir / name)
    published = _published_matches(calculated, geometry)
    comparison = {"performed": False}
    if compare_reference:
        archived = verify_results(REFERENCE_DIR, data_dir, compare_reference=False)
        checks["reference_archive_valid"] = archived["passed"] and archived["published_precision"]["all_match"]
        reference_parameters = _npz(REFERENCE_DIR / "fit_comparison_results.npz")
        differences = {}
        for sd in SCALES:
            for q in (2, 1):
                key, prefix = _model_key(sd, q), _prefix(sd, q)
                eta, _, _, _, _ = _reconstruct(parameters[prefix + "_x"], arrays, q, sd)
                ref_eta, _, _, _, _ = _reconstruct(reference_parameters[prefix + "_x"], arrays, q, sd)
                dp = (np.exp(-np.logaddexp(0, -eta)) - np.exp(-np.logaddexp(0, -ref_eta)))[arrays["test"]]
                differences[key] = {
                    "score_differences": {k: calculated[key][k] - archived["calculated"][key][k] for k in SCORE_NAMES},
                    "same_scores_at_3_decimals": all(f"{calculated[key][k]:.3f}" == f"{archived['calculated'][key][k]:.3f}" for k in SCORE_NAMES),
                    "same_scores_at_6_decimals": all(f"{calculated[key][k]:.6f}" == f"{archived['calculated'][key][k]:.6f}" for k in SCORE_NAMES),
                    "held_out_probability_mean_absolute_change": float(np.abs(dp).mean()),
                    "held_out_probability_max_absolute_change": float(np.abs(dp).max()),
                }
        comparison = {"performed": True, "reference_internal_consistency_passed": archived["passed"],
                      "reference_matches_published_precision": archived["published_precision"]["all_match"],
                      "fits": differences,
                      "interpretation": "Differences from the archive do not by themselves invalidate a new optimization run."}
    report = {
        "passed": all(bool(v) for v in checks.values()),
        "internal_consistency_passed": all(bool(v) for v in checks.values()),
        "checks": {k: bool(v) for k, v in checks.items()},
        "counts": counts, "data_metadata": metadata, "sha256": hashes,
        "split_bytes_match_original_archive": hashes["data"]["fit_comparison_split.npz"] == SPLIT_SHA256,
        "calculated": calculated, "absolute_errors": errors, "geometry": geometry,
        "diagnostics": diagnostics, "figure_alignment": figure_report,
        "published_precision": published, "reference_comparison": comparison,
        "scope": "Independent reconstruction from raw data and saved parameters; no refitting, file writes, or manuscript-source dependencies.",
        "interpretation": "Internal consistency is not convergence certification. The comparison uses one split, and identical parameter priors induce different probability priors.",
    }
    # Enforce the public API's promise of finite, JSON-compatible output.
    json.dumps(report, allow_nan=False)
    return report
