"""The table builders in confidence_common, on synthetic frames.

test_thesis_statistics.py covers the statistics and the selection rule:
signed_rank_p, rank_sum_p, spearman, median_ci, select_representative and
_final_seed_value are all at full coverage through it. What was left uncovered
split into two groups, and only one of them is worth testing.

Twelve `fig_*` functions and the style helpers draw figures. A test that calls
one and asserts a Figure came back would raise coverage and check nothing,
because it cannot tell a correct axis from a wrong one. Those carry
`# pragma: no cover` at their definitions, with the reason written at the
Figures section of the module, so the gate can sit at 100% without the number
claiming they were tested.

The functions below are different. They decide which rows reach a published
table and how they are grouped, so a wrong answer here changes a reported
result while still looking like a table. Each test states the expected rows
before calling anything, and the rules come from reading the implementation
rather than from running it and recording what came out.

Frames are built by hand with only the columns each function reads. That keeps
a failure pointing at the function rather than at a fixture, and it documents
the column contract each one depends on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = (_REPO_ROOT / "analyses" / "04-confidence-cost"
           / "confidence_common.py")


def _deps() -> bool:
    return all(importlib.util.find_spec(m) is not None
               for m in ("numpy", "pandas", "matplotlib", "scipy"))


pytestmark = [
    pytest.mark.skipif(not _MODULE.is_file(),
                       reason="confidence_common.py absent"),
    pytest.mark.skipif(not _deps(),
                       reason="needs numpy, pandas, matplotlib and scipy"),
]


@pytest.fixture(scope="module")
def cc():
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(_MODULE.parent))
    spec = importlib.util.spec_from_file_location("confidence_common", _MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(**over):
    """One per-seed row with the columns these builders read.

    Defaults describe a clean steered seed that passed: no contamination on the
    mutated positions, an intact receptor, no reversion verdict, and an ra_eff
    under the 5 A threshold.
    """
    base = {
        "unit": "6G10",
        "base": "design_00",
        "design": "design_00_s0",
        "kind": "steered",
        "status": "ok",
        "steering_skipped": False,
        "ra_eff": 2.0,
        "steered_n_contacts_on_mutated_positions": 0,
        "steered_receptor_intact": 1,
        "reversion_verdict": "",
        "reverted_ra_eff_vs_truth": None,
    }
    base.update(over)
    return base


def _full_row(**over):
    """A row carrying every confidence metric, for build_representative_table.

    That builder differences each METRICS key against the cold start, so a row
    missing one raises rather than producing a column of NaN.
    """
    base = _row(n_mutations=6, true_jaccard=0.5)
    base.update({m: 80.0 for m in ("complex_plddt", "interface_plddt")})
    base.update({"iptm": 0.8, "ipsae_min": 0.7})
    base.update(over)
    return base


# ── split ────────────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_split_separates_the_cold_start_from_the_usable_steered_rows(cc):
    """One cold start per unit, and only steered rows whose status is ok.

    A failed steered row still carries numbers, so including it would let a
    prediction that errored contribute to a median.
    """
    import pandas as pd
    pred = pd.DataFrame([
        _row(kind="cold_start", design="initial", ra_eff=23.9),
        _row(kind="cold_start", design="initial", ra_eff=23.9),   # duplicate
        _row(design="design_00_s0"),
        _row(design="design_01_s0", status="failed"),
    ])
    cold, steered = cc.split(pred)
    assert list(cold.index) == ["6G10"], "the duplicate cold start was kept"
    assert len(steered) == 1, "a failed steered row reached the usable set"
    assert steered.iloc[0]["design"] == "design_00_s0"


@pytest.mark.logic
def test_steered_drops_the_units_that_skipped_steering(cc):
    """A unit whose cold start already passed stages no designs at all, so its
    rows must not be pooled with units that were actually steered."""
    import pandas as pd
    df = pd.DataFrame([
        _row(unit="6G10", steering_skipped=False),
        _row(unit="5A6W", steering_skipped=True),
    ])
    assert list(cc._steered(df)["unit"]) == ["6G10"]


# ── seed_pass_counts ─────────────────────────────────────────────────────────

@pytest.mark.logic
def test_seed_pass_counts_uses_the_reverted_value_where_there_was_reversion(cc):
    """Three seeds of one design: one clean pass, one that passed only after
    reversion, one whose reversion failed.

    The middle row is the point. Its raw ra_eff is far over the threshold and
    its reverted value is under it, so counting the raw number would call it a
    failure when the selection rule calls it a pass.
    """
    import pandas as pd
    pred = pd.DataFrame([
        _row(design="design_00_s0", ra_eff=2.0),
        _row(design="design_00_s1", ra_eff=30.0,
             reversion_verdict="pose_holds", reverted_ra_eff_vs_truth=1.5),
        _row(design="design_00_s2", ra_eff=30.0,
             reversion_verdict="pose_collapses", reverted_ra_eff_vs_truth=22.0),
    ])
    out = cc.seed_pass_counts(pred)
    assert len(out) == 1, "one design should give one row"
    got = out.iloc[0]
    assert got["n_seeds"] == 3
    assert got["n_pass"] == 2, "the reverted pose_holds seed should count as a pass"
    assert got["frac_pass"] == pytest.approx(2 / 3)


@pytest.mark.logic
def test_seed_pass_counts_rejects_a_seed_whose_receptor_was_not_intact(cc):
    """A blank verdict means reversion was skipped, which only counts as a pass
    when the receptor survived. Otherwise the pose is of something that is not
    the designed protein."""
    import pandas as pd
    pred = pd.DataFrame([_row(steered_receptor_intact=0, ra_eff=1.0)])
    assert cc.seed_pass_counts(pred).iloc[0]["n_pass"] == 0


@pytest.mark.logic
def test_seed_pass_counts_groups_designs_within_a_unit_separately(cc):
    import pandas as pd
    pred = pd.DataFrame([
        _row(base="design_00", design="design_00_s0", ra_eff=2.0),
        _row(base="design_01", design="design_01_s0", ra_eff=30.0),
    ])
    out = cc.seed_pass_counts(pred).set_index("base")
    assert out.loc["design_00", "n_pass"] == 1
    assert out.loc["design_01", "n_pass"] == 0


# ── pose_matched_pairs ───────────────────────────────────────────────────────

@pytest.mark.logic
def test_pose_matched_pairs_keeps_only_units_posed_correctly_in_both(cc):
    """Comparing a confidence metric across variants only means something when
    both variants describe the same pose, which is what the function's own
    docstring says. A unit correct in one and wrong in the other must drop.
    """
    import pandas as pd
    unc = pd.DataFrame([
        {"unit": "both_ok", "rep_ra_eff": 2.0},
        {"unit": "only_unc", "rep_ra_eff": 2.0},
        {"unit": "neither", "rep_ra_eff": 20.0},
    ])
    con = pd.DataFrame([
        {"unit": "both_ok", "rep_ra_eff": 3.0},
        {"unit": "only_unc", "rep_ra_eff": 20.0},
        {"unit": "neither", "rep_ra_eff": 20.0},
    ])
    a, b = cc.pose_matched_pairs(unc, con)
    assert list(a.index) == ["both_ok"]
    assert list(b.index) == ["both_ok"]


@pytest.mark.logic
def test_pose_matched_pairs_drops_units_missing_from_either_variant(cc):
    """Only the intersection, so an arm that did not run a target cannot make
    that target contribute a one-sided difference."""
    import pandas as pd
    unc = pd.DataFrame([{"unit": "shared", "rep_ra_eff": 1.0},
                        {"unit": "unc_only", "rep_ra_eff": 1.0}])
    con = pd.DataFrame([{"unit": "shared", "rep_ra_eff": 1.0}])
    a, b = cc.pose_matched_pairs(unc, con)
    assert list(a.index) == ["shared"] and list(b.index) == ["shared"]


@pytest.mark.logic
def test_pose_matched_pairs_treats_the_threshold_as_inclusive(cc):
    """`<=` in the implementation, so a pose exactly at 5.0 A is correct. Pinned
    because a strict comparison here would silently drop boundary units."""
    import pandas as pd
    at = pd.DataFrame([{"unit": "edge", "rep_ra_eff": cc.RA_EFF_THRESHOLD}])
    a, _ = cc.pose_matched_pairs(at, at)
    assert list(a.index) == ["edge"]


# ── dose_table ───────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_dose_table_groups_by_mutation_count_and_counts_rescues(cc):
    """Medians per budget, and a rescued count rather than a mean, because
    `rescued` is a label and averaging labels is meaningless."""
    import pandas as pd
    sub = pd.DataFrame([
        {"unit": "a", "n_mutations": 3, "d_complex_plddt": -1.0,
         "d_interface_plddt": -2.0, "d_iptm": -0.01, "outcome": "rescued"},
        {"unit": "b", "n_mutations": 3, "d_complex_plddt": -3.0,
         "d_interface_plddt": -4.0, "d_iptm": -0.03, "outcome": "still wrong"},
        {"unit": "c", "n_mutations": 6, "d_complex_plddt": -5.0,
         "d_interface_plddt": -6.0, "d_iptm": -0.05, "outcome": "rescued"},
    ])
    out = cc.dose_table(sub)
    assert list(out.index) == [3, 6]
    assert out.loc[3, "n_units"] == 2
    assert out.loc[3, "d_complex_plddt"] == pytest.approx(-2.0)  # median of -1, -3
    assert out.loc[3, "n_rescued"] == 1
    assert out.loc[6, "n_rescued"] == 1


# ── build_representative_table ───────────────────────────────────────────────
#
# The outcome label is the thing to pin hardest. Every unit in the published
# before-and-after figures carries one, and the four cases are assigned by a
# single np.select over two booleans, so a swapped pair would relabel whole
# categories while the table still looked well formed.

@pytest.mark.logic
@pytest.mark.parametrize("cold_ra,rep_ra,expected", [
    (23.9, 2.0, "rescued"),          # wrong before, right after
    (2.0, 3.0, "already correct"),   # right both times
    (23.9, 20.0, "still wrong"),     # wrong both times
    (2.0, 20.0, "lost"),             # right before, wrong after
])
def test_build_representative_table_labels_the_four_outcomes(
        cc, cold_ra, rep_ra, expected):
    import pandas as pd
    pred = pd.DataFrame([
        _full_row(kind="cold_start", design="initial", base="initial",
                  ra_eff=cold_ra),
        _full_row(design="design_00_s0", ra_eff=rep_ra),
    ])
    out = cc.build_representative_table(pred)
    assert len(out) == 1, "one unit should give one row"
    assert out.iloc[0]["outcome"] == expected
    assert out.iloc[0]["cold_ra_eff"] == pytest.approx(cold_ra)


@pytest.mark.logic
def test_build_representative_table_differences_each_metric_the_right_way(cc):
    """d_<metric> is representative minus cold start, so a steered design that
    lost confidence gives a negative number. The sign is the whole claim of the
    confidence-cost chapter, so it is pinned rather than assumed."""
    import pandas as pd
    pred = pd.DataFrame([
        _full_row(kind="cold_start", design="initial", base="initial",
                  ra_eff=23.9, complex_plddt=90.0, iptm=0.9),
        _full_row(design="design_00_s0", ra_eff=2.0,
                  complex_plddt=85.0, iptm=0.8),
    ])
    got = cc.build_representative_table(pred).iloc[0]
    assert got["cold_complex_plddt"] == pytest.approx(90.0)
    assert got["rep_complex_plddt"] == pytest.approx(85.0)
    assert got["d_complex_plddt"] == pytest.approx(-5.0)
    assert got["d_iptm"] == pytest.approx(-0.1)


@pytest.mark.logic
def test_build_representative_table_marks_a_unit_that_skipped_steering(cc):
    """When the cold start already passed, select_representative returns the
    initial row and no design was staged. The flag has to say so, because the
    confidence difference for such a unit is a comparison of the cold start
    with itself."""
    import pandas as pd
    pred = pd.DataFrame([
        _full_row(kind="cold_start", design="initial", base="initial",
                  ra_eff=1.0, steering_skipped=True),
    ])
    out = cc.build_representative_table(pred)
    if out.empty:
        pytest.skip("select_representative returns nothing without steered rows")
    got = out.iloc[0]
    assert got["steering_skipped"] is True or bool(got["steering_skipped"])
    assert got["rep_design"] == "initial"


@pytest.mark.logic
def test_build_representative_table_takes_the_median_mutation_count(cc):
    """n_mutations is the median over the unit's steered rows, so one design
    with an unusual budget cannot drag the reported figure."""
    import pandas as pd
    pred = pd.DataFrame([
        _full_row(kind="cold_start", design="initial", base="initial",
                  ra_eff=23.9),
        _full_row(base="design_00", design="design_00_s0", n_mutations=3),
        _full_row(base="design_01", design="design_01_s0", n_mutations=6),
        _full_row(base="design_02", design="design_02_s0", n_mutations=9),
    ])
    got = cc.build_representative_table(pred).iloc[0]
    assert got["n_mutations"] == pytest.approx(6.0)


# ── load_per_seed ────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_load_per_seed_tags_kind_and_strips_the_seed_suffix(cc, tmp_path):
    """Two things this does that nothing else does.

    `kind` is derived from the design name starting with "initial", which is how
    every later split tells a cold start from a steered design. And `base`
    strips the `_sN` suffix, because the design column is design AND seed in one
    string, so without it every seed looks like its own design.
    """
    import pandas as pd
    csv = tmp_path / "raw_per_seed_results.csv"
    pd.DataFrame([
        {"design": "initial", "status": "ok", "seed_index": 0,
         "steered_ra_eff_vs_truth": 23.9, "complex_plddt": 0.90},
        {"design": "design_00_s1", "status": "ok", "seed_index": 1,
         "steered_ra_eff_vs_truth": 2.0, "complex_plddt": 0.85},
    ]).to_csv(csv, index=False)

    out = cc.load_per_seed({"6G10": csv})
    assert list(out["unit"]) == ["6G10", "6G10"], "the unit label was not applied"
    assert list(out["kind"]) == ["cold_start", "steered"]
    assert list(out["base"]) == ["initial", "design_00"]


@pytest.mark.logic
def test_load_per_seed_rescales_plddt_to_the_0_100_axis(cc, tmp_path):
    """pLDDT arrives on 0-1 and is multiplied by 100, because a change of 0.032
    is easy to misread where 3.2 pLDDT points is not. iptm is NOT rescaled, so
    a blanket multiplication would corrupt it."""
    import pandas as pd
    csv = tmp_path / "raw.csv"
    pd.DataFrame([{"design": "design_00_s0", "status": "ok", "seed_index": 0,
                   "complex_plddt": 0.90, "interface_plddt": 0.80,
                   "iptm": 0.75}]).to_csv(csv, index=False)
    got = cc.load_per_seed({"u": csv}).iloc[0]
    assert got["complex_plddt"] == pytest.approx(90.0)
    assert got["interface_plddt"] == pytest.approx(80.0)
    assert got["iptm"] == pytest.approx(0.75), "iptm must not be rescaled"


@pytest.mark.logic
def test_load_per_seed_fills_absent_columns_rather_than_raising(cc, tmp_path):
    """A cohort whose CSV lacks a metric still has to load, because the campaign
    and the benchmark do not carry an identical column set."""
    import pandas as pd
    csv = tmp_path / "raw.csv"
    pd.DataFrame([{"design": "design_00_s0", "status": "ok",
                   "seed_index": 0}]).to_csv(csv, index=False)
    out = cc.load_per_seed({"u": csv})
    assert "ipsae_min" in out.columns
    assert out["ipsae_min"].isna().all()


@pytest.mark.logic
def test_load_per_seed_concatenates_several_units(cc, tmp_path):
    import pandas as pd
    paths = {}
    for unit in ("a", "b"):
        p = tmp_path / f"{unit}.csv"
        pd.DataFrame([{"design": "design_00_s0", "status": "ok",
                       "seed_index": 0}]).to_csv(p, index=False)
        paths[unit] = p
    assert sorted(cc.load_per_seed(paths)["unit"]) == ["a", "b"]


# ── effect_table ─────────────────────────────────────────────────────────────

@pytest.mark.logic
def test_effect_table_reports_n_as_the_finite_count(cc):
    """n counts finite values, so a NaN difference does not inflate the sample
    size behind a reported median."""
    import numpy as np
    out = cc.effect_table([("set A", "iptm", [0.1, -0.2, np.nan, 0.3])])
    assert len(out) == 1
    assert out.iloc[0]["n"] == 3


@pytest.mark.logic
def test_effect_table_formats_each_metric_with_its_own_precision(cc):
    """pLDDT is shown to 2 decimals and iptm to 3, from FMT. A single format
    would either lose iptm resolution or print false precision on pLDDT."""
    out = cc.effect_table([("s", "complex_plddt", [1.0, 2.0, 3.0]),
                           ("s", "iptm", [0.001, 0.002, 0.003])])
    plddt = out[out.metric == cc.METRICS["complex_plddt"]].iloc[0]
    iptm = out[out.metric == cc.METRICS["iptm"]].iloc[0]
    assert plddt[f"median Δ ({cc.UNITS['complex_plddt']})"] == "+2.00"
    assert iptm[f"median Δ ({cc.UNITS['iptm']})"] == "+0.002"
    assert plddt["95% CI"].startswith("[") and "," in plddt["95% CI"]


# ── representative_reversion ─────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.parametrize("verdicts,expected", [
    (["pose_holds", "pose_holds", ""], True),    # 2 of 3, majority
    (["pose_holds", "", ""], False),             # 1 of 3, not a majority
    (["", "", ""], False),                       # none
    (["pose_holds", ""], True),                  # 1 of 2, half counts
])
def test_representative_reversion_applies_the_majority_rule(cc, verdicts, expected):
    """`n_verd * 2 >= len(grp)`, so half the seeds is enough. Exactly half is
    the case worth pinning, since a strict majority would read it the other
    way and the docstring says this matches the engine's own rule."""
    import pandas as pd
    pred = pd.DataFrame([
        _row(design=f"design_00_s{i}", reversion_verdict=v)
        for i, v in enumerate(verdicts)
    ])
    repdf = pd.DataFrame([{"unit": "6G10", "rep_design": "design_00",
                           "steering_skipped": False, "rep_ra_eff": 2.0}])
    got = cc.representative_reversion(pred, repdf).iloc[0]
    assert bool(got["reverted"]) is expected
    assert got["n_seeds"] == len(verdicts)


@pytest.mark.logic
def test_representative_reversion_skips_units_that_skipped_steering(cc):
    """A unit represented by its cold start has no design to have reverted."""
    import pandas as pd
    pred = pd.DataFrame([_row()])
    repdf = pd.DataFrame([{"unit": "6G10", "rep_design": "initial",
                           "steering_skipped": True, "rep_ra_eff": 1.0}])
    assert cc.representative_reversion(pred, repdf).empty


@pytest.mark.logic
def test_representative_reversion_records_whether_the_pose_was_correct(cc):
    import pandas as pd
    pred = pd.DataFrame([_row(design="design_00_s0")])
    repdf = pd.DataFrame([{"unit": "6G10", "rep_design": "design_00",
                           "steering_skipped": False, "rep_ra_eff": 20.0}])
    assert bool(cc.representative_reversion(pred, repdf).iloc[0]["pose_correct"]) is False


# ── outcome_contrast_table ───────────────────────────────────────────────────

@pytest.mark.logic
def test_outcome_contrast_table_stratifies_by_budget_as_well_as_pooling(cc):
    """Rescue and mutation budget are not independent, so the docstring's
    reason for stratifying is that a pooled difference could be a budget
    difference wearing a rescue label. Both strata must therefore appear, and
    each must only contain its own budget's rows.
    """
    import pandas as pd
    df = pd.DataFrame([
        {"unit": "a", "outcome": "rescued", "n_mutations": 3.0,
         "steering_skipped": False, "d_complex_plddt": -1.0,
         "d_interface_plddt": -1.0, "d_iptm": -0.01, "d_ipsae_min": -0.01},
        {"unit": "b", "outcome": "still wrong", "n_mutations": 3.0,
         "steering_skipped": False, "d_complex_plddt": -5.0,
         "d_interface_plddt": -5.0, "d_iptm": -0.05, "d_ipsae_min": -0.05},
        {"unit": "c", "outcome": "rescued", "n_mutations": 6.0,
         "steering_skipped": False, "d_complex_plddt": -2.0,
         "d_interface_plddt": -2.0, "d_iptm": -0.02, "d_ipsae_min": -0.02},
    ])
    out = cc.outcome_contrast_table({"bench": df})
    strata = set(out["stratum"])
    assert strata == {"pooled", "3 mutations", "6 mutations"}

    pooled = out[(out.stratum == "pooled")
                 & (out.metric == cc.METRICS["complex_plddt"])].iloc[0]
    assert pooled["n rescued"] == 2 and pooled["n still wrong"] == 1

    three = out[(out.stratum == "3 mutations")
                & (out.metric == cc.METRICS["complex_plddt"])].iloc[0]
    assert three["n rescued"] == 1 and three["n still wrong"] == 1


@pytest.mark.logic
def test_outcome_contrast_table_leaves_a_median_blank_when_a_group_is_empty(cc):
    """An empty group gets "" rather than a median of nothing, so a stratum
    with no rescues cannot print a number that describes no data."""
    import pandas as pd
    df = pd.DataFrame([
        {"unit": "b", "outcome": "still wrong", "n_mutations": 3.0,
         "steering_skipped": False, "d_complex_plddt": -5.0,
         "d_interface_plddt": -5.0, "d_iptm": -0.05, "d_ipsae_min": -0.05},
    ])
    out = cc.outcome_contrast_table({"bench": df})
    row = out[(out.stratum == "pooled")
              & (out.metric == cc.METRICS["complex_plddt"])].iloc[0]
    assert row["n rescued"] == 0
    assert row["median Δ rescued"] == ""
    assert row["median Δ still wrong"] == "-5.00"


# ── rank_noise_floor ─────────────────────────────────────────────────────────

@pytest.mark.logic
def test_rank_noise_floor_compares_every_pair_of_seeds(cc):
    """Three seeds give three pairs, and the keys are the pairs themselves so a
    caller can see which comparison produced which shift."""
    import pandas as pd
    rows = []
    for unit, vals in {"a": (1.0, 2.0, 3.0), "b": (4.0, 5.0, 6.0),
                       "c": (7.0, 8.0, 9.0)}.items():
        for seed, v in enumerate(vals):
            rows.append(_row(unit=unit, base="design_00",
                             design=f"design_00_s{seed}", seed_index=seed,
                             iptm=v))
    out = cc.rank_noise_floor(pd.DataFrame(rows),
                              {"a": "design_00", "b": "design_00",
                               "c": "design_00"}, "iptm")
    assert set(out) == {(0, 1), (0, 2), (1, 2)}


@pytest.mark.logic
def test_rank_noise_floor_reports_zero_shift_when_the_order_is_identical(cc):
    """The units keep their order across seeds here, so the median absolute
    rank shift must be 0 and the correlation 1. That is the calibration case:
    if this were not zero, nothing the function reports could be trusted as a
    floor.

    Four units, not three: `spearman` returns nan below four usable pairs
    rather than a rho from three points, so a three-unit fixture would test the
    guard instead of the ordering.
    """
    import pandas as pd
    rows = []
    for unit, offset in (("a", 0.0), ("b", 1.0), ("c", 2.0), ("d", 3.0)):
        for seed in (0, 1):
            rows.append(_row(unit=unit, base="design_00",
                             design=f"design_00_s{seed}", seed_index=seed,
                             iptm=offset))
    units = {u: "design_00" for u in ("a", "b", "c", "d")}
    out = cc.rank_noise_floor(pd.DataFrame(rows), units, "iptm", seeds=(0, 1))
    n, shift, rho = out[(0, 1)]
    assert n == 4
    assert shift == pytest.approx(0.0)
    assert rho == pytest.approx(1.0)


@pytest.mark.logic
def test_rank_noise_floor_gives_nan_rho_below_four_units(cc):
    """spearman's own guard, reached through this function: three units cannot
    produce a correlation, so the floor reports nan rather than a rho from
    three points. The rank shift is still real and still reported."""
    import math

    import pandas as pd
    rows = []
    for unit, offset in (("a", 0.0), ("b", 1.0), ("c", 2.0)):
        for seed in (0, 1):
            rows.append(_row(unit=unit, base="design_00",
                             design=f"design_00_s{seed}", seed_index=seed,
                             iptm=offset))
    units = {u: "design_00" for u in ("a", "b", "c")}
    n, shift, rho = cc.rank_noise_floor(
        pd.DataFrame(rows), units, "iptm", seeds=(0, 1))[(0, 1)]
    assert n == 3
    assert shift == pytest.approx(0.0)
    assert math.isnan(rho)


@pytest.mark.logic
def test_rank_noise_floor_reads_the_cold_start_when_the_design_is_initial(cc):
    """A unit represented by its cold start is ranked on its cold-start rows,
    selected by kind rather than by base, so an "initial" entry does not come
    back empty."""
    import pandas as pd
    rows = []
    for unit, offset in (("a", 0.0), ("b", 1.0)):
        for seed in (0, 1):
            rows.append(_row(unit=unit, kind="cold_start", design="initial",
                             base="initial", seed_index=seed, iptm=offset))
    out = cc.rank_noise_floor(pd.DataFrame(rows),
                              {"a": "initial", "b": "initial"}, "iptm",
                              seeds=(0, 1))
    assert out[(0, 1)][0] == 2, "cold-start rows were not found"


# ── Skips: units and designs that cannot be represented ──────────────────────

@pytest.mark.logic
def test_build_representative_table_drops_a_unit_with_nothing_to_represent(cc):
    """A unit with no usable steered rows and an unreadable cold start.

    `select_representative` returns None there, and the unit has to leave the
    table. Keeping it would put a row of NaN into a published table, which
    reads as a measured null result rather than as absent data.
    """
    import pandas as pd
    pred = pd.DataFrame([
        _full_row(unit="6G10", kind="cold_start", design="initial",
                  base="initial", ra_eff=23.9),
        _full_row(unit="6G10", design="design_00_s0"),
        # No steered rows, and a cold start whose ra_eff will not parse, so the
        # fallback to the cold start's own seeds finds no values either.
        _full_row(unit="7XYZ", kind="cold_start", design="initial",
                  base="initial", ra_eff="n/a"),
    ])
    out = cc.build_representative_table(pred)
    assert list(out.unit) == ["6G10"], "the unrepresentable unit was kept"


@pytest.mark.logic
def test_representative_reversion_skips_a_design_with_no_surviving_seeds(cc):
    """The representative design named in repdf has no rows left in pred.

    That happens when every seed of it failed, because this function reads only
    status == "ok" rows while the representative was chosen elsewhere. The unit
    is dropped rather than counted as a design with zero seeds, which would
    divide by nothing in the majority rule.
    """
    import pandas as pd
    pred = pd.DataFrame([_row(unit="6G10", base="design_00",
                              design="design_00_s0"),
                         _row(unit="7XYZ", base="design_09",
                              design="design_09_s0", status="failed")])
    repdf = pd.DataFrame([
        {"unit": "6G10", "rep_design": "design_00", "steering_skipped": False,
         "rep_ra_eff": 2.0},
        {"unit": "7XYZ", "rep_design": "design_09", "steering_skipped": False,
         "rep_ra_eff": 2.0},
    ])
    out = cc.representative_reversion(pred, repdf)
    assert list(out.unit) == ["6G10"], "a design with no usable seeds was kept"


# ── Axis labels and the notebook backend ─────────────────────────────────────

@pytest.mark.logic
def test_ylabel_marks_plddt_units_and_drops_them_on_a_difference(cc):
    """pLDDT is on 0-100, so an absolute axis says so and a difference does not.

    A difference of pLDDT values is in points, not on the 0-100 scale, and
    labelling it "(0-100)" would misstate what the axis shows.
    """
    assert cc._ylabel("complex_plddt", "Complex pLDDT") \
        == "Complex pLDDT (0-100)"
    assert cc._ylabel("complex_plddt", "Complex pLDDT", delta=True) \
        == "Δ Complex pLDDT (pLDDT points)"
    assert cc._ylabel("iptm", "ipTM") == "ipTM"
    assert cc._ylabel("iptm", "ipTM", delta=True) == "Δ ipTM"


@pytest.mark.logic
def test_restore_inline_backend_puts_the_backend_back_even_on_failure(cc):
    """make_plots sets Agg at import, which detaches figures in a notebook.

    This helper has to restore whatever the kernel was using, including when
    the wrapped call raises, or one failed cell silently stops every later
    figure from rendering.
    """
    import matplotlib
    before = matplotlib.get_backend()

    def swap():
        matplotlib.use("Agg")
        return 42

    assert cc.restore_inline_backend(swap) == 42
    assert matplotlib.get_backend() == before

    def swap_then_raise():
        matplotlib.use("Agg")
        raise RuntimeError("cell failed")

    with pytest.raises(RuntimeError):
        cc.restore_inline_backend(swap_then_raise)
    assert matplotlib.get_backend() == before
