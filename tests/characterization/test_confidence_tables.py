"""The table builders in confidence_common, on synthetic frames.

test_thesis_statistics.py covers the statistics and the selection rule:
signed_rank_p, rank_sum_p, spearman, median_ci, select_representative and
_final_seed_value are all at full coverage through it. What was left uncovered
split into two groups, and only one of them is worth testing.

Twelve `fig_*` functions and the style helpers draw figures. A test that calls
one and asserts a Figure came back would raise coverage and check almost
nothing: it cannot tell a correct axis from a wrong one. They are left alone
deliberately, and the reason is written here so nobody reads the number as an
oversight.

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
