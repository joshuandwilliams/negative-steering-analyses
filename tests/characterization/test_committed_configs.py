"""Every committed config must still translate to the engine command it ran.

These configs are the run definitions behind the benchmark. The engine that
reads them is a submodule, pinned to the commit the runs were made with, so
this asserts the pairing rather than the configs alone. An engine bump that
changed how a key is read, or a config edited by hand, shows up here instead of
in a number that still looks plausible.

The driver is `bin/negative_steering.py` in the engine. It is loaded by path,
the same way the engine's own tests load it, because `bin/` is a directory of
standalone entry points rather than an importable package.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE = Path(os.environ.get("NEGSTEER_ENGINE", REPO_ROOT / "engine"))
DRIVER = ENGINE / "bin" / "negative_steering.py"
REAL_CONFIG = (REPO_ROOT / "experiments" / "benchmarking" / "unconstrained"
               / "6G10" / "config.yml")


@pytest.fixture(scope="module")
def drv():
    if not DRIVER.is_file():
        pytest.skip(f"no engine at {ENGINE}. git submodule update --init")
    spec = importlib.util.spec_from_file_location("drv", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── the committed configs ────────────────────────────────────────────────────

@pytest.mark.logic
@pytest.mark.skipif(not REAL_CONFIG.is_file(), reason="benchmark config not present")
def test_committed_config_produces_the_expected_engine_call(drv):
    """The shipped 6G10 config still translates to the run that produced the results."""
    cfg = drv._load_yaml(REAL_CONFIG)
    cmd = drv._build_run_cmd(cfg, REAL_CONFIG.parent,
                             REAL_CONFIG.parent / "inputs",
                             REAL_CONFIG.parent / "run", "/fake.img")
    plan = cmd[cmd.index("--plan-extra-args") + 1]
    assert cmd[cmd.index("--seq-name") + 1] == "6G10"
    assert "--mode mild" in plan
    assert "--n-designs 20" in plan
    assert "--num-seeds 3" in plan
    assert "--boltz-constraints" not in plan


@pytest.mark.logic
def test_every_committed_config_is_translatable(drv):
    """No target's config can silently fail to build an engine command."""
    root = REPO_ROOT / "experiments" / "benchmarking"
    configs = sorted(root.glob("*/*/config.yml"))
    if not configs:
        pytest.skip("benchmarking tree not present")
    constrained = 0
    for c in configs:
        cfg = drv._load_yaml(c)
        cmd = drv._build_run_cmd(cfg, c.parent, c.parent / "inputs",
                                 c.parent / "run", "/fake.img")
        assert cmd[0] == "bash"
        plan = cmd[cmd.index("--plan-extra-args") + 1]
        if "--boltz-constraints" in plan:
            constrained += 1
        # The constraints toggle must follow the directory it lives in.
        assert ("--boltz-constraints" in plan) == (c.parent.parent.name == "constrained"), \
            f"{c}: constraints flag does not match its variant directory"
    assert constrained == len(configs) // 2, \
        f"expected half the configs constrained, got {constrained}/{len(configs)}"




@pytest.mark.logic
def test_every_ablation_config_is_translatable(drv):
    """Each control must ablate, and its arm must match its constraints toggle.

    A restrained control that ran unrestrained, or the reverse, would be the
    control for the other claim, which is the one mix-up this tree exists to
    avoid.
    """
    root = REPO_ROOT / "experiments" / "interface-ablation"
    configs = sorted(root.glob("*/*/config.yml"))
    if not configs:
        pytest.skip("interface-ablation tree not generated")
    for c in configs:
        variant = c.parent.parent.name
        cfg = drv._load_yaml(c)
        assert cfg["interface_ablation"] in ("alanine", "scrambled"), c
        prepare = drv._build_prepare_cmd(cfg, c.parent, c.parent / "inputs")
        assert prepare[prepare.index("--ablate") + 1] == cfg["interface_ablation"]
        run = drv._build_run_cmd(cfg, c.parent, c.parent / "inputs",
                                 c.parent / "run", "/fake.img")
        plan = run[run.index("--plan-extra-args") + 1]
        assert "--protected-set-source true_interface" in plan, c
        assert variant.endswith(("_restrained", "_unrestrained")), c
        expected = variant.endswith("_restrained")
        assert ("--boltz-constraints" in plan) == expected, \
            f"{c}: constraints flag does not match the arm in its directory"
        assert cfg["name"].endswith(variant.removeprefix("smoke_")), c


