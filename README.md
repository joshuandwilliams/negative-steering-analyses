# negative-steering-analyses

The benchmark behind
[`structure-negative-steering`](https://github.com/joshuandwilliams/structure-negative-steering),
and the nine analysis documents that read it. The engine itself is not here. It
is the `engine` submodule, pinned to the commit these runs were made with.

```bash
git clone --recurse-submodules \
    https://github.com/joshuandwilliams/negative-steering-analyses.git
```

An existing clone gets the engine with `git submodule update --init`.

## Results

18 solved complexes, each an integrated HMA decoy domain (Pikp, Pikm,
RGA5-HMA) bound to an AVR effector. A pose counts as correct when the
receptor-aligned effector RMSD is under 5 Å, scored on the median of three
seeds.

| Method | Correctly posed | From |
|:---|---:|:---|
| Boltz-2, no MSA | 5 / 18 | `02-model-choice/boltz2_benchmark_comparison.csv` |
| Boltz-2 with reference restraints | 13 / 18 | the same file |
| Boltz-2 with negative steering, 50 sets | 15 / 18 | `01-pose-rescue/dose_sweep_pass_counts.csv` |
| Both together, 20 sets | 18 / 18 | `01-pose-rescue/benchmark_summary.csv` |

Accuracy climbs with the number of steering mutation sets sampled, so the count
is a cost decision rather than a fixed requirement.

| Steering sets | 0 | 3 | 5 | 10 | 20 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| Correctly posed | 4 | 9 | 9 | 11 | 13 | 15 |

Reference restraints are a ceiling, not a rival method. They are derived from
the solved structure, so they cannot run on a novel target. They are here to
show the failures are not intrinsic to Boltz-2.

The 18/18 row carries that caveat twice over. It hands the model the interface
through the restraints and through the protected set.
`experiments/interface-ablation/` is the control for it. That arm permutes the
receptor's interface residues at fixed composition, so a pose still landing
inside the threshold is a false positive by construction.

## What is here

| Path | Contents |
|:---|:---|
| `experiments/` | Run definitions, one `config.yml` per unit. Outputs are HPC-authoritative and gitignored |
| `analyses/` | One folder per analysis document, with its own figures and the data it owns |
| `data/reference_complexes/` | The 18 solved complexes, one copy each |
| `scripts/` | The campaign launchers and the two config generators |
| `contrib/` | Sync helpers and the provenance of one CSV. Nothing in a campaign calls them |
| `engine/` | The engine, as a submodule. Not edited here |
| `site.conf.example` | Every setting that is specific to one cluster |
| `tests/` | What holds the configs, the generators and the statistics to their claims |

Every config in every tree points at `data/reference_complexes/`, so no two
arms can disagree about what truth is.

## Installation

```bash
mamba env create -f environment.yml
mamba activate negsteer-analysis
```

Quarto is not a conda package here. Install it separately and keep it at
1.9.38, the version the committed HTML was rendered with. Figures only
regenerate byte-identically under the pinned matplotlib, so use this
environment rather than a pip virtualenv for anything that writes one.

### Your cluster

Nothing here names a partition, a container path or an email address.
Everything that would is in one file:

```bash
cp site.conf.example site.conf    # then fill in the partitions and the image
```

`site.conf` is gitignored and `site.conf.example` documents every setting. The
minimum for a real run is `NEGSTEER_BOLTZ_IMG` and `NEGSTEER_PARTITION_GPU`.

## Running a campaign

Each campaign is a head job that submits one GPU task per prediction. The
engine is found through the submodule, so nothing needs a path on the command
line.

```bash
./scripts/submit.sh sweep                  # all five doses, one head job
./scripts/submit.sh sweep n10              # one dose
./scripts/submit.sh ablation scrambled_restrained
./scripts/submit.sh bench --dry-run        # the 18-target benchmark
```

`--dry-run` prints the `sbatch` command and submits nothing. `-resume` is
always passed, so a head job that hits its walltime can be resubmitted with
the identical command and skips what already finished.

The two trees that are generated rather than written by hand:

```bash
python3 scripts/make_dose_sweep.py            # the 90 dose-sweep configs
python3 scripts/make_ablation_controls.py     # the interface-ablation controls
```

Each dose-sweep config is the tracked benchmark config with three lines
changed, `name`, `reference_pdb` and `negsteer_n_designs`. The generator
refuses to write one that differs anywhere else, and
`tests/characterization/test_dose_sweep_configs.py` asserts the same thing
about what is committed.

## Analyses

Nine Quarto documents, numbered in the order Chapter 4 uses them. Each folder
carries its own `.qmd`, figures, and the data it owns, so a figure's provenance
is its directory. Rendered HTML is a build artefact and is gitignored.

| Document | Asks |
|:---|:---|
| `01-pose-rescue` | does steering rescue the pose |
| `02-model-choice` | steering against reference restraints |
| `03-compute-cost` | what a steering run costs |
| `04-confidence-cost` | does steering cost confidence |
| `05-campaign-confidence` | the same question on a design campaign |
| `06-confidence-synthesis` | the two confidence results together |
| `07-ipsae-instability` | whether ipSAE ranks reliably at this scale |
| `08-tier-failure-breakdown` | how often each complex is corrected, and how it fails |
| `09-mutation-excluded-ipsae` | ipSAE with the steering mutations excluded |

To regenerate, pull the summaries down from the cluster (about 16 MB, no raw
model outputs) and re-render:

```bash
contrib/sync_from_hpc.sh --name benchmarking --results-only
cd analyses/01-pose-rescue && quarto render 01-pose-rescue.qmd
```

Every analysis falls back to its committed cached table when the run tree is
absent, so all of them re-render anywhere.

## Tests

```bash
pytest -m logic                  # everything that needs no run
pytest                           # adds the assertions on a finished run tree
```

`logic` covers the committed configs, the generators that wrote them, and the
statistics and selection rules behind the documents. The config tests import
the engine's driver from the submodule, so they assert the pairing of a config
with the engine that read it, not the config alone. They skip with a reason
when the submodule is not checked out.

`check_test_run` asserts on a finished benchmark run. It skips cleanly when no
run tree is present, and `NEGSTEER_BENCH_DIR` points it elsewhere.

## Provenance

Which runner produced which committed table.

| cohort | runner | committed outputs |
|:---|:---|:---|
| `experiments/dose-sweep/` (90 target × dose units) | the Nextflow workflow, head job `nf_dosesweep` | the pose-rescue figures and `n50_pose_metrics.csv`, `dockq_scores.csv`, `dose_sweep_pass_counts.csv`, `dose_sweep_maxrss.csv`, `tier_failure_breakdown_n50.csv`, and the steered series in `02-model-choice` |
| `experiments/benchmarking/` (18 targets × 2 arms) | the single-job runner, one `sbatch` per target | `benchmark_summary.csv`, `negsteer_maxrss.csv`, `confidence_per_prediction*.csv` |
| the `pikp1_avrpikf` design campaign | `receptor-resurfacing-pipeline`, not this repository | `campaign_data/*.csv`, `sequence_match.csv`, `ipsae_*.csv`, `mutation_excluded_*.csv` |

The provenance of each memory figure is recorded at
`analyses/03-compute-cost/stage_runtime_comparison.py:106-122`, which is where
the split is visible: the dose-sweep cohort's peak memory comes from a Nextflow
trace, the 18-target cohort's from Slurm `MaxRSS` because those jobs produced
no trace.

Two staged inputs come from outside this repository.
`boltz2_benchmark_comparison.csv` holds unconstrained and reference-restrained
Boltz-2 `ra_eff` per target, from `structure-prediction-benchmarking`'s
`combined_metrics.csv`. `negsteer_maxrss.csv` holds peak memory from Slurm
accounting, collected on the cluster by `contrib/collect_negsteer_maxrss.sh`
and copied back.

## Known limits

The dose axis is not a controlled titration. Only two budgets are well
populated, 3 and 6 mutations, and budget is constant within a target. It is
always a between-target contrast, so a within-target budget sweep would settle
it.

The benchmark generates 20 candidate steered designs per target, so the
representative is the best of 20. Any comparison against a cohort run at a
different depth carries that difference.

Pooling per-target medians gives each target one vote regardless of how many
predictions it contributed. Absence of a difference is not evidence of
equivalence. These tests can exclude an effect the size of the observed pLDDT
drop, not an arbitrarily small one.

## Citing

The engine has its own `CITATION.cff`. Cite this repository for the benchmark
and the analyses, and that one for the method and the code.
