#!/bin/bash
#
# Submit one of this repository's campaigns, with this site's partition and
# accounting.
#
#   ./scripts/submit.sh sweep                       # all five doses
#   ./scripts/submit.sh sweep     n10
#   ./scripts/submit.sh ablation  scrambled_restrained
#   ./scripts/submit.sh bench                       # the 18-target benchmark
#
# The campaigns live here. The engine they run lives in the `engine` submodule,
# or wherever NEGSTEER_REPO_DIR points, and has its own submit.sh for a single
# target or the test run.
#
# Each launcher carries the resources it needs in its own #SBATCH header, and
# none of those are site-specific. What IS site-specific is the partition
# name, and sometimes an account or a QOS. Those live in site.conf, and this
# passes them on the command line, where they beat both the environment and
# the header (sbatch(1), INPUT ENVIRONMENT VARIABLES).
#
# So `sbatch scripts/run_dose_sweep.slurm.sh <dose>` still works with no
# site.conf at all. The job then lands on the cluster's default partition,
# which is right for a head job.
#
# --dry-run prints the sbatch command and submits nothing.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"

# The loader lives in the engine, so there is one copy of the precedence rule.
# It reads this repository's site.conf, not the engine's.
ENGINE_DIR="${NEGSTEER_REPO_DIR:-${REPO}/engine}"
LOADER="${ENGINE_DIR}/scripts/site_conf.sh"
if [ ! -f "${LOADER}" ]; then
    echo "ERROR: no engine at ${ENGINE_DIR}." >&2
    echo "Run: git submodule update --init, or set NEGSTEER_REPO_DIR." >&2
    exit 1
fi
export NEGSTEER_SITE_CONF="${NEGSTEER_SITE_CONF:-${REPO}/site.conf}"
export NEGSTEER_REPO_DIR="${ENGINE_DIR}"
# shellcheck source=/dev/null
. "${LOADER}"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

ROLE="${1:-}"
[ "$#" -gt 0 ] && shift

# Every role here is a head job, so none of them wants a GPU. The GPU goes to
# the tasks the head job submits, sized by the engine's nextflow.config.
case "${ROLE}" in
    sweep)
        SCRIPT="${REPO}/scripts/run_dose_sweep.slurm.sh"
        PARTITION="${NEGSTEER_PARTITION_HEAD_LONG:-${NEGSTEER_PARTITION_HEAD:-}}"
        ;;
    ablation)
        SCRIPT="${REPO}/scripts/run_ablation_controls.slurm.sh"
        PARTITION="${NEGSTEER_PARTITION_HEAD:-}"
        ;;
    bench)
        # Not a head job and not sbatched here: submit_benchmarks.sh runs on
        # the login node and sbatches one GPU job per target itself, each with
        # its own partition from the engine's runner.
        exec bash "${REPO}/scripts/submit_benchmarks.sh" "$@"
        ;;
    *)
        echo "Usage: ./scripts/submit.sh [--dry-run] <role> [args]" >&2
        echo "" >&2
        echo "  sweep    [dose]      the dose sweep, one head job for all of it" >&2
        echo "  ablation [variant]   the interface-ablation controls" >&2
        echo "  bench    [args]      the 18-target benchmark, one job per target" >&2
        echo "" >&2
        echo "Partitions and accounting come from site.conf." >&2
        echo "The engine's own submit.sh runs a single target or the test run." >&2
        exit 1
        ;;
esac

if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: no launcher at ${SCRIPT}" >&2
    exit 1
fi

CMD=(sbatch)
[ -n "${PARTITION}" ]                && CMD+=(-p "${PARTITION}")
[ -n "${NEGSTEER_SLURM_ACCOUNT:-}" ] && CMD+=(-A "${NEGSTEER_SLURM_ACCOUNT}")
[ -n "${NEGSTEER_SLURM_QOS:-}" ]     && CMD+=(--qos "${NEGSTEER_SLURM_QOS}")
CMD+=("${SCRIPT}" "$@")

if [ "${DRY_RUN}" -eq 1 ]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

cd "${REPO}"
exec "${CMD[@]}"
