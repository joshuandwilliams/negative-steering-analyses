#!/bin/bash
#
# Sync this repository to the HPC via SSH.
#
# Usage:
#   ./contrib/sync_to_hpc.sh              # real sync
#   ./contrib/sync_to_hpc.sh --dry        # dry run, no changes
#   ./contrib/sync_to_hpc.sh --list-stale # report cluster-only paths, change nothing
#
# Why --list-stale exists:
#   rsync --delete does NOT delete excluded paths. It protects them, which is
#   the behaviour that stops this push destroying results. The cost is that
#   anything matching an exclude survives on the cluster forever, so a renamed
#   or dropped directory is never cleaned up and the cluster silently
#   accumulates two of everything. --list-stale reports what is up there and
#   not down here so pruning is a deliberate, visible act.
#
#   It only ever reads. It prints commands; it does not run them, and this
#   script never deletes anything on the cluster. --delete-excluded would
#   "fix" the accumulation by wiping every result. Never use it.
#
# Direction and authority:
#   The Mac is authoritative for the configs, the scripts and the analysis
#   documents. The cluster is authoritative for every run output. The HPC has
#   no git.
#
# CRITICAL — the run outputs are protected:
#   Every config in experiments/ sits beside the inputs/ and run/ trees its
#   run produced, which are gitignored and HPC-only. The first pass below
#   therefore excludes experiments/ entirely, so --delete can never wipe a
#   result. A second pass pushes the config.yml files additively, with no
#   --delete at all.
#
#   No *.pdb is pushed from experiments/. The eighteen reference complexes go
#   up once from data/, which is the copy every config resolves to.
#
# Also excluded:
#   - analyses/  The Quarto documents are authored and rendered on the Mac from
#     results pulled down by sync_from_hpc.sh. Nothing on the HPC reads them,
#     and the rendered .html plus the *_files/ bundle are gitignored build
#     artefacts that do not belong on the cluster.
#   - engine/  The engine is its own repository. Push it from its own checkout,
#     or clone it on the cluster, and point NEGSTEER_REPO_DIR at it. Syncing a
#     submodule's working tree from here would make the cluster's engine a
#     copy with no commit of its own, which is the opposite of pinning it.
#   - .claude/  Local editor/agent config, Mac-side only.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Host and remote path come from site.conf, which is gitignored. HOST is
# anything ssh accepts, including an alias from ~/.ssh/config.
# The site.conf loader lives in the engine, so there is one copy of the
# precedence rule rather than two. NEGSTEER_SITE_CONF, which it honours, is
# set to this repository's own file.
ENGINE_DIR="${NEGSTEER_REPO_DIR:-${REPO_ROOT}/engine}"
LOADER="${ENGINE_DIR}/scripts/site_conf.sh"
if [ ! -f "${LOADER}" ]; then
    echo "ERROR: no site.conf loader at ${LOADER}." >&2
    echo "Run: git submodule update --init, or set NEGSTEER_REPO_DIR." >&2
    exit 1
fi
export NEGSTEER_SITE_CONF="${NEGSTEER_SITE_CONF:-${REPO_ROOT}/site.conf}"
# shellcheck source=/dev/null
. "${LOADER}"

if [ -z "${NEGSTEER_HPC_HOST:-}" ] || [ -z "${NEGSTEER_HPC_DEST:-}" ]; then
    echo "ERROR: set NEGSTEER_HPC_HOST and NEGSTEER_HPC_DEST in site.conf." >&2
    echo "See site.conf.example." >&2
    exit 1
fi

HPC_USER_HOST="${NEGSTEER_HPC_HOST}"
HPC_DEST="${NEGSTEER_HPC_DEST}"

DRY_RUN=""
case "${1:-}" in
    --dry|--dry-run)
        DRY_RUN="--dry-run"
        echo "=== DRY RUN -- no files will be changed ==="
        ;;
    --list-stale)
        cd "$REPO_ROOT"
        echo "=== cluster paths with no local counterpart ==="
        echo "Reads only. Nothing here is deleted, by this script or otherwise."
        echo

        # A benchmark target is a directory holding a config.yml. Comparing on
        # that rather than on directory names keeps derived inputs/ and run/
        # trees out of the report.
        find_targets="find . -name config.yml -maxdepth 3 \
                      | sed -e 's|^\./||' -e 's|/config.yml\$||' | sort"

        local_targets=$(cd experiments/benchmarking 2>/dev/null \
            && eval "$find_targets")
        remote_targets=$(ssh "$HPC_USER_HOST" \
            "cd ${HPC_DEST}experiments/benchmarking 2>/dev/null && $find_targets" 2>/dev/null)

        echo "--- benchmarking targets on the cluster only ---"
        stale=$(comm -23 <(echo "$remote_targets") <(echo "$local_targets"))
        [ -n "$stale" ] && echo "$stale" | sed 's/^/  /' || echo "  (none)"

        echo
        echo "--- repo-root entries on the cluster only ---"
        local_root=$(ls -A | sort)
        remote_root=$(ssh "$HPC_USER_HOST" "cd ${HPC_DEST} && ls -A" 2>/dev/null | sort)
        comm -23 <(echo "$remote_root") <(echo "$local_root") | sed 's/^/  /' || true

        echo
        echo "Nothing above has been touched. To retire one, rename rather than"
        echo "delete, so the name is freed but the data survives:"
        echo "  ssh $HPC_USER_HOST 'cd ${HPC_DEST} && mv <path> <path>_superseded_\$(date +%Y%m%d)'"
        exit 0
        ;;
esac

cd "$REPO_ROOT"

rsync -av --delete $DRY_RUN -e ssh \
    --exclude='.git/' \
    --exclude='experiments/' \
    --exclude='analyses/' \
    --exclude='engine/' \
    --exclude='.claude/' \
    --exclude='.vscode/' \
    --exclude='.idea/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='~$*' \
    --exclude='*.img' \
    --exclude='*.sif' \
    --exclude='work/' \
    --exclude='.nextflow*' \
    --exclude='jdk-*/' \
    --exclude='nxf_home/' \
    --exclude='*_results/' \
    --exclude='*.out' \
    --exclude='*.err' \
    --exclude='rsync_dryrun*.txt' \
    --exclude='rsync_deletions*.txt' \
    "$REPO_ROOT/" "${HPC_USER_HOST}:${HPC_DEST}"

# Push experiment DEFINITIONS additively (NO --delete): every tree's config.yml
# must reach the HPC, but the HPC experiments/ tree (prior results + run
# outputs) is authoritative and must never be pruned by a code sync. Derived
# inputs/ and run/ outputs are excluded.
#
# No PDB is pushed from here. The eighteen reference complexes live in data/
# and ride along with the first pass above, which is the only copy every config
# in every tree resolves to.
if [ -d "$REPO_ROOT/experiments/benchmarking" ]; then
    rsync -av --no-perms --no-owner --no-group $DRY_RUN -e ssh \
        --prune-empty-dirs \
        --include='benchmarking/' \
        --include='benchmarking/**/' \
        --include='benchmarking/**/config.yml' \
        --include='dose-sweep/' \
        --include='dose-sweep/**/' \
        --include='dose-sweep/**/config.yml' \
        --include='interface-ablation/' \
        --include='interface-ablation/**/' \
        --include='interface-ablation/**/config.yml' \
        --exclude='*' \
        "$REPO_ROOT/experiments/" "${HPC_USER_HOST}:${HPC_DEST}experiments/"
fi

echo
echo "Sync complete -> ${HPC_USER_HOST}:${HPC_DEST}"
echo "(repo pushed with --delete; experiments/ results untouched —"
echo " only benchmarking/ test definitions are pushed, additively.)"
