#!/usr/bin/env bash
#
# Full Toronto benchmark sweep for the LLM-GP construction-heuristic experiment.
#
# Runs every instance in the Toronto examination-timetabling benchmark set,
# ordered from the SMALLEST to the LARGEST instance (by number of exams), so
# that cheap runs finish and can be inspected before the expensive ones start.
#
# REPRODUCIBILITY: every experiment runs inside the project Docker container
# (docker compose run --rm gp). There is deliberately no local-execution mode -
# a run on the host would not be comparable with a teammate's containerised run.
# Before the sweep the script writes results/benchmark/environment.json: the
# container image ID, Dockerfile hash, git commit, in-container Python and
# package versions, and the Ollama model digest. Diff that file against your
# teammate's to confirm you are actually running the same thing.
#
# The LLM sampling temperature is left at the built-in default (see
# RunConfig.temperature in src/llm_gp_hh/config.py): --temperature is NOT passed
# and $LLM_TEMPERATURE is cleared, so no temperature override can leak in from
# the shell. Every other parameter matches the README preliminary configuration
# and is held identical across all instances; only the instance, its period
# count, and the output directory vary.
#
# Output layout (existing results are never modified or deleted):
#
#   results/benchmark/
#     environment.json                   <- provenance fingerprint (this script)
#     <NN>_<instance>/
#       benchmark_run.json               <- per-run metadata (this script)
#       console.log                      <- full stdout/stderr of the run
#       <instance>-<UTCstamp>-seed<seed>/  <- the experiment's own output
#         run.json summary.json best_heuristic.json
#         candidates.jsonl llm_calls.jsonl generations.csv
#
# Usage:
#   ./scripts/run_toronto_benchmark.sh
#   ONLY="hec-s-92 sta-f-83" ./scripts/run_toronto_benchmark.sh
#   START_AT=kfu-s-93 SKIP_EXISTING=1 ./scripts/run_toronto_benchmark.sh
#
# By default a failing instance does NOT abort the sweep (CONTINUE_ON_ERROR=1);
# the remaining instances still run and the script exits non-zero at the end
# with a list of the failures. Set CONTINUE_ON_ERROR=0 to abort on first error.

set -euo pipefail

# --------------------------------------------------------------------------
# Instance list: "<instance> <periods>", smallest -> largest.
# Sizes (exams / students) are from data/toronto; periods are from the README.
# --------------------------------------------------------------------------
INSTANCES=(
  "hec-s-92 18"   #   81 exams,  2823 students
  "sta-f-83 13"   #  139 exams,   611 students
  "yor-f-83 21"   #  181 exams,   941 students
  "ute-s-92 10"   #  184 exams,  2750 students
  "ear-f-83 24"   #  190 exams,  1125 students
  "tre-s-92 23"   #  261 exams,  4362 students
  "lse-f-91 18"   #  381 exams,  2726 students
  "kfu-s-93 20"   #  461 exams,  5349 students
  "rye-s-93 23"   #  486 exams, 11483 students
  "car-f-92 32"   #  543 exams, 18419 students
  "uta-s-92 35"   #  622 exams, 21266 students
  "car-s-91 35"   #  682 exams, 16926 students
  "pur-s-93 43"   # 2419 exams, 30032 students  (by far the largest)
)

# --------------------------------------------------------------------------
# Fixed experiment configuration. Matches the README preliminary configuration.
# These MUST match your teammate's values for the comparison to mean anything.
# --------------------------------------------------------------------------
PROFILE="${PROFILE:-dev}"
POPULATION_SIZE="${POPULATION_SIZE:-20}"
GENERATIONS="${GENERATIONS:-10}"
TOURNAMENT_SIZE="${TOURNAMENT_SIZE:-4}"
INITIAL_BATCH_SIZE="${INITIAL_BATCH_SIZE:-4}"
CROSSOVER_RATE="${CROSSOVER_RATE:-0.8}"
MUTATION_RATE="${MUTATION_RATE:-0.2}"
RETRY_LIMIT="${RETRY_LIMIT:-2}"
SEED="${SEED:-1001}"
MODEL="${MODEL:-qwen3-coder:30b}"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"   # 1 -> skip instances already marked "ok"
ONLY="${ONLY:-}"                      # space-separated instance names to run
START_AT="${START_AT:-}"              # resume the ordered list at this instance

# The default temperature must come from RunConfig, not from the environment.
unset LLM_TEMPERATURE

# --------------------------------------------------------------------------
# Paths. Data and results paths are container-side; the compose file bind-mounts
# ./data -> /app/data (ro) and ./results -> /app/results.
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_PREFIX="/app/data/toronto"
RESULTS_PREFIX="/app/results/benchmark"
HOST_SWEEP_ROOT="${REPO_ROOT}/results/benchmark"

# --------------------------------------------------------------------------
# Preflight: fail early and loudly rather than half way through the sweep.
# --------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not on PATH. This sweep runs only in" >&2
  echo "       the project container; see README setup." >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: 'docker compose' is unavailable (needs Compose v2)." >&2
  exit 1
fi
if ! docker compose config --services 2>/dev/null | grep -qx gp; then
  echo "ERROR: no 'gp' service in docker-compose.yml (run from the repo root)." >&2
  exit 1
fi

mkdir -p "${HOST_SWEEP_ROOT}"

# The Docker image bakes src/ in at build time, so a stale image would run old
# code and silently break comparability. Rebuild before the sweep. SKIP_BUILD=1
# is for when you have just built manually - never to dodge a stale image.
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "Building gp image (set SKIP_BUILD=1 to skip)..."
  docker compose build gp
else
  echo "SKIP_BUILD=1: using the existing gp image without rebuilding."
fi

# --------------------------------------------------------------------------
# Provenance fingerprint - the thing to diff against a teammate's run
# --------------------------------------------------------------------------
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git diff --quiet 2>/dev/null && echo false || echo true)"
OLLAMA_HOST_VALUE="${OLLAMA_HOST:-http://127.0.0.1:11434}"
SWEEP_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

DOCKER_VERSION="$(docker --version 2>/dev/null || echo unknown)"
COMPOSE_VERSION="$(docker compose version --short 2>/dev/null || echo unknown)"
IMAGE_NAME="$(docker compose config --images gp 2>/dev/null | head -n1 || echo unknown)"
IMAGE_ID="$(docker image inspect "${IMAGE_NAME}" --format '{{.Id}}' 2>/dev/null || echo unknown)"
IMAGE_CREATED="$(docker image inspect "${IMAGE_NAME}" --format '{{.Created}}' 2>/dev/null || echo unknown)"
DOCKERFILE_SHA256="$(sha256sum Dockerfile 2>/dev/null | cut -d' ' -f1 || echo unknown)"
COMPOSEFILE_SHA256="$(sha256sum docker-compose.yml 2>/dev/null | cut -d' ' -f1 || echo unknown)"

# Python / package versions as they actually exist inside the image.
CONTAINER_PYTHON="$(docker compose run --rm --entrypoint python gp -c \
  'import sys; print(sys.version.split()[0])' 2>/dev/null | tr -d '\r' | tail -n1 || echo unknown)"
CONTAINER_PKGS="$(docker compose run --rm --entrypoint python gp -c \
  'import importlib.metadata as m; print(",".join(sorted(f"{d.name}=={d.version}" for d in m.distributions())))' \
  2>/dev/null | tr -d '\r' | tail -n1 || echo unknown)"

# Ollama model digest: a different model blob means different results even with
# an identical container, so it belongs in the fingerprint.
MODEL_DIGEST="$(curl -s --max-time 5 "${OLLAMA_HOST_VALUE}/api/tags" 2>/dev/null \
  | python3 -c 'import json, sys
want = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    print("unknown"); raise SystemExit
for entry in data.get("models", []):
    if want in (entry.get("model"), entry.get("name")):
        print(entry.get("digest", "unknown")); break
else:
    print("not-installed")' "${MODEL}" 2>/dev/null || echo unknown)"

# Host hardware. Ollama runs on the HOST (the container uses network_mode: host
# to reach it), so inference determinism depends on this machine's CPU/GPU, not
# on the image. Two teammates with identical containers but different GPUs can
# legitimately get different LLM output.
HOST_KERNEL="$(uname -sr 2>/dev/null || echo unknown)"
HOST_CPU="$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || echo unknown)"
HOST_CPU="${HOST_CPU:-unknown}"
HOST_GPU="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | paste -sd'; ' - || true)"
HOST_GPU="${HOST_GPU:-none-detected}"

if [[ "${GIT_DIRTY}" == "true" ]]; then
  echo
  echo "WARNING: the working tree has uncommitted changes (git_dirty=true)."
  echo "         The image was built from this dirty tree, so your teammate"
  echo "         cannot reproduce it from commit ${GIT_COMMIT} alone."
  echo
fi
if [[ "${MODEL_DIGEST}" == "not-installed" ]]; then
  echo "ERROR: model '${MODEL}' is not present on the Ollama server at" >&2
  echo "       ${OLLAMA_HOST_VALUE}. Run: ollama pull ${MODEL}" >&2
  exit 1
fi

python3 - "${HOST_SWEEP_ROOT}/environment.json" <<PY
import json, sys
fingerprint = {
    "recorded_utc": "${SWEEP_STARTED_UTC}",
    "execution": "docker (docker compose run --rm gp)",
    "docker_version": "${DOCKER_VERSION}",
    "compose_version": "${COMPOSE_VERSION}",
    "image_name": "${IMAGE_NAME}",
    "image_id": "${IMAGE_ID}",
    "image_created": "${IMAGE_CREATED}",
    "dockerfile_sha256": "${DOCKERFILE_SHA256}",
    "compose_file_sha256": "${COMPOSEFILE_SHA256}",
    "container_python": "${CONTAINER_PYTHON}",
    "container_packages": "${CONTAINER_PKGS}".split(",") if "${CONTAINER_PKGS}" != "unknown" else "unknown",
    "git_commit": "${GIT_COMMIT}",
    "git_dirty": "${GIT_DIRTY}" == "true",
    "host_kernel": "${HOST_KERNEL}",
    "host_cpu": "${HOST_CPU}",
    "host_gpu": "${HOST_GPU}",
    "ollama_host": "${OLLAMA_HOST_VALUE}",
    "model": "${MODEL}",
    "model_digest": "${MODEL_DIGEST}",
    "temperature": "default (RunConfig.temperature; not overridden)",
    "parameters": {
        "profile": "${PROFILE}",
        "population_size": ${POPULATION_SIZE},
        "generations": ${GENERATIONS},
        "tournament_size": ${TOURNAMENT_SIZE},
        "initial_batch_size": ${INITIAL_BATCH_SIZE},
        "crossover_rate": ${CROSSOVER_RATE},
        "mutation_rate": ${MUTATION_RATE},
        "retry_limit": ${RETRY_LIMIT},
        "seed": ${SEED},
    },
    "instance_order": [entry.split()[0] for entry in """$(printf '%s\n' "${INSTANCES[@]}")""".strip().splitlines()],
}
with open(sys.argv[1], "w") as handle:
    json.dump(fingerprint, handle, indent=2)
    handle.write("\n")
PY

echo "==================================================================="
echo "Toronto benchmark sweep (smallest -> largest)"
echo "  instances    : ${#INSTANCES[@]} (default temperature, not overridden)"
echo "  profile      : ${PROFILE}  pop=${POPULATION_SIZE} gens=${GENERATIONS}"
echo "  model        : ${MODEL}"
echo "  model digest : ${MODEL_DIGEST}"
echo "  seed         : ${SEED}"
echo "  execution    : docker  image=${IMAGE_NAME}"
echo "  image id     : ${IMAGE_ID}"
echo "  git commit   : ${GIT_COMMIT} (dirty=${GIT_DIRTY})"
echo "  host gpu     : ${HOST_GPU}"
echo "  output root  : ${HOST_SWEEP_ROOT}"
echo "  fingerprint  : results/benchmark/environment.json"
echo "  started      : ${SWEEP_STARTED_UTC}"
echo "==================================================================="

# --------------------------------------------------------------------------
# Run one experiment per instance
# --------------------------------------------------------------------------
COMPLETED=()
FAILED=()
SKIPPED=()
INDEX=0
REACHED_START=0
[[ -z "${START_AT}" ]] && REACHED_START=1

for ENTRY in "${INSTANCES[@]}"; do
  read -r INSTANCE PERIODS <<< "${ENTRY}"
  INDEX=$((INDEX + 1))
  TAG="$(printf '%02d_%s' "${INDEX}" "${INSTANCE}")"
  HOST_OUT_DIR="${HOST_SWEEP_ROOT}/${TAG}"
  CONTAINER_OUT_DIR="${RESULTS_PREFIX}/${TAG}"

  if [[ -n "${START_AT}" && "${REACHED_START}" -eq 0 ]]; then
    if [[ "${INSTANCE}" == "${START_AT}" ]]; then
      REACHED_START=1
    else
      SKIPPED+=("${INSTANCE} (before START_AT)")
      continue
    fi
  fi

  if [[ -n "${ONLY}" && " ${ONLY} " != *" ${INSTANCE} "* ]]; then
    SKIPPED+=("${INSTANCE} (not in ONLY)")
    continue
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${HOST_OUT_DIR}/benchmark_run.json" ]] \
     && grep -q '"status": "ok"' "${HOST_OUT_DIR}/benchmark_run.json"; then
    echo "[${TAG}] already completed - skipping (SKIP_EXISTING=1)"
    SKIPPED+=("${INSTANCE} (already ok)")
    continue
  fi

  if [[ ! -f "data/toronto/${INSTANCE}.crs" || ! -f "data/toronto/${INSTANCE}.stu" ]]; then
    echo "ERROR: missing data files for ${INSTANCE} under data/toronto/." >&2
    FAILED+=("${INSTANCE} (missing data)")
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] && continue || exit 1
  fi

  mkdir -p "${HOST_OUT_DIR}"
  STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  STARTED_EPOCH="$(date +%s)"

  # --temperature is deliberately omitted so the built-in default is used.
  CMD=(
    docker compose run --rm gp
    --crs "${DATA_PREFIX}/${INSTANCE}.crs"
    --stu "${DATA_PREFIX}/${INSTANCE}.stu"
    --periods "${PERIODS}"
    --profile "${PROFILE}"
    --population-size "${POPULATION_SIZE}"
    --generations "${GENERATIONS}"
    --tournament-size "${TOURNAMENT_SIZE}"
    --initial-batch-size "${INITIAL_BATCH_SIZE}"
    --crossover-rate "${CROSSOVER_RATE}"
    --mutation-rate "${MUTATION_RATE}"
    --retry-limit "${RETRY_LIMIT}"
    --seed "${SEED}"
    --model "${MODEL}"
    --results-dir "${CONTAINER_OUT_DIR}"
  )

  echo
  echo "-------------------------------------------------------------------"
  echo "[${TAG}] instance=${INSTANCE} periods=${PERIODS}  (${INDEX}/${#INSTANCES[@]})"
  echo "[${TAG}] started=${STARTED_UTC}"
  echo "[${TAG}] ${CMD[*]}"
  echo "-------------------------------------------------------------------"

  # Record metadata BEFORE the run so a failure still leaves a breadcrumb.
  cat > "${HOST_OUT_DIR}/benchmark_run.json" <<JSON
{
  "instance": "${INSTANCE}",
  "periods": ${PERIODS},
  "order_index": ${INDEX},
  "temperature": "default (RunConfig.temperature; not overridden)",
  "model": "${MODEL}",
  "model_digest": "${MODEL_DIGEST}",
  "profile": "${PROFILE}",
  "population_size": ${POPULATION_SIZE},
  "generations": ${GENERATIONS},
  "tournament_size": ${TOURNAMENT_SIZE},
  "initial_batch_size": ${INITIAL_BATCH_SIZE},
  "crossover_rate": ${CROSSOVER_RATE},
  "mutation_rate": ${MUTATION_RATE},
  "retry_limit": ${RETRY_LIMIT},
  "seed": ${SEED},
  "execution": "docker",
  "image_name": "${IMAGE_NAME}",
  "image_id": "${IMAGE_ID}",
  "ollama_host": "${OLLAMA_HOST_VALUE}",
  "git_commit": "${GIT_COMMIT}",
  "git_dirty": ${GIT_DIRTY},
  "started_utc": "${STARTED_UTC}",
  "finished_utc": null,
  "wall_seconds": null,
  "status": "running",
  "command": "${CMD[*]}"
}
JSON

  set +e
  "${CMD[@]}" 2>&1 | tee "${HOST_OUT_DIR}/console.log"
  STATUS=${PIPESTATUS[0]}
  set -e

  FINISHED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  WALL_SECONDS=$(( $(date +%s) - STARTED_EPOCH ))

  sed -i "s/\"finished_utc\": null/\"finished_utc\": \"${FINISHED_UTC}\"/" "${HOST_OUT_DIR}/benchmark_run.json"
  sed -i "s/\"wall_seconds\": null/\"wall_seconds\": ${WALL_SECONDS}/" "${HOST_OUT_DIR}/benchmark_run.json"

  if [[ ${STATUS} -ne 0 ]]; then
    sed -i "s/\"status\": \"running\"/\"status\": \"failed\"/" "${HOST_OUT_DIR}/benchmark_run.json"
    echo
    echo "ERROR: ${INSTANCE} failed with exit code ${STATUS}." >&2
    echo "       See ${HOST_OUT_DIR}/console.log ." >&2
    FAILED+=("${INSTANCE} (exit ${STATUS})")
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      echo "       CONTINUE_ON_ERROR=0 - aborting sweep." >&2
      exit "${STATUS}"
    fi
    echo "       Continuing with the next instance." >&2
    continue
  fi

  # Identify the run directory the experiment just created (newest under TAG).
  RUN_DIR="$(find "${HOST_OUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name "${INSTANCE}-*" -printf '%T@ %p\n' \
    | sort -nr | head -n1 | cut -d' ' -f2-)"
  RUN_DIR_REL="${RUN_DIR#${REPO_ROOT}/}"

  sed -i "s/\"status\": \"running\"/\"status\": \"ok\"/" "${HOST_OUT_DIR}/benchmark_run.json"
  python3 - "$HOST_OUT_DIR/benchmark_run.json" "$RUN_DIR_REL" <<'PY' 2>/dev/null || true
import json, sys
path, run_dir = sys.argv[1], sys.argv[2]
data = json.load(open(path))
data["experiment_run_dir"] = run_dir
json.dump(data, open(path, "w"), indent=2)
open(path, "a").write("\n")
PY

  COMPLETED+=("${INSTANCE} (${WALL_SECONDS}s)")
  echo "[${TAG}] done  finished=${FINISHED_UTC}  wall=${WALL_SECONDS}s  -> ${RUN_DIR_REL:-<run dir not found>}"
done

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo
echo "==================================================================="
echo "Toronto benchmark sweep finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  started     : ${SWEEP_STARTED_UTC}"
echo "  results     : results/benchmark/"
echo "  fingerprint : results/benchmark/environment.json"
echo "                (diff this against your teammate's before comparing runs)"
echo
echo "Completed (${#COMPLETED[@]}):"
if [[ ${#COMPLETED[@]} -eq 0 ]]; then echo "  (none)"; else printf '  %s\n' "${COMPLETED[@]}"; fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo
  echo "Skipped (${#SKIPPED[@]}):"
  printf '  %s\n' "${SKIPPED[@]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo
  echo "Failed (${#FAILED[@]}):"
  printf '  %s\n' "${FAILED[@]}"
  echo "==================================================================="
  exit 1
fi
echo "==================================================================="
