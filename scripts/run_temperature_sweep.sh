#!/usr/bin/env bash
#
# Temperature sweep for the LLM-GP construction-heuristic experiment.
#
# Research goal: measure how the Ollama/Qwen sampling *temperature* affects
#   (a) candidate diversity, and
#   (b) downstream heuristic / timetable performance.
#
# Temperature is the ONLY variable this script changes. Every other parameter
# (instance, GP profile, population size, generations, tournament size, initial
# batch size, crossover/mutation rate, retry limit, seed, model) is held fixed
# and identical across all runs.
#
# Each temperature gets its own results subtree:
#
#   results/temperature/T_<temp>/
#     sweep_run.json                     <- sweep metadata (this script)
#     console.log                        <- full stdout/stderr of the run
#     <instance>-<UTCstamp>-seed<seed>/  <- the experiment's own output
#       run.json summary.json best_heuristic.json
#       candidates.jsonl llm_calls.jsonl generations.csv
#
# Existing results are never modified or deleted. If a run fails, the whole
# sweep aborts immediately with a non-zero exit code.

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration (override via environment variables; do NOT edit per-run)
# --------------------------------------------------------------------------

# Temperatures to test, in order.
read -r -a TEMPERATURES <<< "${TEMPERATURES:-0.4 0.6 0.8 1.0}"

# Fixed experiment configuration. Matches the README preliminary configuration.
INSTANCE="${INSTANCE:-car-f-92}"
PERIODS="${PERIODS:-32}"
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

# RUN_MODE=docker  -> docker compose run --rm gp   (default; matches README)
# RUN_MODE=local   -> python -m llm_gp_hh.experiments.run   (needs local install)
RUN_MODE="${RUN_MODE:-docker}"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${RUN_MODE}" == "docker" ]]; then
  DATA_PREFIX="/app/data/toronto"
  RESULTS_PREFIX="/app/results/temperature"
else
  DATA_PREFIX="data/toronto"
  RESULTS_PREFIX="results/temperature"
fi

HOST_SWEEP_ROOT="${REPO_ROOT}/results/temperature"
mkdir -p "${HOST_SWEEP_ROOT}"

# The Docker image bakes src/ in at build time, so a stale image will not have
# the --temperature CLI argument. Rebuild once before the sweep (set
# SKIP_BUILD=1 to skip, e.g. when you have just built manually or RUN_MODE=local).
if [[ "${RUN_MODE}" == "docker" && "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "Building gp image (set SKIP_BUILD=1 to skip)..."
  docker compose build gp
fi

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git diff --quiet 2>/dev/null && echo false || echo true)"
OLLAMA_HOST_VALUE="${OLLAMA_HOST:-http://127.0.0.1:11434}"

echo "==================================================================="
echo "Temperature sweep"
echo "  temperatures : ${TEMPERATURES[*]}"
echo "  instance     : ${INSTANCE} (periods ${PERIODS}, profile ${PROFILE})"
echo "  model        : ${MODEL}"
echo "  seed         : ${SEED}"
echo "  run mode     : ${RUN_MODE}"
echo "  git commit   : ${GIT_COMMIT} (dirty=${GIT_DIRTY})"
echo "  output root  : ${HOST_SWEEP_ROOT}"
echo "==================================================================="

# --------------------------------------------------------------------------
# Run one experiment per temperature
# --------------------------------------------------------------------------

for TEMP in "${TEMPERATURES[@]}"; do
  TAG="T_${TEMP}"
  HOST_OUT_DIR="${HOST_SWEEP_ROOT}/${TAG}"
  CONTAINER_OUT_DIR="${RESULTS_PREFIX}/${TAG}"
  mkdir -p "${HOST_OUT_DIR}"

  STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  # Fixed argument list. Only --temperature and --results-dir vary between runs.
  ARGS=(
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
    --temperature "${TEMP}"
    --results-dir "${CONTAINER_OUT_DIR}"
  )

  if [[ "${RUN_MODE}" == "docker" ]]; then
    CMD=(docker compose run --rm gp "${ARGS[@]}")
  else
    CMD=(python -m llm_gp_hh.experiments.run "${ARGS[@]}")
  fi

  echo
  echo "-------------------------------------------------------------------"
  echo "[${TAG}] temperature=${TEMP}  started=${STARTED_UTC}"
  echo "[${TAG}] ${CMD[*]}"
  echo "-------------------------------------------------------------------"

  # Record metadata BEFORE the run so a failure still leaves a breadcrumb.
  cat > "${HOST_OUT_DIR}/sweep_run.json" <<JSON
{
  "temperature": ${TEMP},
  "model": "${MODEL}",
  "instance": "${INSTANCE}",
  "periods": ${PERIODS},
  "profile": "${PROFILE}",
  "population_size": ${POPULATION_SIZE},
  "generations": ${GENERATIONS},
  "tournament_size": ${TOURNAMENT_SIZE},
  "initial_batch_size": ${INITIAL_BATCH_SIZE},
  "crossover_rate": ${CROSSOVER_RATE},
  "mutation_rate": ${MUTATION_RATE},
  "retry_limit": ${RETRY_LIMIT},
  "seed": ${SEED},
  "run_mode": "${RUN_MODE}",
  "ollama_host": "${OLLAMA_HOST_VALUE}",
  "git_commit": "${GIT_COMMIT}",
  "git_dirty": ${GIT_DIRTY},
  "started_utc": "${STARTED_UTC}",
  "finished_utc": null,
  "status": "running",
  "command": "${CMD[*]}"
}
JSON

  set +e
  "${CMD[@]}" 2>&1 | tee "${HOST_OUT_DIR}/console.log"
  STATUS=${PIPESTATUS[0]}
  set -e

  FINISHED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ ${STATUS} -ne 0 ]]; then
    sed -i "s/\"status\": \"running\"/\"status\": \"failed\"/" "${HOST_OUT_DIR}/sweep_run.json"
    sed -i "s/\"finished_utc\": null/\"finished_utc\": \"${FINISHED_UTC}\"/" "${HOST_OUT_DIR}/sweep_run.json"
    echo
    echo "ERROR: experiment for ${TAG} (temperature=${TEMP}) failed with exit code ${STATUS}." >&2
    echo "       See ${HOST_OUT_DIR}/console.log . Aborting sweep." >&2
    exit "${STATUS}"
  fi

  # Identify the run directory the experiment just created (newest under T_x).
  RUN_DIR="$(find "${HOST_OUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name "${INSTANCE}-*" -printf '%T@ %p\n' \
    | sort -nr | head -n1 | cut -d' ' -f2-)"
  RUN_DIR_REL="${RUN_DIR#${REPO_ROOT}/}"

  sed -i "s/\"status\": \"running\"/\"status\": \"ok\"/" "${HOST_OUT_DIR}/sweep_run.json"
  sed -i "s/\"finished_utc\": null/\"finished_utc\": \"${FINISHED_UTC}\"/" "${HOST_OUT_DIR}/sweep_run.json"
  python3 - "$HOST_OUT_DIR/sweep_run.json" "$RUN_DIR_REL" <<'PY' 2>/dev/null || true
import json, sys
path, run_dir = sys.argv[1], sys.argv[2]
data = json.load(open(path))
data["experiment_run_dir"] = run_dir
json.dump(data, open(path, "w"), indent=2)
open(path, "a").write("\n")
PY

  echo "[${TAG}] done  finished=${FINISHED_UTC}  -> ${RUN_DIR_REL:-<run dir not found>}"
done

echo
echo "==================================================================="
echo "Sweep complete. Results:"
for TEMP in "${TEMPERATURES[@]}"; do
  echo "  results/temperature/T_${TEMP}/"
done
echo "==================================================================="
