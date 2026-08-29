#!/usr/bin/env bash
# Run the same experiment configuration across several random seeds.
#
# One run is not enough for scientific conclusions (see README), so this
# repeats the configured experiment for each seed and writes every run into
# its own directory under results/.
#
# Usage:
#   scripts/run_suite.sh [INSTANCE] [PERIODS] [SEED ...]
#
# Defaults reproduce the README "first full experiment" on car-f-92 with
# seeds 1001-1005.
set -euo pipefail

INSTANCE="${1:-car-f-92}"
PERIODS="${2:-32}"
shift || true
shift || true
SEEDS=("$@")
if [ "${#SEEDS[@]}" -eq 0 ]; then
  SEEDS=(1001 1002 1003 1004 1005)
fi

DATA_DIR="${DATA_DIR:-data/toronto}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# Experiment configuration (override via environment if needed).
PROFILE="${PROFILE:-dev}"
POPULATION_SIZE="${POPULATION_SIZE:-20}"
GENERATIONS="${GENERATIONS:-10}"
TOURNAMENT_SIZE="${TOURNAMENT_SIZE:-4}"
INITIAL_BATCH_SIZE="${INITIAL_BATCH_SIZE:-4}"
CROSSOVER_RATE="${CROSSOVER_RATE:-0.8}"
MUTATION_RATE="${MUTATION_RATE:-0.2}"
RETRY_LIMIT="${RETRY_LIMIT:-2}"
MODEL="${MODEL:-qwen3-coder:30b}"

echo "Suite: instance=${INSTANCE} periods=${PERIODS} seeds=${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  echo "======================================================================"
  echo "  seed ${seed}"
  echo "======================================================================"
  python -m llm_gp_hh.experiments.run \
    --crs "${DATA_DIR}/${INSTANCE}.crs" \
    --stu "${DATA_DIR}/${INSTANCE}.stu" \
    --periods "${PERIODS}" \
    --profile "${PROFILE}" \
    --population-size "${POPULATION_SIZE}" \
    --generations "${GENERATIONS}" \
    --tournament-size "${TOURNAMENT_SIZE}" \
    --initial-batch-size "${INITIAL_BATCH_SIZE}" \
    --crossover-rate "${CROSSOVER_RATE}" \
    --mutation-rate "${MUTATION_RATE}" \
    --retry-limit "${RETRY_LIMIT}" \
    --model "${MODEL}" \
    --results-dir "${RESULTS_DIR}" \
    --seed "${seed}"
done

echo "Suite complete. Results written under ${RESULTS_DIR}/"
