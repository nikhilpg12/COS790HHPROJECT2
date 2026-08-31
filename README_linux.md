# LLM-GP Hyper-Heuristic for Educational Timetabling

## Project Overview

This project investigates whether a **Large Language Model (LLM)** can replace the generative components of a vanilla Genetic Programming (GP) generation construction hyper-heuristic for educational timetabling.

The system uses a standard GP evolutionary framework while using the LLM for:

- Initial GP population generation
- GP crossover
- GP mutation

The conventional GP framework is retained for:

- Population management
- Tournament selection
- GP tree representation
- Timetable construction
- Fitness evaluation
- Termination
- Experiment logging

The LLM does **not** generate timetables directly. It generates GP expression trees that act as low-level construction heuristics.

## Research Question

> Can replacing the generative components of a vanilla genetic-programming generation construction hyper-heuristic with a large language model produce effective low-level construction heuristics for educational timetabling?

## Technology

- Python 3.11
- Docker
- Ollama
- Qwen3-Coder 30B
- Vanilla Genetic Programming
- Toronto Examination Timetabling Benchmarks

---

# Setup

## 1. Install Ollama

Install Ollama on Windows.

Pull the Qwen model:

```powershell
ollama pull qwen3-coder:30b
```

Confirm that the model is installed:

```powershell
ollama list
```

## 2. Allow Docker to Access Ollama

Run:

```powershell
setx OLLAMA_HOST "0.0.0.0:11434"
```

Restart Ollama after setting the environment variable.

## 3. Build the Docker Image

From the project directory:

```powershell
docker compose build --no-cache
```

---

# Preliminary Experiment Configuration

The preliminary Toronto experiments use:

| Parameter | Value |
|---|---:|
| Population size | 20 |
| Generations | 10 |
| Candidate evaluations | 200 |
| Tournament size | 4 |
| Initial LLM batch size | 4 |
| Crossover rate | 0.8 |
| Mutation rate | 0.2 |
| Retry limit | 2 |
| Seed | 1001 |
| Model | qwen3-coder:30b |

Results are written to:

```text
results/
```

---

# Toronto Benchmark Experiments

## car-f-92

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/car-f-92.crs \
  --stu /app/data/toronto/car-f-92.stu \
  --periods 32 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## car-s-91

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/car-s-91.crs \
  --stu /app/data/toronto/car-s-91.stu \
  --periods 35 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## ear-f-83

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/ear-f-83.crs \
  --stu /app/data/toronto/ear-f-83.stu \
  --periods 24 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## hec-s-92

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/hec-s-92.crs \
  --stu /app/data/toronto/hec-s-92.stu \
  --periods 18 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## kfu-s-93

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/kfu-s-93.crs \
  --stu /app/data/toronto/kfu-s-93.stu \
  --periods 20 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## lse-f-91

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/lse-f-91.crs \
  --stu /app/data/toronto/lse-f-91.stu \
  --periods 18 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## pur-s-93

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/pur-s-93.crs \
  --stu /app/data/toronto/pur-s-93.stu \
  --periods 43 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## rye-s-93

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/rye-s-93.crs \
  --stu /app/data/toronto/rye-s-93.stu \
  --periods 23 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## sta-f-83

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/sta-f-83.crs \
  --stu /app/data/toronto/sta-f-83.stu \
  --periods 13 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## tre-s-92

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/tre-s-92.crs \
  --stu /app/data/toronto/tre-s-92.stu \
  --periods 23 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## uta-s-92

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/uta-s-92.crs \
  --stu /app/data/toronto/uta-s-92.stu \
  --periods 35 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## ute-s-92

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/ute-s-92.crs \
  --stu /app/data/toronto/ute-s-92.stu \
  --periods 10 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

## yor-f-83

```bash
docker compose run --rm gp \
  --crs /app/data/toronto/yor-f-83.crs \
  --stu /app/data/toronto/yor-f-83.stu \
  --periods 21 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001 \
  --results-dir /app/results
```

---

# Temperature Sweep Experiment

This experiment investigates whether the Ollama/Qwen **sampling temperature**
affects (a) candidate diversity and (b) downstream heuristic / timetable
performance. Temperature is the **only** variable that changes; every other
parameter is held fixed and identical across all runs.

## How temperature is configured

Precedence (first match wins):

1. `--temperature <float>` CLI argument to `llm_gp_hh.experiments.run`
2. `LLM_TEMPERATURE` environment variable
3. Built-in default (`0.4`) — unchanged from the previous behaviour

The value is stored on `RunConfig.temperature`, passed to `OllamaClient`, and
forwarded to `ollama.chat(..., options={"seed": ..., "temperature": ...})`. It
is recorded in every run's `run.json`. The per-call Ollama `seed` is unchanged.

## How to run the sweep

```bash
./scripts/run_temperature_sweep.sh
```

`RUN_MODE=docker` by default (matches the benchmark commands above);
`RUN_MODE=local` uses a local install. The script runs `docker compose build gp`
once first (the image bakes in `src/`); set `SKIP_BUILD=1` to skip. Fixed parameters are overridable via
environment variables (`TEMPERATURES`, `INSTANCE`, `PERIODS`, `SEED`, `MODEL`,
etc.) but should be kept constant for a valid experiment.

## Temperatures tested

```text
0.2  0.4  0.6  0.8  1.0
```

## Where results are stored

```text
results/
  temperature/
    T_0.2/
      sweep_run.json                 # temperature, model, full config, seed, timestamps, git commit
      console.log
      <instance>-<UTCstamp>-seed<seed>/   # unchanged experiment output (run.json, summary.json, candidates.jsonl, ...)
    T_0.4/ ...
    T_0.6/ ...
    T_0.8/ ...
    T_1.0/ ...
```

Results from different temperatures are never mixed. Existing results are never
modified or deleted. If any run fails the sweep aborts with a non-zero exit
code.

---

# Fitness

Individuals are evaluated using:

```text
Fitness = (HCV + 1) × SCV
```

where:

- **HCV** = Hard Constraint Violations
- **SCV** = Soft Constraint Violations

The first objective is to obtain:

```text
HCV = 0
```

Once a feasible timetable is obtained, lower SCV values are preferred.

---

# Experimental Output

Each experiment records information such as:

- Generated GP heuristic
- Generation
- HCV
- SCV
- Fitness
- Crossover and mutation operations
- LLM calls
- Evaluation time
- Best heuristic found

The output directory is:

```text
results/
```
