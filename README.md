# 🧠 COS790 — LLM-Assisted Genetic Programming Hyper-Heuristic

## Automated Generation of Construction Heuristics for Educational Timetabling

This project investigates whether a **Large Language Model (LLM)** can replace selected generative components of a traditional **Genetic Programming (GP) generation construction hyper-heuristic** for educational timetabling.

The implementation is based on the generation construction hyper-heuristic approach described by **Pillay & Özcan (2019)** for the **Toronto Examination Timetabling Problem**.

The key idea is:

> Keep the conventional Genetic Programming framework, but use a local Large Language Model to generate the initial heuristic population and perform crossover and mutation.

The LLM used in this project is:

```text
Qwen3-Coder 30B
```

running locally through:

```text
Ollama
```

---

# 🌱 What Does This Project Do?

The LLM **does not directly create a timetable**.

Instead, Qwen generates and evolves **GP heuristic trees**.

These heuristics answer the question:

> Which unscheduled exam should be scheduled next?

The fixed timetable constructor then places the selected exam into an appropriate feasible period.

The complete process is:

```text
                 VANILLA GP FRAMEWORK
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    Initialisation    Selection      Variation
          │              │              │
          │         Tournament          │
          │          Selection          │
          │                             │
          ▼                             ▼
        Qwen                     ┌────────────┐
          │                      │            │
          │                      ▼            ▼
          │                  Crossover     Mutation
          │                      │            │
          └──────────────┬───────┴────────────┘
                         ▼
                GP Heuristic Trees
                         │
                         ▼
              Score Unscheduled Exams
                         │
                         ▼
               Select Highest Score
                         │
                         ▼
               Timetable Constructor
                         │
                         ▼
                    HCV + SCV
                         │
                         ▼
                      Fitness
                         │
                         ▼
                Next GP Generation
```

---

# 🧩 Example Heuristic

A generated GP heuristic might look like this:

```text
(* (/ c g) (+ a b))
```

The terminals represent information about the current timetabling state.

| Terminal | Meaning |
|---|---|
| `a` | Minimum incremental Toronto proximity penalty over currently feasible periods |
| `b` | Clashes with currently unallocated exams |
| `c` | Total potential clashes |
| `d` | Exam enrolment |
| `e` | Number of currently feasible periods |
| `f` | Clashes with already allocated exams |
| `g` | Total number of students |
| `h` | Total number of available periods |

The heuristic is evaluated for every unscheduled exam.

The exam with the **highest heuristic score** is selected next.

The timetable constructor then attempts to place that exam into a feasible period with the lowest incremental soft-constraint penalty.

---

# 🏗️ What Is Conventional GP and What Does the LLM Replace?

The project keeps the following parts of Genetic Programming conventional:

- GP population
- GP tree representation
- Tournament selection
- Terminal set
- Function set
- Timetable constructor
- Fitness calculation
- Population evolution loop
- Experimental logging
- Termination conditions

The LLM replaces:

- Initial population generation
- GP crossover
- GP mutation

Therefore:

```text
Traditional GP
    ↓
Random tree generation
Random subtree crossover
Random mutation
```

becomes:

```text
LLM-Assisted GP
    ↓
Qwen initial tree generation
Qwen crossover
Qwen mutation
```

---

# 🧬 GP Grammar

The LLM may only create trees using the predefined GP grammar.

## Terminals

```text
a b c d e f g h
```

## Arithmetic Functions

```text
+
-
*
/
```

Division is protected.

If the denominator is zero, the evaluator returns `1`.

## Relational Functions

```text
<
>
<=
>=
==
!=
```

## Conditional Function

```text
if
```

Example:

```text
(if (> a b) c d)
```

---

# 📁 Project Structure

```text
COS790HHPROJECT2/
│
├── README.md
├── pyproject.toml
│
├── data/
│   └── toronto/
│       ├── car-f-92.crs
│       ├── car-f-92.stu
│       └── ...
│
├── src/
│   └── llm_gp_hh/
│       │
│       ├── config.py
│       ├── rng.py
│       ├── live_smoke.py
│       │
│       ├── gp/
│       │   ├── evolution.py
│       │   ├── individual.py
│       │   ├── selection.py
│       │   └── tree.py
│       │
│       ├── llm/
│       │   ├── ollama_client.py
│       │   ├── operators.py
│       │   ├── prompts.py
│       │   └── protocol.py
│       │
│       ├── toronto/
│       │   ├── attributes.py
│       │   ├── constructor.py
│       │   ├── fitness.py
│       │   ├── model.py
│       │   └── parser.py
│       │
│       └── experiments/
│           ├── logging.py
│           ├── reference_results.py
│           └── run.py
│
└── tests/
    ├── fixtures/
    ├── test_config.py
    ├── test_evolution.py
    ├── test_llm_operators.py
    ├── test_logging.py
    ├── test_selection.py
    ├── test_toronto_attributes.py
    ├── test_toronto_constructor.py
    ├── test_toronto_parser.py
    └── test_tree.py
```

---

# 🚀 Installation

The project supports:

- ✅ Windows
- ✅ macOS
- ✅ Linux

You will need:

1. Git
2. Python 3.11 or newer
3. Ollama
4. Qwen3-Coder 30B

If you only want a consistent, reproducible environment to run experiments in,
skip the per-OS setup below and jump to **🐳 Running with Docker**.

---

# 🐳 Running with Docker

Docker gives every group member the same Python version and the same pinned
dependency versions, so experiment runs are reproducible across machines.

### What runs where

The container runs **only the Python experiment code**. The Qwen model is large
and GPU-dependent, so **Ollama keeps running on the host** (exactly as in the
native setup). The container talks to it over HTTP.

```text
Docker container (Python experiment)  ──HTTP──►  Ollama on host  ──►  qwen3-coder:30b
```

So you still need Ollama installed on the host with the model pulled:

```bash
ollama pull qwen3-coder:30b
ollama serve   # Linux: leave running. macOS/Windows: the Ollama app is enough.
```

### Prerequisites

- Docker
- Ollama running on the host with `qwen3-coder:30b` pulled

### 1. Build the image

```bash
docker build -t llm-gp-hh .
```

Dependency versions are pinned in [`requirements.lock.txt`](requirements.lock.txt),
so the build is reproducible.

### 2. Run a single experiment

Results are written to `results/` on the host via a bind mount. `--user` keeps
the generated files owned by you rather than root. `--add-host` lets the
container reach Ollama on the host as `host.docker.internal` (needed on Linux;
harmless on macOS/Windows).

```bash
mkdir -p results

docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/results:/app/results" \
  llm-gp-hh \
  python -m llm_gp_hh.experiments.run \
    --crs ./data/toronto/car-f-92.crs \
    --stu ./data/toronto/car-f-92.stu \
    --periods 32 \
    --profile dev \
    --population-size 20 \
    --generations 10 \
    --tournament-size 4 \
    --initial-batch-size 4 \
    --crossover-rate 0.8 \
    --mutation-rate 0.2 \
    --retry-limit 2 \
    --seed 1001
```

The benchmark data in `data/` is baked into the image, so `--crs` / `--stu`
paths are relative to the project directory as usual.

> **Windows PowerShell:** use `${PWD}` instead of `$PWD`, drop the
> `--user "$(id -u):$(id -g)"` line, and put everything on one line with
> backticks (`` ` ``) for line continuation. The Ollama desktop app just needs
> to be running.

If your Ollama runs somewhere else (or on a non-default port), override the URL:

```bash
docker run --rm -e OLLAMA_HOST=http://192.168.1.50:11434 ... llm-gp-hh ...
```

### 3. Run the full experiment suite

`scripts/run_suite.sh` repeats one configuration across several seeds (default:
seeds 1001–1005 on `car-f-92`), writing each run into its own directory under
`results/`.

```bash
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/results:/app/results" \
  llm-gp-hh \
  bash scripts/run_suite.sh car-f-92 32 1001 1002 1003 1004 1005
```

Configuration can be overridden with environment variables, e.g.
`-e POPULATION_SIZE=20 -e GENERATIONS=10 -e MODEL=qwen3-coder:30b`
(see the top of [`scripts/run_suite.sh`](scripts/run_suite.sh)).

### 4. Persist / access results from the host

The `-v "$PWD/results:/app/results"` bind mount means every run directory
(`run.json`, `candidates.jsonl`, `llm_calls.jsonl`, `generations.csv`,
`best_heuristic.json`, `summary.json`) appears under `./results/` on the host
and survives the container exiting. Nothing else needs to be persisted.

### Run the tests in the container (optional)

```bash
docker run --rm llm-gp-hh python -m pytest -q
```

---

# 🪟 Windows Setup

## 1. Install Git

Check whether Git is installed:

```powershell
git --version
```

If Git is installed, you should see something similar to:

```text
git version 2.x.x
```

If not, install **Git for Windows**.

---

## 2. Install Python

Python **3.11 is recommended**.

Check your installed versions:

```powershell
py --list
```

or:

```powershell
python --version
```

You should have:

```text
Python 3.11+
```

---

## 3. Install Ollama

Install Ollama for Windows.

After installation, check:

```powershell
ollama --version
```

---

## 4. Clone the Repository

```powershell
git clone https://github.com/nikhilpg12/COS790HHPROJECT2.git
```

Enter the project:

```powershell
cd COS790HHPROJECT2
```

---

## 5. Create a Virtual Environment

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should now see something similar to:

```text
(.venv) PS C:\...
```

### PowerShell Activation Error

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 6. Install the Project

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project:

```powershell
pip install -e ".[dev]"
```

---

## 7. Download Qwen3-Coder

```powershell
ollama pull qwen3-coder:30b
```

The model is large, so downloading may take some time.

Check that it is installed:

```powershell
ollama list
```

You should see:

```text
qwen3-coder:30b
```

---

# 🍎 macOS Setup

## 1. Install Git

Check:

```bash
git --version
```

If you use Homebrew:

```bash
brew install git
```

---

## 2. Install Python 3.11

Using Homebrew:

```bash
brew install python@3.11
```

Check:

```bash
python3.11 --version
```

You should see:

```text
Python 3.11.x
```

---

## 3. Install Ollama

Using Homebrew:

```bash
brew install ollama
```

Alternatively, install the official Ollama macOS application.

Check:

```bash
ollama --version
```

---

## 4. Clone the Repository

```bash
git clone https://github.com/nikhilpg12/COS790HHPROJECT2.git
```

Enter the project:

```bash
cd COS790HHPROJECT2
```

---

## 5. Create a Virtual Environment

```bash
python3.11 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## 6. Install the Project

```bash
python -m pip install --upgrade pip
```

Then:

```bash
pip install -e ".[dev]"
```

---

## 7. Download Qwen3-Coder

```bash
ollama pull qwen3-coder:30b
```

Check:

```bash
ollama list
```

You should see:

```text
qwen3-coder:30b
```

---

# 🐧 Linux Setup

These instructions are suitable for Ubuntu/Debian-based systems.

---

## 1. Install Git

```bash
sudo apt update
```

```bash
sudo apt install git
```

Check:

```bash
git --version
```

---

## 2. Install Python

```bash
sudo apt install python3 python3-venv python3-pip
```

Check:

```bash
python3 --version
```

You need:

```text
Python 3.11+
```

---

## 3. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Check:

```bash
ollama --version
```

If Ollama is not running:

```bash
ollama serve
```

Leave that terminal open and use another terminal for the experiment.

---

## 4. Clone the Repository

```bash
git clone https://github.com/nikhilpg12/COS790HHPROJECT2.git
```

Enter the project:

```bash
cd COS790HHPROJECT2
```

---

## 5. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

## 6. Install the Project

```bash
python -m pip install --upgrade pip
```

Then:

```bash
pip install -e ".[dev]"
```

---

## 7. Download Qwen3-Coder

```bash
ollama pull qwen3-coder:30b
```

Check:

```bash
ollama list
```

---

# ✅ Test the Installation

Before running an experiment, run the automated tests.

The command is the same on all operating systems:

```bash
pytest -q
```

All tests should pass.

Example:

```text
38 passed
```

If tests fail, fix the installation problem before running a full experiment.

---

# 🤖 Test Qwen Before Running an Experiment

You can test the model directly.

Run:

```bash
ollama run qwen3-coder:30b
```

Then type something simple:

```text
Write a Python function that calculates the mean of a list.
```

If Qwen responds, the model is working correctly.

Exit Ollama using:

```text
/bye
```

---

# 🧪 First Small Experiment

Before running the larger experiment, it is a good idea to perform a small test run.

This confirms that:

- Qwen is accessible
- GP trees can be generated
- Timetables can be constructed
- Fitness can be calculated
- Results can be saved

The first benchmark used is:

```text
car-f-92 I
```

The actual dataset filenames are:

```text
car-f-92.crs
car-f-92.stu
```

The instance contains:

```text
32 periods
```

---

# 🪟 Windows — Small Test Run

```powershell
python -m llm_gp_hh.experiments.run `
  --crs .\data\toronto\car-f-92.crs `
  --stu .\data\toronto\car-f-92.stu `
  --periods 32 `
  --profile dev `
  --population-size 4 `
  --generations 2 `
  --tournament-size 2 `
  --initial-batch-size 4 `
  --crossover-rate 0.8 `
  --mutation-rate 0.2 `
  --retry-limit 2 `
  --seed 1001
```

---

# 🍎 macOS — Small Test Run

```bash
python -m llm_gp_hh.experiments.run \
  --crs ./data/toronto/car-f-92.crs \
  --stu ./data/toronto/car-f-92.stu \
  --periods 32 \
  --profile dev \
  --population-size 4 \
  --generations 2 \
  --tournament-size 2 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001
```

---

# 🐧 Linux — Small Test Run

```bash
python -m llm_gp_hh.experiments.run \
  --crs ./data/toronto/car-f-92.crs \
  --stu ./data/toronto/car-f-92.stu \
  --periods 32 \
  --profile dev \
  --population-size 4 \
  --generations 2 \
  --tournament-size 2 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001
```

---

# 👀 What Should You See?

The program displays progress while it runs.

Example:

```text
========================================================================
[RUN] car-f-92 | model=qwen3-coder:30b | seed=1001
      population=4 | generations=2 | tournament=2
      crossover=0.80 | mutation=0.20 | initial_batch=4
========================================================================

[GEN 0] Creating initial population...

[LLM] Requesting 4 initial heuristic tree(s) from Qwen...

[EVAL] g0-i0 | evaluating (+ a (* b c))
       HCV=11 | SCV=4.8726 | Fitness=58.4709

[EVAL] g0-i1 | evaluating (if (< d e) (+ f g) h)
       HCV=46 | SCV=2.7896 | Fitness=131.1096
```

During later generations you will see:

```text
[SELECT] Crossover
[LLM] Asking Qwen to perform crossover...

[SELECT] Mutation
[LLM] Asking Qwen to mutate the parent...

[PROGRESS] Generation 1: 10/20 offspring evaluated
```

If an LLM crossover or mutation fails, the operation is skipped and the evolutionary run continues.

Example:

```text
[SKIP] Mutation failed
       Skipping this mutation attempt and continuing...
```

---

# 🧪 First Full Experiment

Once the small test succeeds, run the first proper experiment.

Configuration:

| Parameter | Value |
|---|---:|
| Instance | `car-f-92 I` |
| Population | 20 |
| Generations | 10 |
| Tournament size | 4 |
| Crossover rate | 0.8 |
| Mutation rate | 0.2 |
| LLM retry limit | 2 |
| Initial batch size | 4 |
| Model | `qwen3-coder:30b` |
| Seed | 1001 |

This evaluates:

```text
20 × 10 = 200 heuristic candidates
```

---

# 🪟 Windows — First Full Experiment

```powershell
python -m llm_gp_hh.experiments.run `
  --crs .\data\toronto\car-f-92.crs `
  --stu .\data\toronto\car-f-92.stu `
  --periods 32 `
  --profile dev `
  --population-size 20 `
  --generations 10 `
  --tournament-size 4 `
  --initial-batch-size 4 `
  --crossover-rate 0.8 `
  --mutation-rate 0.2 `
  --retry-limit 2 `
  --seed 1001
```

---

# 🍎 macOS — First Full Experiment

```bash
python -m llm_gp_hh.experiments.run \
  --crs ./data/toronto/car-f-92.crs \
  --stu ./data/toronto/car-f-92.stu \
  --periods 32 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001
```

---

# 🐧 Linux — First Full Experiment

```bash
python -m llm_gp_hh.experiments.run \
  --crs ./data/toronto/car-f-92.crs \
  --stu ./data/toronto/car-f-92.stu \
  --periods 32 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001
```

---

# 📊 Understanding the Results

Three important values are reported.

## HCV

```text
Hard Constraint Violations
```

Hard constraints represent invalid timetable conditions.

The most important target is:

```text
HCV = 0
```

This means the timetable is **feasible**.

---

## SCV

```text
Soft Constraint Violations
```

SCV measures timetable quality.

Once:

```text
HCV = 0
```

has been achieved, a lower SCV is better.

---

## Fitness

The fitness function is:

```text
Fitness = (HCV + 1) × SCV
```

Lower fitness is better.

However, always inspect:

```text
HCV
SCV
```

individually when analysing experimental results.

---

# 🎯 What Is a Good Result?

The first objective is:

```text
HCV = 0
```

because this means the evolved heuristic successfully guided the constructor to produce a feasible timetable.

Once feasibility is achieved, the objective becomes reducing:

```text
SCV
```

The published Pillay & Özcan AHH reference for `car-f-92 I` is approximately:

```text
HCV = 0
SCV = 4.32
```

This provides a useful reference point for experimental comparison.

---

# 🏆 Example Successful Result

A successful experimental run produced the following heuristic:

```text
(* (+ (if (> f d) f d) (/ c e)) (+ (/ f g) (/ c e)))
```

with:

```text
HCV = 0
SCV = 4.60595
```

This means the generated heuristic successfully produced a **feasible Toronto examination timetable**.

---

# 📈 Evolution Across Generations

A typical successful run may initially produce infeasible timetables.

For example:

```text
Generation 0: HCV = 5
Generation 1: HCV = 4
Generation 2: HCV = 4
Generation 3: HCV = 4
Generation 4: HCV = 1
Generation 5: HCV = 0
Generation 6: HCV = 0
Generation 7: HCV = 0
Generation 8: HCV = 0
Generation 9: HCV = 0
```

This demonstrates the evolutionary search progressively discovering better construction heuristics.

---

# 📂 Experiment Results

Experiment outputs are stored inside:

```text
results/
```

Each run creates a separate directory.

Important files include:

```text
run.json
candidates.jsonl
llm_calls.jsonl
generations.csv
best_heuristic.json
summary.json
```

---

## `run.json`

Contains the experimental configuration.

Example information:

```text
model
population size
number of generations
crossover rate
mutation rate
seed
runtime
```

---

## `candidates.jsonl`

Contains every evaluated GP heuristic.

Each candidate records information such as:

```text
tree
generation
operation
parents
HCV
SCV
fitness
evaluation time
```

---

## `llm_calls.jsonl`

Contains the LLM interaction history.

This includes:

```text
prompt
Qwen response
operation
latency
valid/invalid status
errors
```

This file is useful for analysing how the LLM behaved during evolution.

---

## `generations.csv`

Contains summary information for every generation.

Example:

```text
generation
population size
best fitness
best HCV
best SCV
mean fitness
mean HCV
mean SCV
crossover calls
mutation calls
```

---

## `best_heuristic.json`

Contains the best heuristic discovered during the run.

---

## `summary.json`

Contains the final experiment summary and published reference comparison.

---

# 🎲 Repeating Experiments

One run is **not sufficient for scientific conclusions**.

Use different random seeds for independent runs.

For example:

```text
1001
1002
1003
1004
1005
```

Do not treat repeated runs with exactly the same seed as statistically independent experiments.

---

# 🧪 Example Five-Run Experiment

Run the same configuration using:

```text
Seed 1001
Seed 1002
Seed 1003
Seed 1004
Seed 1005
```

Then compare:

- Feasibility rate
- Best HCV
- Best SCV
- Mean HCV
- Mean SCV
- Fitness
- Runtime
- Number of invalid LLM calls
- Heuristic complexity

---

# ⚠️ Troubleshooting

## `ollama` command not found

Restart the terminal after installing Ollama.

Check:

```bash
ollama --version
```

---

## Qwen Model Missing

Run:

```bash
ollama pull qwen3-coder:30b
```

Then:

```bash
ollama list
```

---

## Cannot Connect to Ollama

Start Ollama.

On Linux:

```bash
ollama serve
```

On Windows/macOS, make sure the Ollama application is running.

---

## PowerShell Will Not Activate `.venv`

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Python Module Not Found

Make sure the virtual environment is active.

Then reinstall:

```bash
pip install -e ".[dev]"
```

---

## Tests Fail

Run:

```bash
pytest -q
```

Read the first failure carefully before running experiments.

---

## LLM Mutation or Crossover Fails

The LLM may occasionally:

- return an unchanged parent
- return invalid GP syntax
- generate a malformed expression

The implementation validates LLM output.

If a mutation or crossover remains invalid after the configured retries, that operator attempt is skipped and evolution continues.

The experiment should therefore continue filling the population with valid offspring.

---

# 💻 Hardware Considerations

`qwen3-coder:30b` is a large local model.

Performance depends heavily on:

- GPU
- VRAM
- RAM
- CPU
- Ollama configuration

The model may run on CPU-only systems, but generation will generally be significantly slower.

The GP timetable evaluations themselves also require computation, particularly on larger Toronto benchmark instances.

---

# 🔒 Local LLM Execution

One important design choice is that the LLM runs locally.

```text
Experiment
   ↓
Local Python Code
   ↓
Local Ollama API
   ↓
Qwen3-Coder 30B
```

No cloud LLM API is required.

---

# 📚 Research Context

This project investigates the automated creation of **generation construction hyper-heuristics** for educational timetabling.

The baseline framework is based on:

> Pillay, N. & Özcan, E. (2019). Automated generation of constructive ordering heuristics for educational timetabling.

The research question is centred on whether an LLM can effectively replace the generative components of Genetic Programming while retaining the conventional evolutionary framework.

---

# 🔬 Core Research Idea

Traditional GP:

```text
GP Framework
    │
    ├── Random Initialisation
    ├── Tournament Selection
    ├── GP Crossover
    ├── GP Mutation
    │
    ▼
Heuristic
```

Proposed approach:

```text
GP Framework
    │
    ├── Qwen Initialisation
    ├── Tournament Selection
    ├── Qwen Crossover
    ├── Qwen Mutation
    │
    ▼
Heuristic
```

Everything downstream remains conventional:

```text
Heuristic
    ↓
Timetable Constructor
    ↓
HCV + SCV
    ↓
Fitness
```

---

# 🚦 Quick Start

If everything is already installed, the complete setup is:

## Windows

```powershell
git clone https://github.com/nikhilpg12/COS790HHPROJECT2.git
cd COS790HHPROJECT2

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"

ollama pull qwen3-coder:30b

pytest -q
```

Then run:

```powershell
python -m llm_gp_hh.experiments.run `
  --crs .\data\toronto\car-f-92.crs `
  --stu .\data\toronto\car-f-92.stu `
  --periods 32 `
  --profile dev `
  --population-size 20 `
  --generations 10 `
  --tournament-size 4 `
  --initial-batch-size 4 `
  --crossover-rate 0.8 `
  --mutation-rate 0.2 `
  --retry-limit 2 `
  --seed 1001
```

## macOS / Linux

```bash
git clone https://github.com/nikhilpg12/COS790HHPROJECT2.git
cd COS790HHPROJECT2

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"

ollama pull qwen3-coder:30b

pytest -q
```

Then run:

```bash
python -m llm_gp_hh.experiments.run \
  --crs ./data/toronto/car-f-92.crs \
  --stu ./data/toronto/car-f-92.stu \
  --periods 32 \
  --profile dev \
  --population-size 20 \
  --generations 10 \
  --tournament-size 4 \
  --initial-batch-size 4 \
  --crossover-rate 0.8 \
  --mutation-rate 0.2 \
  --retry-limit 2 \
  --seed 1001
```

---

# ✅ You Are Ready

If the following all work:

```text
✅ Python 3.11+
✅ Git
✅ Virtual environment
✅ pip install -e ".[dev]"
✅ Ollama
✅ qwen3-coder:30b
✅ pytest -q
```

then you are ready to run the LLM-assisted Genetic Programming hyper-heuristic.

The first milestone to look for is:

```text
HCV = 0
```

Once that occurs, the generated heuristic has successfully guided the construction of a **feasible examination timetable**.

---

# 📖 Reference

Pillay, N., & Özcan, E. (2019).

**Automated generation of constructive ordering heuristics for educational timetabling.**

*Annals of Operations Research, 275*, 181–208.

DOI:

```text
10.1007/s10479-017-2625-x
```

---

# 🎓 Project

```text
COS790
Large Language Models for the Automated Creation of
Generation Construction Hyper-Heuristics
for Educational Timetabling
```
