# LLM-assisted GP hyper-heuristic - reproducible experiment environment.
#
# The container runs ONLY the Python experiment code. The Qwen model keeps
# running on the host (or any reachable machine) through Ollama; the container
# talks to it over HTTP via the OLLAMA_HOST environment variable.
FROM python:3.11-slim

# Keep Python predictable and unbuffered so long experiment logs stream live.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app

# 1. Install the pinned dependency set first so this layer is cached.
COPY requirements.lock.txt ./
RUN pip install --upgrade pip && pip install -r requirements.lock.txt

# 2. Install the project itself (no deps - they are already pinned above).
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
RUN pip install --no-deps -e .

# Benchmark data is baked in; results are expected to be a bind mount.
COPY data/ ./data/
RUN mkdir -p results

# Default: show the experiment CLI help.
CMD ["python", "-m", "llm_gp_hh.experiments.run", "--help"]
