mkdir -p results
docker run --rm \
  --network host \
  -e OLLAMA_HOST=http://127.0.0.1:11434 \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/results:/app/results" \
  llm-gp-hh \
  python -m llm_gp_hh.experiments.run \
    --crs ./data/toronto/car-f-92.crs --stu ./data/toronto/car-f-92.stu \
    --periods 32 --profile dev \
    --population-size 20 --generations 10 --tournament-size 4 \
    --initial-batch-size 4 --crossover-rate 0.8 --mutation-rate 0.2 \
    --retry-limit 2 --seed 1001