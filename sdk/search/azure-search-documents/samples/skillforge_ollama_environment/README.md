# SkillForge AI + Ollama standalone environment

This folder provides a **separate runnable environment** for the SkillForge AI sample:

- Source workflow: `../sample_skillforge_langgraph_ollama.py`
- Container entrypoint wrapper: `./app.py`
- Dependencies: `./requirements.txt`
- Containerization: `./Dockerfile`
- Kubernetes manifests: `./kubernetes/*`

## 1. Local Python run

From `sdk/search/azure-search-documents/samples/skillforge_ollama_environment`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## 2. Environment variables

Set these for Ollama:

```bash
export OLLAMA_BASE_URL="http://host.docker.internal:11434/v1"
export OLLAMA_MODEL="llama3.1:8b"
export OLLAMA_API_KEY="ollama"
export JOB_SEARCH_QUERY_SUFFIX="gen ai python remote"
```

> For local non-container runs, `OLLAMA_BASE_URL` can stay `http://localhost:11434/v1`.

## 3. Docker build and run

Build from the parent `samples` directory so the Docker context contains both the wrapper and sample file:

```bash
cd ..
docker build -f skillforge_ollama_environment/Dockerfile -t skillforge-ollama:latest .
```

Run:

```bash
docker run --rm \
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434/v1" \
  -e OLLAMA_MODEL="llama3.1:8b" \
  -e OLLAMA_API_KEY="ollama" \
  -e JOB_SEARCH_QUERY_SUFFIX="gen ai python remote" \
  skillforge-ollama:latest
```

## 4. Kubernetes deployment

The manifests in `kubernetes/` include:

- `configmap.yaml` for non-secret settings
- `secret.example.yaml` template for API key
- `deployment.yaml` to run the container

Apply in order:

```bash
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secret.example.yaml
kubectl apply -f kubernetes/deployment.yaml
```

> Update the deployment image (`your-registry/skillforge-ollama:latest`) before applying.
