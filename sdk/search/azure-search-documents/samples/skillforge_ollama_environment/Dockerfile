FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY skillforge_ollama_environment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY sample_skillforge_langgraph_ollama.py /app/sample_skillforge_langgraph_ollama.py
COPY skillforge_ollama_environment/app.py /app/app.py

CMD ["python", "app.py"]
