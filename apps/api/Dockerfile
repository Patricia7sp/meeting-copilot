FROM python:3.11-slim
WORKDIR /app
COPY apps/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# main.py espera orchestrator.py + window.py no mesmo dir (sys.path local)
COPY apps/api/main.py ./main.py
COPY apps/orchestrator/orchestrator.py ./orchestrator.py
COPY apps/orchestrator/llm_openrouter.py ./llm_openrouter.py
COPY apps/orchestrator/classifier.py ./classifier.py
COPY packages/context/window.py ./window.py
# UI servida em /ui a partir do arquivo original
COPY apps/ui/index.html ./ui_index.html
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
