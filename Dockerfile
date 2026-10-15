# ---- Stage 1: train the model inside the build (reproducible, no binaries in git)
FROM python:3.13-slim AS trainer
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/ data/
RUN python -m src.train --data data/telco_churn.csv --out models

# ---- Stage 2: slim runtime image
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home appuser
COPY src/ src/
COPY app/ app/
COPY --from=trainer /build/models/ models/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
