# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements-serving.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip==25.3 && /opt/venv/bin/pip install -r requirements-serving.txt

FROM python:3.11-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/app/src
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src /app/src
COPY configs/serving.docker.yaml /app/configs/serving.yaml
USER appuser
EXPOSE 8000
CMD ["uvicorn", "constituency_insights.api:app", "--host", "0.0.0.0", "--port", "8000"]
