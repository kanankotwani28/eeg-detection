# Environment only. NO DATA is copied into the image (TDBRAIN's data-use agreement forbids redistribution).
# Mount data at runtime:
#   docker run --rm -v /path/to/TDBRAIN:/data/tdbrain:ro -v /path/to/mumtaz:/data/mumtaz:ro \
#     -v $PWD/artifacts:/app/artifacts -v $PWD/mlruns:/app/mlruns eeg-fl-dp make reproduce
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy MLFLOW_DISABLE_AGENT_HINT=1 \
    TDBRAIN_PATH=/data/tdbrain MUMTAZ_PATH=/data/mumtaz
RUN apt-get update && apt-get install -y --no-install-recommends make git && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY . .
RUN rm -rf data artifacts mlruns && mkdir -p data artifacts
EXPOSE 8501
CMD ["make", "test"]
