FROM python:3.12-slim

WORKDIR /app

# libgomp1 needed by scikit-learn / umap-learn
RUN apt-get update && apt-get install -y \
    libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# Install backend deps first (layer caching)
COPY requirements-cloudrun.txt .
RUN pip install --no-cache-dir -r requirements-cloudrun.txt

# Copy backend code (mounted from hitl-dss-react/backend at build time)
COPY backend_deps/ /app/backend/

# Copy evaluation code
COPY . /app/evaluation/

# Both on PYTHONPATH
ENV PYTHONPATH="/app/backend:/app/evaluation"

ENTRYPOINT ["python", "-u", "/app/evaluation/workers/eval_worker.py"]
