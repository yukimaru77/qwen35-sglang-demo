FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    docker.io \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY scripts /app/scripts
COPY assets /app/assets
COPY notebooks /app/notebooks
COPY README.md /app/README.md

RUN pip install --no-cache-dir pillow requests jupyter nbformat nbconvert

CMD ["bash"]
