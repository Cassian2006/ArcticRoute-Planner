# syntax=docker/dockerfile:1.6
# Multi-stage optional (builder minimal)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (keep slim)
# RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources && sed -i 's|security.debian.org|mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources && apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    curl \
#    git \
#    ca-certificates \
#    libgdal-dev \
#    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy only dependency manifests first to leverage layer cache
COPY requirements.txt /app/requirements.txt
COPY ArcticRoute/requirements.txt /app/ArcticRoute.requirements.txt
COPY streamlit-maps-main/requirements.txt /app/maps.requirements.txt

# Install requirements (best-effort if sub file missing)
RUN python -m pip install --upgrade pip && \
    if [ -f /app/ArcticRoute.requirements.txt ]; then \
      pip install -r /app/ArcticRoute.requirements.txt; \
    fi && \
    if [ -f /app/requirements.txt ]; then \
      pip install -r /app/requirements.txt; \
    fi && \
    if [ -f /app/maps.requirements.txt ]; then \
      pip install -r /app/maps.requirements.txt; \
    fi

# Copy source
COPY . /app

# Create non-root user and data mount point
RUN useradd -m -u 10001 arctic && \
    mkdir -p /data && \
    mkdir -p /app/outputs /app/reports && \
    chown -R arctic:arctic /app /data

USER arctic

# Expose Streamlit default port
EXPOSE 8501

# Default environment
ENV ARCTICROUTE_ROOT=/app \
    PYTHONPATH=/app \
    ARCTICROUTE_DATA=/data

# Default entrypoint supports CLI or UI; can be overridden by compose
ENTRYPOINT ["python", "-m", "api.cli"]
# To run UI: docker run ... python -m ArcticRoute.api.cli serve --ui --port 8501

