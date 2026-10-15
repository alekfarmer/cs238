# syntax=docker/dockerfile:1
# Images for AA228/CS238 Project 1. Everything is installed at build time.

# --- runtime: Python plus the libraries the pipeline needs ---------------------
FROM python:3.12.7-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /work

# --- dev: runtime plus test / lint / type-check tools -------------------------
FROM runtime AS dev
COPY docker/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# --- tools: GitHub CLI, used only to open pull requests -----------------------
FROM debian:bookworm-slim AS tools
ENV HOME=/tmp
RUN apt-get update \
    && apt-get install -y --no-install-recommends gh git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
