# Hugging Face Spaces image. Streamlit only — cloud mode runs the graph in-process,
# so there is no FastAPI process and nothing to supervise.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv

# spaces run the container as uid 1000, so everything is built under that user from here on
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:/home/user/.local/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR $HOME/app

# deps before source: the lock changes far less often than the code, so this layer is cached
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    --extra files --extra scraping --extra search --extra observability --extra export

# README.md is not documentation here — hatchling reads it to build the project itself
COPY --chown=user README.md ./
COPY --chown=user amaris ./amaris
COPY --chown=user frontend ./frontend
COPY --chown=user .streamlit ./.streamlit
RUN uv sync --frozen --no-dev \
    --extra files --extra scraping --extra search --extra observability --extra export

# playwright's own list, not a hand-written apt one: the package names differ across debian releases
USER root
RUN playwright install-deps chromium && rm -rf /var/lib/apt/lists/*
USER user

# both downloads happen on the first question otherwise, and the first visitor pays for them
RUN playwright install chromium \
    && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# no docker here, so redis and the api are absent and the graph runs inside streamlit
ENV DEPLOYMENT_MODE=cloud \
    LOG_DIR=/tmp/logs \
    SQLITE_CHECKPOINT_DB=/tmp/checkpoints.db \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860
CMD ["streamlit", "run", "frontend/app.py"]
