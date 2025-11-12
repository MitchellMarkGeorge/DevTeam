FROM python:3.12-slim as base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# set working directory
WORKDIR /app

# Install dependencies
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
    
COPY . /app

# new image without uv
FROM python:3.12-slim

WORKDIR /app

# copy app directory from previous stage
COPY --from=base /app /app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
# Run the application.
CMD ["fastapi", "run", "app/main.py", "--port", "8000", "--host", "0.0.0.0"]