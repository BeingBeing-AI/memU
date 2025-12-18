ARG BASE_IMAGE=starfy-docker-image-cn-shanghai.cr.volces.com/memu/memu-build-base:1.0.0
FROM ${BASE_IMAGE} AS base

# -----------------------------------------------------------------------------
# Dependency stage: resolve and install Python deps without the local project
# -----------------------------------------------------------------------------
FROM base AS deps

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project

# -----------------------------------------------------------------------------
# Builder stage: compile the Rust extension and package memu
# -----------------------------------------------------------------------------
FROM base AS builder

ENV CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup \
    RUSTUP_DIST_SERVER=https://rsproxy.cn \
    RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
ENV PATH="/root/.cargo/bin:${PATH}"

COPY . .

RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/app/target \
    maturin build --release --locked --strip --out /tmp/dist

RUN set -eux; \
    wheel="$(ls -1 /tmp/dist/memu_py-*.whl | head -n 1)"; \
    python -m pip install --no-deps --no-cache-dir --target /tmp/package "${wheel}"

# -----------------------------------------------------------------------------
# Runtime stage: start from cached deps, overlay source + compiled artifacts
# -----------------------------------------------------------------------------
FROM deps AS runtime

# Copy the rest of the application (invalidates only when sources change)
COPY . .

# Overwrite the source package with the freshly built binary artifacts
COPY --from=builder /tmp/package/memu /app/src/memu

# Expose the port that FastAPI will run on
EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:${PYTHONPATH:-}"

# 使用 Gunicorn 启动，设置 worker 数量
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "8", "-b", "0.0.0.0:8000", "src.main:app"]
