FROM python:3.14-slim AS base

ARG DEBIAN_MIRROR="https://mirrors.ivolces.com/debian"
ARG DEBIAN_SECURITY_MIRROR="https://mirrors.ivolces.com/debian-security"
ARG PYPI_MIRROR="https://mirrors.ivolces.com/pypi/simple"

# Point apt and pip/uv to mirrors close to Beijing for faster downloads
RUN set -eux; \
  . /etc/os-release; \
  rm -f /etc/apt/sources.list.d/debian.sources; \
  rm -f /etc/apt/sources.list.d/debian-security.sources || true; \
  printf '%s\n' \
    "deb ${DEBIAN_MIRROR} ${VERSION_CODENAME} main contrib non-free non-free-firmware" \
    "deb ${DEBIAN_MIRROR} ${VERSION_CODENAME}-updates main contrib non-free non-free-firmware" \
    "deb ${DEBIAN_MIRROR} ${VERSION_CODENAME}-backports main contrib non-free non-free-firmware" \
    "deb ${DEBIAN_SECURITY_MIRROR} ${VERSION_CODENAME}-security main contrib non-free non-free-firmware" \
    > /etc/apt/sources.list; \
  printf "[global]\nindex-url = %s\n" "${PYPI_MIRROR}" > /etc/pip.conf

ENV UV_INDEX_URL=${PYPI_MIRROR} \
    PIP_INDEX_URL=${PYPI_MIRROR}

# Install system build dependencies required by Rust-based Python packages
RUN --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt \
  apt-get update \
  && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

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
    RUSTUP_HOME=/root/.rustup
ENV PATH="/root/.cargo/bin:${PATH}"

# Install curl for rustup installation (isolated to this stage)
RUN --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt \
  apt-get update \
  && apt-get install -y --no-install-recommends curl pkg-config \
  && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable \
    && /root/.cargo/bin/rustc --version \
    && /root/.cargo/bin/cargo --version

RUN python -m pip install --no-cache-dir maturin==1.7.4

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

CMD [ \
  "uvicorn", \
  "src.main:app", \
  "--host", "0.0.0.0", \
  "--port", "8000" \
]
