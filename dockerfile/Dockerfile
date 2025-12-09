FROM python:3.14-slim

ARG DEBIAN_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/debian"
ARG DEBIAN_SECURITY_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/debian-security"
ARG PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

# Point apt and pip/uv to mirrors close to Beijing for faster downloads
RUN set -eux; \
  . /etc/os-release; \
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

# Copy dependency files needed for dependency resolution
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project

# Copy the application code
COPY . .

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev

# Expose the port that FastAPI will run on
EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

CMD [ \
  "uvicorn", \
  "src.main:app", \
  "--host", "0.0.0.0", \
  "--port", "8000" \
]
