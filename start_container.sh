#!/usr/bin/env bash
set -euo pipefail

# Define default values; override by exporting CONTAINER_NAME or DOCKER_RUN_ARGS before running.
CONTAINER_NAME="${CONTAINER_NAME:-memu-server}"
# Map host 8080 to container 8000 (FastAPI/gunicorn default); add env flags as needed.
DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS:---network=bridge -p 8080:8000 -e ENV_VAR=value}"

: "${SERVER_IMAGE:?SERVER_IMAGE must be set to the image tag to run}"

echo "Starting a new container '${CONTAINER_NAME}' with image '${SERVER_IMAGE}'..."
docker run -d --name "${CONTAINER_NAME}" ${DOCKER_RUN_ARGS} "${SERVER_IMAGE}"
echo "Service has been started with the latest image."
