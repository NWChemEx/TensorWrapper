ARG PARENT_IMAGE_NAME

FROM ${PARENT_IMAGE_NAME}:latest

ARG SCALAPACK_VERSION

# Install libscalapack ##
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y libscalapack-openmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
