ARG PARENT_IMAGE_NAME

FROM ${PARENT_IMAGE_NAME}:latest

ARG OPENMPI_VERSION

# Install libopenmpi ##
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y openmpi-bin \
       libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
