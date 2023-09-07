ARG PARENT_IMAGE_NAME

FROM ${PARENT_IMAGE_NAME}:latest

ARG LIBOPENBLAS_VERSION

# Install libopenblas ##
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y libopenblas-base \
       libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
