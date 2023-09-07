ARG PARENT_IMAGE_NAME

FROM ${PARENT_IMAGE_NAME}:latest

ARG LIBCBLAS_VERSION

# Install libcblas ##
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y libgslcblas0 \
       libgsl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
