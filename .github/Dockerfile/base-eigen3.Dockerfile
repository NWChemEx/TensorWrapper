ARG PARENT_IMAGE_NAME

FROM ${PARENT_IMAGE_NAME}:latest

ARG EIGEN3_VERSION

# Install libeigen3 ##
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y libeigen3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
