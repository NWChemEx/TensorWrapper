ARG PARENT_IMAGE_NAME

FROM ${PARENT_IMAGE_NAME}:latest

ARG LAPACKE_VERSION

# Install liblapacke ##
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y liblapacke \
       liblapacke-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
