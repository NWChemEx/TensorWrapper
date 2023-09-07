ARG PARENT_IMAGE_NAME

FROM ${PARENT_IMAGE_NAME}:latest

ARG TILEDARRAY_VERSION

# Install tiledarray
ENV INSTALL_PATH=../install
RUN git clone https://github.com/ValeevGroup/TiledArray.git tiledarray \
    && cd tiledarray \
    && git checkout ${TILEDARRAY_VERSION} \
    && cmake -Bbuild -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
    && cmake --build build \
    && cmake --build build --target install \
    && cd .. \
    && rm -rf tiledarray
