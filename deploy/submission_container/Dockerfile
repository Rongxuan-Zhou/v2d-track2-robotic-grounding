# Maniphys Lab, V2D Challenge Track 2 submission container (see /submission/README.md).
# Base: the FlashCHORD image every counted episode was evaluated in (flash_chord:latest on the training hosts).
ARG BASE_IMAGE=flash_chord:latest
FROM ${BASE_IMAGE}
# The kit's Python requirements, in a separate venv so the evaluator runtime in /opt/venv stays as evaluated.
COPY submission/kit/requirements.txt /tmp/kit-requirements.txt
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN uv venv /opt/kitvenv --python 3.12 && \
    uv pip install --python /opt/kitvenv/bin/python --index-url ${PIP_INDEX} -r /tmp/kit-requirements.txt "pandas<3"
# The evaluator ran with the pinned FlashCHORD source (8c778f9) mounted at /workspace; ship it there.
RUN rm -rf /workspace
COPY fc_8c778f9 /workspace
COPY submission /submission
RUN chmod 755 /submission/run.sh && chmod -R a+rX /submission
WORKDIR /submission
CMD ["/submission/run.sh", "verify"]
