FROM ubuntu:20.04 AS base
USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'deb http://archive.ubuntu.com/ubuntu jammy main universe' > /etc/apt/sources.list.d/jammy-temp.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3.10-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN mkdir -p /kvs

FROM base AS hf

RUN pip install "huggingface_hub[cli]"

RUN hf download PaddlePaddle/PP-OCRv5_server_det inference.json inference.pdiparams inference.yml --local-dir ./kvs/models/det --cache-dir ./tmp/hf_tmp_cache && \
    hf download PaddlePaddle/PP-OCRv5_server_rec inference.json inference.pdiparams inference.yml --local-dir ./kvs/models/rec --cache-dir ./tmp/hf_tmp_cache

FROM base AS venv

COPY requirements.txt requirements.txt

RUN python -m venv /kvs/.venv
ENV PATH="/kvs/.venv/bin:${PATH}"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

FROM base AS final

RUN groupadd -g 1000 kvs \
    && useradd -m -s /bin/bash -u 1000 -g 1000 kvs

COPY --chown=kvs:kvs --from=hf /kvs/models /kvs/models
COPY --chown=kvs:kvs --from=venv /kvs/.venv /kvs/.venv

ENV PATH="/kvs/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libre2-5 libssl1.1 libb64-0d libnuma1 libarchive13 libgl1-mesa-glx libglib2.0-0 libgomp1 ccache binutils build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN paddlex --install paddle2onnx && paddlex --install hpi-cpu
ENV USE_HPIP=1

USER kvs
WORKDIR /kvs

COPY --chown=kvs:kvs . .
EXPOSE 8000
CMD ["hypercorn", "main:app", "--workers", "4", "--log-level", "info", "--access-log", "-", "--error-log", "-", "-b", "0.0.0.0:8000", "--log-config", "json:logging.json"]