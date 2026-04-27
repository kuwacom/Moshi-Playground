ARG PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG HF_REPO=llm-jp/llm-jp-moshi-v1
ARG LORA_NAME=moshi-lora
ARG MOSHI_COMMIT=061cc4c630d9e11722e08b7d02b1836ba58f30e8

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_REPO=${HF_REPO} \
    LORA_NAME=${LORA_NAME} \
    HF_HOME=/opt/moshi/models/huggingface \
    NO_TORCH_COMPILE=1

WORKDIR /opt/moshi

# 学習環境全体ではなく推論に必要な Moshi と sphn だけを入れてイメージを軽くする
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      "sphn==0.1.12" \
      "moshi @ https://github.com/kyutai-labs/moshi/archive/${MOSHI_COMMIT}.zip#subdirectory=moshi" \
      "huggingface_hub"

COPY lora/ /opt/moshi/lora/
COPY huggingface/ /opt/moshi/models/huggingface/
COPY start-lora-server.sh /usr/local/bin/start-lora-server

RUN chmod +x /usr/local/bin/start-lora-server && \
    test -f /opt/moshi/lora/lora.safetensors && \
    test -f /opt/moshi/lora/config.json

EXPOSE 8998

ENTRYPOINT ["/usr/local/bin/start-lora-server"]
