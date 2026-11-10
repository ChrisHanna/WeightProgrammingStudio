FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_ALLOW_INSECURE_SERIALIZATION=0

COPY requirements.txt /tmp/nf-reload-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --no-deps --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/nf-reload-requirements.txt

ENTRYPOINT ["python"]