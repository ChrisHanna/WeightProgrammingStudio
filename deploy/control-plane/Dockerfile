FROM rust:1.93-slim-bookworm@sha256:5b9332190bb3b9ece73b810cd1f1e9f06343b294ce184bcb067f0747d7d333ea AS rust
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS builder
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV PATH="/usr/local/cargo/bin:${PATH}" RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo RUSTUP_TOOLCHAIN=1.93.1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir maturin==1.16.0
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml pyproject.toml README.md LICENSE NOTICE ./
COPY crates ./crates
COPY protocols/reference-audit-deadline-v2.json ./protocols/reference-audit-deadline-v2.json
COPY python ./python
RUN --mount=type=cache,target=/src/target \
    --mount=type=cache,target=/usr/local/cargo/registry \
    maturin build --locked --release --out /wheels --interpreter python

FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m pip install --no-cache-dir torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && groupadd --gid 10001 controlplane \
    && useradd --uid 10001 --gid 10001 --no-create-home controlplane \
    && mkdir -p /var/lib/control-plane /opt/control-plane \
    && chown 10001:10001 /var/lib/control-plane
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl \
    && python -m pip install --no-cache-dir 'neural-foundry[control-plane]' \
    && rm -rf /wheels
COPY deploy/control-plane/scripts/entrypoint.py deploy/control-plane/scripts/healthcheck.py /opt/control-plane/
WORKDIR /var/lib/control-plane
ENTRYPOINT ["python", "/opt/control-plane/entrypoint.py"]
