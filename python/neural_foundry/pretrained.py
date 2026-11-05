"""Declared local pretrained-model adapter for dense Qwen2 architectures (T059).

Provides offline-only pinned Hugging Face directory intake, site/config mapping,
tokenizer provenance verification, and native Rust surgeon/IR integration.
Downloads and remote custom code (`trust_remote_code`) are denied by default.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from . import surgeon
from .inspection import BoundedInspector, InspectionCap, SiteDescriptor
from .records import canonicalize, content_digest

QWEN2_FACTORY_ID = "hf.qwen2.dense"
QWEN2_FACTORY_VERSION = "1"
QWEN2_SMOKE_ADAPTER = "nf.qwen2_dense.v1"


class PretrainedAdapterError(RuntimeError):
    """Typed pretrained-model adapter error with stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _parse_safetensors(data: bytes) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    if len(data) < 8:
        raise PretrainedAdapterError("INVALID_SAFETENSORS", "Truncated Safetensors header")
    header_len = struct.unpack("<Q", data[:8])[0]
    if header_len == 0 or header_len > 1_048_576 or 8 + header_len > len(data):
        raise PretrainedAdapterError("INVALID_SAFETENSORS", "Invalid Safetensors header size")
    header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    if not isinstance(header, dict):
        raise PretrainedAdapterError("INVALID_SAFETENSORS", "Safetensors header must be an object")
    body = data[8 + header_len :]
    dtype_map = {"F32": "f32", "F64": "f64", "BF16": "bf16", "F16": "f16"}
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise PretrainedAdapterError("INVALID_SAFETENSORS", f"Invalid tensor entry {name!r}")
        raw_dtype = info.get("dtype")
        if raw_dtype not in dtype_map:
            raise PretrainedAdapterError(
                "UNSUPPORTED_DTYPE", f"Unsupported tensor dtype {raw_dtype!r} for {name!r}"
            )
        shape = tuple(int(dim) for dim in info.get("shape", ()))
        offsets = info.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > len(body)
        ):
            raise PretrainedAdapterError("INVALID_SAFETENSORS", f"Invalid offsets for {name!r}")
        tensors[name] = (dtype_map[raw_dtype], shape, body[offsets[0] : offsets[1]])
    return tensors


def _build_safetensors(tensors: Mapping[str, tuple[str, Sequence[int], bytes]]) -> bytes:
    header: dict[str, Any] = {}
    payload = bytearray()
    wire_dtype = {"f32": "F32", "f64": "F64"}
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": wire_dtype[dtype],
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    raw_header = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    pad = (8 - (len(raw_header) % 8)) % 8
    raw_header += b" " * pad
    return struct.pack("<Q", len(raw_header)) + raw_header + bytes(payload)


@dataclass(frozen=True)
class Qwen2IntakeBundle:
    """Verified local Qwen2 checkpoint intake bundle."""

    manifest: dict[str, Any]
    manifest_digest: str
    checkpoint_bytes: bytes
    checkpoint_digest: str
    assets: dict[str, bytes]
    config: dict[str, Any]
    tokenizer_provenance: dict[str, Any]
    sites: tuple[SiteDescriptor, ...]
    tensors: dict[str, tuple[tuple[int, ...], str, bytes]]
    qualification_stratum: str

    def load_native_source(self) -> surgeon.SourceHandle:
        """Load the verified Qwen2 checkpoint into the authoritative Rust surgeon."""
        return surgeon.load(self.manifest, self.checkpoint_bytes, self.assets)

    def inspector(self, cap: InspectionCap | None = None) -> BoundedInspector:
        """Create a bounded weight/activation inspector over the declared Qwen2 sites."""
        return BoundedInspector(
            model_id=self.manifest_digest,
            sites=self.sites,
            tensors=self.tensors,
            cap=cap,
        )

    def smoke_suite(
        self,
        probes: Sequence[Sequence[int]] = ((1, 2), (2, 3)),
        *,
        max_abs_output: float = 1_000.0,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "adapter": QWEN2_SMOKE_ADAPTER,
            "probes": [
                {
                    "input": [float(tok) for tok in probe],
                    "expected": None,
                    "absolute_tolerance": 0.0,
                    "max_abs_output": max_abs_output,
                }
                for probe in probes
            ],
        }

    def build_torch_module(self) -> Any:
        """Construct an isolated local PyTorch module for bounded activation capture."""
        import torch
        from torch import nn

        cfg = self.config
        h = int(cfg["hidden_size"])
        inter = int(cfg["intermediate_size"])
        layers = int(cfg["num_hidden_layers"])
        heads = int(cfg["num_attention_heads"])
        kv_heads = int(cfg["num_key_value_heads"])
        vocab = int(cfg["vocab_size"])
        head_dim = h // heads
        kv_dim = kv_heads * head_dim
        eps = float(cfg.get("rms_norm_eps", 1e-6))

        class _RMSNorm(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float64), requires_grad=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
                return x * rms * self.weight

        class _Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(h, h, bias=False, dtype=torch.float64)
                self.k_proj = nn.Linear(h, kv_dim, bias=False, dtype=torch.float64)
                self.v_proj = nn.Linear(h, kv_dim, bias=False, dtype=torch.float64)
                self.o_proj = nn.Linear(h, h, bias=False, dtype=torch.float64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                if kv_heads != heads:
                    repeat = heads // kv_heads
                    k = k.repeat_interleave(repeat, dim=-1)
                    v = v.repeat_interleave(repeat, dim=-1)
                scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
                seq_len = x.shape[-2]
                mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float64),
                    diagonal=1,
                )
                probs = torch.softmax(scores + mask, dim=-1)
                ctx = torch.matmul(probs, v)
                out: torch.Tensor = self.o_proj(ctx)
                return out

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(h, inter, bias=False, dtype=torch.float64)
                self.up_proj = nn.Linear(h, inter, bias=False, dtype=torch.float64)
                self.down_proj = nn.Linear(inter, h, bias=False, dtype=torch.float64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                gate = self.gate_proj(x)
                act = (gate * torch.sigmoid(gate)) * self.up_proj(x)
                out: torch.Tensor = self.down_proj(act)
                return out

        class _Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_layernorm = _RMSNorm(h)
                self.self_attn = _Attention()
                self.post_attention_layernorm = _RMSNorm(h)
                self.mlp = _MLP()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + self.self_attn(self.input_layernorm(x))
                x = x + self.mlp(self.post_attention_layernorm(x))
                return x

        class _Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab, h, dtype=torch.float64)
                self.layers = nn.ModuleList([_Layer() for _ in range(layers)])
                self.norm = _RMSNorm(h)

        class _Qwen2Mini(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Backbone()
                self.lm_head = nn.Linear(h, vocab, bias=False, dtype=torch.float64)

            def forward(self, token_ids: Any) -> torch.Tensor:
                if not isinstance(token_ids, torch.Tensor):
                    token_ids = torch.tensor(token_ids, dtype=torch.long)
                if token_ids.ndim == 1:
                    token_ids = token_ids.unsqueeze(0)
                x: torch.Tensor = self.model.embed_tokens(token_ids)
                for layer in self.model.layers:
                    x = layer(x)
                x = self.model.norm(x)
                out: torch.Tensor = self.lm_head(x)
                return out

        mod = _Qwen2Mini()
        state_dict = mod.state_dict()
        with torch.no_grad():
            for name, (shape, dtype, raw) in self.tensors.items():
                count = len(raw) // (8 if dtype == "f64" else 4)
                fmt = f"<{count}d" if dtype == "f64" else f"<{count}f"
                vals = struct.unpack(fmt, raw)
                tensor = torch.tensor(vals, dtype=torch.float64).reshape(shape)
                state_dict[name].copy_(tensor)
            if cfg.get("tie_word_embeddings", False):
                mod.lm_head.weight = mod.model.embed_tokens.weight
            for p in mod.parameters():
                p.requires_grad_(False)
        mod.eval()
        return mod


def write_mini_qwen2_checkpoint(
    directory: str | PathLike[str],
    *,
    hidden_size: int = 4,
    intermediate_size: int = 8,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 2,
    num_key_value_heads: int = 1,
    vocab_size: int = 16,
    tie_word_embeddings: bool = True,
    dtype: str = "f64",
    qualification_stratum: str = "mini_config_offline_fixture",
) -> Path:
    """Create a deterministic offline mini Qwen2 checkpoint directory without network calls."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    head_dim = hidden_size // num_attention_heads
    kv_dim = num_key_value_heads * head_dim

    def make_bytes(shape: Sequence[int], scale: float = 0.125) -> bytes:
        count = 1
        for dim in shape:
            count *= dim
        vals = [((idx % 7) - 3) * scale + 0.05 for idx in range(count)]
        if dtype == "f64":
            return struct.pack(f"<{count}d", *vals)
        return struct.pack(f"<{count}f", *vals)

    def ones_bytes(dim: int) -> bytes:
        if dtype == "f64":
            return struct.pack(f"<{dim}d", *([1.0] * dim))
        return struct.pack(f"<{dim}f", *([1.0] * dim))

    embed_bytes = make_bytes((vocab_size, hidden_size), 0.1)
    tensors: dict[str, tuple[str, Sequence[int], bytes]] = {
        "model.embed_tokens.weight": (dtype, (vocab_size, hidden_size), embed_bytes),
        "model.norm.weight": (dtype, (hidden_size,), ones_bytes(hidden_size)),
        "lm_head.weight": (
            dtype,
            (vocab_size, hidden_size),
            embed_bytes if tie_word_embeddings else make_bytes((vocab_size, hidden_size), 0.15),
        ),
    }
    for layer in range(num_hidden_layers):
        p = f"model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = (dtype, (hidden_size,), ones_bytes(hidden_size))
        tensors[f"{p}.self_attn.q_proj.weight"] = (
            dtype,
            (hidden_size, hidden_size),
            make_bytes((hidden_size, hidden_size), 0.1),
        )
        tensors[f"{p}.self_attn.k_proj.weight"] = (
            dtype,
            (kv_dim, hidden_size),
            make_bytes((kv_dim, hidden_size), 0.1),
        )
        tensors[f"{p}.self_attn.v_proj.weight"] = (
            dtype,
            (kv_dim, hidden_size),
            make_bytes((kv_dim, hidden_size), 0.1),
        )
        tensors[f"{p}.self_attn.o_proj.weight"] = (
            dtype,
            (hidden_size, hidden_size),
            make_bytes((hidden_size, hidden_size), 0.1),
        )
        tensors[f"{p}.post_attention_layernorm.weight"] = (
            dtype,
            (hidden_size,),
            ones_bytes(hidden_size),
        )
        tensors[f"{p}.mlp.gate_proj.weight"] = (
            dtype,
            (intermediate_size, hidden_size),
            make_bytes((intermediate_size, hidden_size), 0.1),
        )
        tensors[f"{p}.mlp.up_proj.weight"] = (
            dtype,
            (intermediate_size, hidden_size),
            make_bytes((intermediate_size, hidden_size), 0.1),
        )
        tensors[f"{p}.mlp.down_proj.weight"] = (
            dtype,
            (hidden_size, intermediate_size),
            make_bytes((hidden_size, intermediate_size), 0.1),
        )

    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_act": "silu",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": tie_word_embeddings,
        "trust_remote_code": False,
    }
    vocab = {f"<tok_{i}>": i for i in range(vocab_size)}
    tokenizer = {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "special": True},
            {"id": 1, "content": "<|im_start|>", "special": True},
            {"id": 2, "content": "<|im_end|>", "special": True},
        ],
    }
    provenance = {
        "schema_version": 1,
        "source_revision": "sha256:" + "a" * 64,
        "license": "Apache-2.0",
        "usage_rights": "Apache-2.0 local test fixture",
        "qualification_stratum": qualification_stratum,
    }
    (root / "config.json").write_bytes(canonicalize(config))
    (root / "tokenizer.json").write_bytes(canonicalize(tokenizer))
    (root / "provenance.json").write_bytes(canonicalize(provenance))
    (root / "model.safetensors").write_bytes(_build_safetensors(tensors))
    return root


def intake_local_qwen2(
    source: str | PathLike[str],
    *,
    allow_download: bool = False,
    trust_remote_code: bool = False,
    expected_revision: str | None = None,
) -> Qwen2IntakeBundle:
    """Validate and load a pinned local Qwen2 checkpoint directory without network access."""
    if allow_download:
        raise PretrainedAdapterError(
            "DOWNLOAD_DISABLED",
            "Network model downloads are disabled by default; supply an explicit local directory",
        )
    if trust_remote_code:
        raise PretrainedAdapterError(
            "REMOTE_CODE_FORBIDDEN",
            "trust_remote_code is forbidden; only static registered Qwen2 architectures are supported",
        )
    source_str = str(source)
    if "://" in source_str or (not Path(source).is_absolute() and "/" in source_str and not Path(source).exists()):
        raise PretrainedAdapterError(
            "DOWNLOAD_DISABLED",
            f"Remote hub identifier or URL {source_str!r} is rejected; local directory required",
        )
    root = Path(source).resolve()
    if not root.is_dir():
        raise PretrainedAdapterError("MISSING_DIRECTORY", f"Checkpoint directory not found: {root}")

    for forbidden_ext in ("*.bin", "*.pt", "*.pth", "*.pkl", "*.py"):
        if list(root.glob(forbidden_ext)):
            raise PretrainedAdapterError(
                "UNSAFE_ARTIFACT",
                f"Executable or pickle model artifact matching {forbidden_ext} is forbidden",
            )

    config_path = root / "config.json"
    tokenizer_path = root / "tokenizer.json"
    provenance_path = root / "provenance.json"
    checkpoint_path = root / "model.safetensors"
    for required in (config_path, tokenizer_path, provenance_path, checkpoint_path):
        if not required.is_file():
            raise PretrainedAdapterError("MISSING_ASSET", f"Required asset missing: {required.name}")

    config_bytes = canonicalize(config_path.read_bytes())
    config = json.loads(config_bytes)
    if config.get("trust_remote_code") is True or "auto_map" in config:
        raise PretrainedAdapterError(
            "REMOTE_CODE_FORBIDDEN", "Config requests remote custom code execution"
        )
    if (
        config.get("model_type") != "qwen2"
        or config.get("architectures") != ["Qwen2ForCausalLM"]
        or config.get("hidden_act", "silu") != "silu"
    ):
        raise PretrainedAdapterError(
            "UNSUPPORTED_ARCHITECTURE",
            "Only dense Qwen2ForCausalLM with silu activation is supported by this adapter",
        )

    h = int(config.get("hidden_size", 0))
    inter = int(config.get("intermediate_size", 0))
    layers = int(config.get("num_hidden_layers", 0))
    heads = int(config.get("num_attention_heads", 0))
    kv_heads = int(config.get("num_key_value_heads", 0))
    vocab_size = int(config.get("vocab_size", 0))
    tie = bool(config.get("tie_word_embeddings", False))
    if (
        h <= 0
        or inter <= 0
        or layers <= 0
        or heads <= 0
        or kv_heads <= 0
        or kv_heads > heads
        or h % heads != 0
        or heads % kv_heads != 0
        or vocab_size <= 0
    ):
        raise PretrainedAdapterError(
            "UNSUPPORTED_CONFIG", "Invalid Qwen2 dimensions or GQA head layout"
        )

    tokenizer_bytes = canonicalize(tokenizer_path.read_bytes())
    tokenizer_data = json.loads(tokenizer_bytes)
    tok_vocab = tokenizer_data.get("model", {}).get("vocab", {})
    if not isinstance(tok_vocab, dict) or len(tok_vocab) > vocab_size:
        raise PretrainedAdapterError(
            "TOKENIZER_MISMATCH", "Tokenizer vocabulary is invalid or exceeds model vocab_size"
        )
    tokenizer_digest = content_digest(tokenizer_bytes)

    provenance_bytes = canonicalize(provenance_path.read_bytes())
    provenance = json.loads(provenance_bytes)
    if expected_revision is not None and provenance.get("source_revision") != expected_revision:
        raise PretrainedAdapterError(
            "REVISION_MISMATCH", "Pinned source_revision does not match expected_revision"
        )
    stratum = str(provenance.get("qualification_stratum", ""))
    if stratum not in {
        "mini_config_offline_fixture",
        "user_supplied_local_pretrained_checkpoint",
    }:
        raise PretrainedAdapterError(
            "INVALID_PROVENANCE",
            "qualification_stratum must distinguish mini_config_offline_fixture from user_supplied_local_pretrained_checkpoint",
        )

    checkpoint_bytes = checkpoint_path.read_bytes()
    parsed_tensors = _parse_safetensors(checkpoint_bytes)
    head_dim = h // heads
    kv_dim = kv_heads * head_dim

    expected_shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab_size, h),
        "model.norm.weight": (h,),
        "lm_head.weight": (vocab_size, h),
    }
    sites: list[SiteDescriptor] = [
        SiteDescriptor(
            name="embed_tokens",
            kind="weight",
            tensor_name="model.embed_tokens.weight",
            module_path="model.embed_tokens",
            shape=(vocab_size, h),
            dtype=parsed_tensors.get("model.embed_tokens.weight", ("f64", (), b""))[0],
        )
    ]
    for layer in range(layers):
        p = f"model.layers.{layer}"
        expected_shapes[f"{p}.input_layernorm.weight"] = (h,)
        expected_shapes[f"{p}.self_attn.q_proj.weight"] = (h, h)
        expected_shapes[f"{p}.self_attn.k_proj.weight"] = (kv_dim, h)
        expected_shapes[f"{p}.self_attn.v_proj.weight"] = (kv_dim, h)
        expected_shapes[f"{p}.self_attn.o_proj.weight"] = (h, h)
        expected_shapes[f"{p}.post_attention_layernorm.weight"] = (h,)
        expected_shapes[f"{p}.mlp.gate_proj.weight"] = (inter, h)
        expected_shapes[f"{p}.mlp.up_proj.weight"] = (inter, h)
        expected_shapes[f"{p}.mlp.down_proj.weight"] = (h, inter)

        layer_dtype = parsed_tensors.get("model.embed_tokens.weight", ("f64", (), b""))[0]
        sites.extend(
            [
                SiteDescriptor(
                    name=f"layers.{layer}.self_attn",
                    kind="attention",
                    tensor_name=f"{p}.self_attn.o_proj.weight",
                    module_path=f"{p}.self_attn",
                    shape=(h, h),
                    dtype=layer_dtype,
                ),
                SiteDescriptor(
                    name=f"layers.{layer}.mlp.down_proj",
                    kind="mlp",
                    tensor_name=f"{p}.mlp.down_proj.weight",
                    module_path=f"{p}.mlp",
                    shape=(h, inter),
                    dtype=layer_dtype,
                ),
                SiteDescriptor(
                    name=f"layers.{layer}.residual",
                    kind="residual",
                    tensor_name=None,
                    module_path=p,
                    shape=(h,),
                    dtype=layer_dtype,
                ),
            ]
        )
    sites.append(
        SiteDescriptor(
            name="lm_head",
            kind="weight",
            tensor_name="lm_head.weight",
            module_path="lm_head",
            shape=(vocab_size, h),
            dtype=parsed_tensors.get("lm_head.weight", ("f64", (), b""))[0],
        )
    )

    if set(parsed_tensors) != set(expected_shapes):
        raise PretrainedAdapterError(
            "INVENTORY_MISMATCH",
            "Checkpoint tensor names do not match declared Qwen2 architecture inventory",
        )

    parameters: dict[str, Any] = {}
    tensor_map: dict[str, tuple[tuple[int, ...], str, bytes]] = {}
    for name, exp_shape in sorted(expected_shapes.items()):
        dtype, actual_shape, raw = parsed_tensors[name]
        if actual_shape != exp_shape:
            raise PretrainedAdapterError(
                "SHAPE_MISMATCH",
                f"Tensor {name!r} has shape {actual_shape}, expected {exp_shape}",
            )
        if tie and name in {"model.embed_tokens.weight", "lm_head.weight"}:
            group = "shared.embed_lm_head"
        else:
            group = name
        parameters[name] = {
            "shape": list(actual_shape),
            "dtype": dtype,
            "digest": _sha256(raw),
            "storage_group": group,
        }
        tensor_map[name] = (actual_shape, dtype, raw)

    if tie and parameters["model.embed_tokens.weight"]["digest"] != parameters["lm_head.weight"]["digest"]:
        raise PretrainedAdapterError(
            "TIED_WEIGHT_MISMATCH",
            "tie_word_embeddings=True requires identical bytes for embed_tokens and lm_head",
        )

    param_set_entries = [
        {
            "name": name,
            "shape": spec["shape"],
            "dtype": spec["dtype"],
            "content_digest": spec["digest"],
        }
        for name, spec in sorted(parameters.items())
    ]
    parameter_digest = content_digest(
        canonicalize({"schema_version": 1, "kind": "tensor_set", "tensors": param_set_entries})
    )
    config_digest = content_digest(config_bytes)
    checkpoint_digest = _sha256(checkpoint_bytes)
    provenance_digest = content_digest(provenance_bytes)

    manifest = {
        "schema_version": 1,
        "architecture": {
            "id": QWEN2_FACTORY_ID,
            "version": QWEN2_FACTORY_VERSION,
            "config": config_digest,
        },
        "parameters": parameters,
        "buffers": {},
        "parameter_digest": parameter_digest,
        "format": "dense_safetensors",
        "packing_policy": None,
        "assets": {
            "checkpoint": checkpoint_digest,
            "config": config_digest,
            "tokenizer": tokenizer_digest,
            "provenance": provenance_digest,
        },
        "usage_rights": str(provenance["usage_rights"]),
    }
    manifest_bytes = canonicalize(manifest)
    manifest_digest = content_digest(manifest_bytes)
    assets = {
        "config": config_bytes,
        "tokenizer": tokenizer_bytes,
        "provenance": provenance_bytes,
    }
    return Qwen2IntakeBundle(
        manifest=manifest,
        manifest_digest=manifest_digest,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_digest=checkpoint_digest,
        assets=assets,
        config=config,
        tokenizer_provenance={
            "tokenizer_digest": tokenizer_digest,
            "vocab_entries": len(tok_vocab),
            "source_revision": provenance["source_revision"],
            "license": provenance["license"],
            "qualification_stratum": stratum,
        },
        sites=tuple(sites),
        tensors=tensor_map,
        qualification_stratum=stratum,
    )
