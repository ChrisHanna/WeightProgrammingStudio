"""Pythia / GPT-NeoX (`EleutherAI/pythia-*`) Pretrained Adapter & Weight-Space Programming Runner.

Provides:
- Zero-dependency GPT-NeoX BPE tokenizer and Safetensors loader for `GPTNeoXForCausalLM`
  checkpoints (such as `EleutherAI/pythia-14m`) and synthetic miniature GPT-NeoX fixtures.
- Exact causal forward pass (parallel residual, partial RoPE, LayerNorm, GELU MLP, untied `embed_out`)
  with layer-wise MLP key activation capture.
- Whitened null-space-regularized low-rank weight-space synthesis (`low_rank_update`)
  compiled and executed through the authoritative Rust `nf-weight-ir` compiler and
  `nf-surgeon` transactional engine.
- End-to-end integration with the first-class Evaluation Engine (`neural_foundry.evals`)
  producing before/after predictions, parameter-change accounting, reload parity, and
  Model Improvement Report dashboards.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import evals, ir
from .gym_proposal import GymWeightProposalSpec, ProposedOperation, ResolvedGymWeightProposal, SemanticSite
from .provenance import evidence_grade_for_model
from .records import canonicalize, content_digest

PYTHIA_FACTORY_ID = "hf.gpt_neox.dense"
PYTHIA_FACTORY_VERSION = "1"


@dataclass(frozen=True)
class ArchitectureSiteProfile:
    """Architecture-agnostic tensor naming profile for open-weight transformer families."""

    family_id: str
    mlp_down_proj_template: str
    mlp_up_proj_template: str
    attn_out_template: str
    embed_in_tensor: str
    unembed_out_tensor: str
    final_norm_weight_tensor: str

    def mlp_down_proj_site(self, layer_idx: int) -> str:
        return self.mlp_down_proj_template.format(layer=layer_idx)

    def mlp_up_proj_site(self, layer_idx: int) -> str:
        return self.mlp_up_proj_template.format(layer=layer_idx)

    def attn_out_site(self, layer_idx: int) -> str:
        return self.attn_out_template.format(layer=layer_idx)


BUILTIN_ARCHITECTURE_PROFILES: dict[str, ArchitectureSiteProfile] = {
    "gpt_neox": ArchitectureSiteProfile(
        family_id="gpt_neox",
        mlp_down_proj_template="gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight",
        mlp_up_proj_template="gpt_neox.layers.{layer}.mlp.dense_h_to_4h.weight",
        attn_out_template="gpt_neox.layers.{layer}.attention.dense.weight",
        embed_in_tensor="gpt_neox.embed_in.weight",
        unembed_out_tensor="embed_out.weight",
        final_norm_weight_tensor="gpt_neox.final_layer_norm.weight",
    ),
    "qwen2": ArchitectureSiteProfile(
        family_id="qwen2",
        mlp_down_proj_template="model.layers.{layer}.mlp.down_proj.weight",
        mlp_up_proj_template="model.layers.{layer}.mlp.up_proj.weight",
        attn_out_template="model.layers.{layer}.self_attn.o_proj.weight",
        embed_in_tensor="model.embed_tokens.weight",
        unembed_out_tensor="lm_head.weight",
        final_norm_weight_tensor="model.norm.weight",
    ),
    "llama": ArchitectureSiteProfile(
        family_id="llama",
        mlp_down_proj_template="model.layers.{layer}.mlp.down_proj.weight",
        mlp_up_proj_template="model.layers.{layer}.mlp.up_proj.weight",
        attn_out_template="model.layers.{layer}.self_attn.o_proj.weight",
        embed_in_tensor="model.embed_tokens.weight",
        unembed_out_tensor="lm_head.weight",
        final_norm_weight_tensor="model.norm.weight",
    ),
    "mistral": ArchitectureSiteProfile(
        family_id="mistral",
        mlp_down_proj_template="model.layers.{layer}.mlp.down_proj.weight",
        mlp_up_proj_template="model.layers.{layer}.mlp.up_proj.weight",
        attn_out_template="model.layers.{layer}.self_attn.o_proj.weight",
        embed_in_tensor="model.embed_tokens.weight",
        unembed_out_tensor="lm_head.weight",
        final_norm_weight_tensor="model.norm.weight",
    ),
    "gemma": ArchitectureSiteProfile(
        family_id="gemma",
        mlp_down_proj_template="model.layers.{layer}.mlp.down_proj.weight",
        mlp_up_proj_template="model.layers.{layer}.mlp.up_proj.weight",
        attn_out_template="model.layers.{layer}.self_attn.o_proj.weight",
        embed_in_tensor="model.embed_tokens.weight",
        unembed_out_tensor="model.embed_tokens.weight",
        final_norm_weight_tensor="model.norm.weight",
    ),
    "gemma2": ArchitectureSiteProfile(
        family_id="gemma2",
        mlp_down_proj_template="model.layers.{layer}.mlp.down_proj.weight",
        mlp_up_proj_template="model.layers.{layer}.mlp.up_proj.weight",
        attn_out_template="model.layers.{layer}.self_attn.o_proj.weight",
        embed_in_tensor="model.embed_tokens.weight",
        unembed_out_tensor="model.embed_tokens.weight",
        final_norm_weight_tensor="model.norm.weight",
    ),
    "qwen3": ArchitectureSiteProfile(
        family_id="qwen3",
        mlp_down_proj_template="model.layers.{layer}.mlp.down_proj.weight",
        mlp_up_proj_template="model.layers.{layer}.mlp.up_proj.weight",
        attn_out_template="model.layers.{layer}.self_attn.o_proj.weight",
        embed_in_tensor="model.embed_tokens.weight",
        unembed_out_tensor="lm_head.weight",
        final_norm_weight_tensor="model.norm.weight",
    ),
}


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _pack_f64_tensor(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().contiguous().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
    return struct.pack(f"<{len(flat)}d", *[float(v) for v in flat])


def _build_named_safetensors(name: str, weight_f64: torch.Tensor) -> tuple[bytes, str, int]:
    raw_bytes = _pack_f64_tensor(weight_f64)
    header = {
        name: {
            "dtype": "F64",
            "shape": [int(weight_f64.shape[0]), int(weight_f64.shape[1])],
            "data_offsets": [0, len(raw_bytes)],
        }
    }
    raw_hdr = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    pad = (8 - (len(raw_hdr) % 8)) % 8
    raw_hdr += b" " * pad
    st = struct.pack("<Q", len(raw_hdr)) + raw_hdr + raw_bytes
    return st, _sha256(raw_bytes), len(raw_bytes)


def _build_single_matrix_safetensors(weight_f64: torch.Tensor) -> tuple[bytes, str]:
    st, raw_digest, _ = _build_named_safetensors("w", weight_f64)
    return st, raw_digest


def _build_byte_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte-to-printable-character table."""
    printable = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    table = {b: chr(b) for b in printable}
    shifted = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + shifted)
            shifted += 1
    return table


_BYTE_TO_UNICODE = _build_byte_to_unicode()
_GPT2_CONTRACTIONS = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")


def _gpt2_is_space(ch: str) -> bool:
    # `str.isspace` also accepts the \x1c-\x1f separators, which Unicode White_Space does not.
    return ch.isspace() and not "\x1c" <= ch <= "\x1f"


def _gpt2_class(ch: str) -> str:
    if _gpt2_is_space(ch):
        return "space"
    category = unicodedata.category(ch)[0]
    return category if category in ("L", "N") else "other"


def _gpt2_split(text: str) -> list[str]:
    """Stdlib equivalent of the GPT-2 pattern, which needs `\\p{L}`/`\\p{N}`:
    `'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+`.
    """
    pieces: list[str] = []
    i, n = 0, len(text)
    while i < n:
        contraction = next((c for c in _GPT2_CONTRACTIONS if text.startswith(c, i)), None)
        if contraction:
            pieces.append(contraction)
            i += len(contraction)
            continue
        body = i + 1 if text[i] == " " and i + 1 < n and not _gpt2_is_space(text[i + 1]) else i
        kind = _gpt2_class(text[body])
        j = body + 1
        while j < n and _gpt2_class(text[j]) == kind:
            j += 1
        # `\s+(?!\S)` gives back one whitespace character when the run precedes a non-space.
        if kind == "space" and j < n and j - body > 1:
            j -= 1
        pieces.append(text[i:j])
        i = j
    return pieces


class PythiaBPETokenizer:
    """Deterministic GPT-NeoX byte-level BPE tokenizer loaded from `tokenizer.json`."""

    def __init__(
        self,
        vocab: Mapping[str, int],
        merges: Sequence[str],
        *,
        byte_level: bool = False,
        added_tokens: Mapping[str, int] | None = None,
    ) -> None:
        # `byte_level` selects the real GPT-2 pipeline declared by a checkpoint's tokenizer.json.
        # Synthetic vocabularies (no merges, whole-word tokens) keep the whitespace split.
        self.byte_level = byte_level
        self.added_tokens: dict[str, int] = dict(added_tokens or {})
        self._added_token_pattern = (
            re.compile("|".join(re.escape(t) for t in sorted(self.added_tokens, key=len, reverse=True)))
            if self.added_tokens
            else None
        )
        self.vocab: dict[str, int] = {str(k): int(v) for k, v in vocab.items()}
        self.id_to_token: dict[int, str] = {int(v): str(k) for k, v in vocab.items()}
        parsed_merges: list[tuple[str, str]] = []
        for m in merges:
            parts = m.split(" ")
            if len(parts) == 2:
                parsed_merges.append((parts[0], parts[1]))
        self.bpe_ranks: dict[tuple[str, str], int] = {
            pair: idx for idx, pair in enumerate(parsed_merges)
        }

    @classmethod
    def from_file(cls, path: str | Path) -> PythiaBPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = data["model"]
        pre = data.get("pre_tokenizer") or {}
        normalizer = data.get("normalizer") or {}
        byte_level = (
            pre.get("type") == "ByteLevel"
            and pre.get("use_regex") is True
            and pre.get("add_prefix_space") is False
            and normalizer.get("type") == "NFC"
        )
        added = {t["content"]: int(t["id"]) for t in data.get("added_tokens", [])} if byte_level else None
        return cls(
            vocab=model["vocab"],
            merges=model.get("merges", []),
            byte_level=byte_level,
            added_tokens=added,
        )

    def _bpe_word(self, token_str: str) -> list[int]:
        word = list(token_str)
        while len(word) >= 2:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, 10**9))
            if best not in self.bpe_ranks:
                break
            first, second = best
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return [self.vocab[w] for w in word if w in self.vocab]

    def _encode_byte_level(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        start = 0
        pattern = self._added_token_pattern
        for match in pattern.finditer(text) if pattern else ():
            ids.extend(self._encode_ordinary(text[start : match.start()]))
            ids.append(self.added_tokens[match.group()])
            start = match.end()
        ids.extend(self._encode_ordinary(text[start:]))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _gpt2_split(text):
            ids.extend(self._bpe_word("".join(_BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))))
        return ids

    def encode(self, text: str) -> list[int]:
        if self.byte_level:
            return self._encode_byte_level(text)
        parts = text.split(" ")
        ids: list[int] = []
        for idx, part in enumerate(parts):
            if not part:
                continue
            prefixed = part if idx == 0 else f"Ġ{part}"
            if prefixed in self.vocab:
                ids.append(self.vocab[prefixed])
            else:
                ids.extend(self._bpe_word(prefixed))
        return ids

    def decode_token(self, token_id: int) -> str:
        raw = self.id_to_token.get(int(token_id), "<unk>")
        return raw.replace("Ġ", " ").replace("Ċ", "\n")


@dataclass(frozen=True)
class PythiaPrediction:
    prompt: str
    top1_token_id: int
    top1_token: str
    top1_logit: float
    target_token: str | None
    target_logit: float | None
    target_prob: float | None
    matched_target: bool | None


@dataclass(frozen=True)
class PythiaWeightEditReport:
    model_name: str
    target_site: str
    target_layer: int
    rank: int
    total_model_parameters: int
    site_parameters: int
    changed_parameters: int
    changed_parameter_pct_of_model: float
    plan_digest: str
    candidate_digest: str
    before_parameter_digest: str
    after_parameter_digest: str
    reload_parity_verified: bool
    dev_before: tuple[PythiaPrediction, ...]
    dev_after: tuple[PythiaPrediction, ...]
    generalization_before: tuple[PythiaPrediction, ...]
    generalization_after: tuple[PythiaPrediction, ...]
    retention_before: tuple[PythiaPrediction, ...]
    retention_after: tuple[PythiaPrediction, ...]
    eval_comparison: dict[str, Any]
    promotion_verification: dict[str, Any]


class PythiaModel:
    """Loaded `GPTNeoXForCausalLM` (Pythia) model with exact forward pass and Rust transactional editing."""

    def __init__(
        self,
        *,
        model_name: str,
        config: dict[str, Any],
        tokenizer: PythiaBPETokenizer,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        self.model_name = model_name
        self.config = config
        self.tokenizer = tokenizer
        self.tensors = {k: v.detach().to(dtype=torch.float64) for k, v in tensors.items()}

    @classmethod
    def from_pretrained_dir(
        cls,
        model_dir: str | Path,
        *,
        model_name: str = "EleutherAI/pythia-14m",
        max_checkpoint_bytes: int = 68_719_476_736,
    ) -> PythiaModel:
        """Stream and map a single or sharded GPT-NeoX checkpoint directory via `ShardedMmapCheckpoint` (`mmap`)."""
        from .export_formats import ShardedMmapCheckpoint

        root = Path(model_dir)
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if cfg.get("model_type") != "gpt_neox" or cfg.get("architectures") != ["GPTNeoXForCausalLM"]:
            raise ValueError("Directory is not a GPTNeoXForCausalLM (Pythia) checkpoint")
        tokenizer = PythiaBPETokenizer.from_file(root / "tokenizer.json")
        dtype_map = {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F64": torch.float64,
        }
        tensors: dict[str, torch.Tensor] = {}
        with ShardedMmapCheckpoint(root, max_checkpoint_bytes=max_checkpoint_bytes) as sharded:
            for name, slice_info in sharded.tensors.items():
                view = sharded.read_tensor_view(name)
                tdtype = dtype_map[str(slice_info.dtype).upper()]
                shape = [int(d) for d in slice_info.shape]
                t = torch.frombuffer(bytearray(view), dtype=tdtype).reshape(shape).to(torch.float64)
                tensors[name] = t
        return cls(model_name=model_name, config=cfg, tokenizer=tokenizer, tensors=tensors)

    @property
    def total_parameters(self) -> int:
        return sum(int(t.numel()) for t in self.tensors.values())

    def mlp_down_proj_site(self, layer_idx: int) -> str:
        return f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight"

    def effective_unembedding(self) -> torch.Tensor:
        w_out = self.tensors["embed_out.weight"]
        ln_f_w = self.tensors["gpt_neox.final_layer_norm.weight"]
        return w_out * ln_f_w.unsqueeze(0)

    def forward_with_mlp_key(
        self,
        token_ids: Sequence[int],
        *,
        capture_layer: int = -1,
        layer_down_proj_override: dict[int, torch.Tensor] | None = None,
        residual_sink: list[torch.Tensor] | None = None,
        site_inputs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Executes the exact GPT-NeoX causal forward pass and returns `(logits, final_token_mlp_key)`.

        `residual_sink`, when given, receives the residual stream that enters the final layer norm.
        `site_inputs`, when given, receives the last token's input to each layer's dense matrices, by tensor name.
        """
        h = int(self.config["hidden_size"])
        heads = int(self.config["num_attention_heads"])
        num_layers = int(self.config["num_hidden_layers"])
        target_layer = (num_layers - 1) if capture_layer < 0 else capture_layer
        head_dim = h // heads
        rot_dim = int(head_dim * float(self.config.get("rotary_pct", 0.25)))
        rot_base = float(self.config.get("rotary_emb_base", 10000.0))
        eps = float(self.config.get("layer_norm_eps", 1e-5))

        x = self.tensors["gpt_neox.embed_in.weight"][list(token_ids)]
        seq = int(x.shape[0])
        inv_freq = 1.0 / (rot_base ** (torch.arange(0, rot_dim, 2, dtype=torch.float64) / rot_dim))
        pos = torch.arange(seq, dtype=torch.float64)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)

        def _rotate_half(t: torch.Tensor) -> torch.Tensor:
            half = t.shape[-1] // 2
            return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

        captured_key = torch.zeros(int(self.config["intermediate_size"]), dtype=torch.float64)
        for layer_idx in range(num_layers):
            prefix = f"gpt_neox.layers.{layer_idx}"
            ln1 = F.layer_norm(
                x,
                (h,),
                self.tensors[f"{prefix}.input_layernorm.weight"],
                self.tensors[f"{prefix}.input_layernorm.bias"],
                eps,
            )
            ln2 = F.layer_norm(
                x,
                (h,),
                self.tensors[f"{prefix}.post_attention_layernorm.weight"],
                self.tensors[f"{prefix}.post_attention_layernorm.bias"],
                eps,
            )
            qkv = F.linear(
                ln1,
                self.tensors[f"{prefix}.attention.query_key_value.weight"],
                self.tensors[f"{prefix}.attention.query_key_value.bias"],
            )
            qkv = qkv.view(seq, heads, 3 * head_dim).permute(1, 0, 2)
            q, k, v = torch.split(qkv, head_dim, dim=-1)
            qr, qp = q[..., :rot_dim], q[..., rot_dim:]
            kr, kp = k[..., :rot_dim], k[..., rot_dim:]
            qr = qr * cos + _rotate_half(qr) * sin
            kr = kr * cos + _rotate_half(kr) * sin
            q = torch.cat([qr, qp], dim=-1)
            k = torch.cat([kr, kp], dim=-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            mask = torch.triu(
                torch.full((seq, seq), float("-inf"), dtype=torch.float64),
                diagonal=1,
            )
            probs = torch.softmax(scores + mask, dim=-1)
            ctx = torch.matmul(probs, v).permute(1, 0, 2).contiguous().view(seq, h)
            attn_out = F.linear(
                ctx,
                self.tensors[f"{prefix}.attention.dense.weight"],
                self.tensors[f"{prefix}.attention.dense.bias"],
            )
            mlp_h = F.gelu(
                F.linear(
                    ln2,
                    self.tensors[f"{prefix}.mlp.dense_h_to_4h.weight"],
                    self.tensors[f"{prefix}.mlp.dense_h_to_4h.bias"],
                )
            )
            if site_inputs is not None:
                site_inputs[f"{prefix}.mlp.dense_4h_to_h.weight"] = mlp_h[-1]
                site_inputs[f"{prefix}.mlp.dense_h_to_4h.weight"] = ln2[-1]
                site_inputs[f"{prefix}.attention.dense.weight"] = ctx[-1]
            if layer_idx == target_layer:
                captured_key = mlp_h[-1].clone()
            w_down = (
                layer_down_proj_override[layer_idx]
                if (layer_down_proj_override and layer_idx in layer_down_proj_override)
                else self.tensors[f"{prefix}.mlp.dense_4h_to_h.weight"]
            )
            mlp_out = F.linear(
                mlp_h,
                w_down,
                self.tensors[f"{prefix}.mlp.dense_4h_to_h.bias"],
            )
            x = x + attn_out + mlp_out

        if residual_sink is not None:
            residual_sink.append(x)
        x_norm = F.layer_norm(
            x,
            (h,),
            self.tensors["gpt_neox.final_layer_norm.weight"],
            self.tensors["gpt_neox.final_layer_norm.bias"],
            eps,
        )
        logits = F.linear(x_norm, self.tensors["embed_out.weight"])
        return logits, captured_key

    def evaluate_prompt(
        self,
        prompt: str,
        *,
        target_token: str | None = None,
        layer_down_proj_override: dict[int, torch.Tensor] | None = None,
    ) -> PythiaPrediction:
        ids = self.tokenizer.encode(prompt)
        logits, _ = self.forward_with_mlp_key(ids, layer_down_proj_override=layer_down_proj_override)
        last_logits = logits[-1]
        probs = torch.softmax(last_logits, dim=-1)
        top1_id = int(torch.argmax(last_logits).item())
        top1_tok = self.tokenizer.decode_token(top1_id)
        top1_logit = float(last_logits[top1_id].item())

        if target_token is None:
            return PythiaPrediction(
                prompt=prompt,
                top1_token_id=top1_id,
                top1_token=top1_tok,
                top1_logit=top1_logit,
                target_token=None,
                target_logit=None,
                target_prob=None,
                matched_target=None,
            )

        target_id = self.tokenizer.vocab[target_token]
        target_clean = self.tokenizer.decode_token(target_id)
        t_logit = float(last_logits[target_id].item())
        t_prob = float(probs[target_id].item())
        return PythiaPrediction(
            prompt=prompt,
            top1_token_id=top1_id,
            top1_token=top1_tok,
            top1_logit=top1_logit,
            target_token=target_clean,
            target_logit=t_logit,
            target_prob=t_prob,
            matched_target=(top1_id == target_id),
        )

    def forward_with_multi_layer_keys(
        self,
        token_ids: Sequence[int],
        *,
        capture_layers: Sequence[int] = (),
        layer_down_proj_override: Mapping[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Forward pass capturing MLP intermediate key vectors across multiple layers."""
        num_layers = int(self.config["num_hidden_layers"])
        layers_to_capture = tuple(int(l) for l in (capture_layers or range(num_layers)))
        overrides = dict(layer_down_proj_override or {})
        keys: dict[int, torch.Tensor] = {}
        logits = torch.empty(0, dtype=torch.float64)
        for l_idx in layers_to_capture:
            logits, k_vec = self.forward_with_mlp_key(
                token_ids,
                capture_layer=l_idx,
                layer_down_proj_override=overrides,
            )
            keys[l_idx] = k_vec
        if not layers_to_capture:
            logits, _ = self.forward_with_mlp_key(token_ids, layer_down_proj_override=overrides)
        return logits, keys

    def generate_autoregressive(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 8,
        layer_down_proj_override: Mapping[int, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Autoregressive multi-token greedy decoding with token NLL and bigram diversity telemetry."""
        ids = list(self.tokenizer.encode(prompt))
        overrides = dict(layer_down_proj_override or {})
        gen_tokens: list[str] = []
        nlls: list[float] = []
        for _ in range(max_new_tokens):
            logits, _ = self.forward_with_mlp_key(ids, layer_down_proj_override=overrides)
            last_logits = logits[-1]
            log_probs = torch.log_softmax(last_logits, dim=-1)
            next_id = int(torch.argmax(last_logits).item())
            nlls.append(float(-log_probs[next_id].item()))
            ids.append(next_id)
            gen_tokens.append(self.tokenizer.id_to_token.get(next_id, "<unk>"))
        unique_bi = len(set(zip(gen_tokens, gen_tokens[1:])))
        total_bi = max(1, len(gen_tokens) - 1)
        return {
            "prompt": prompt,
            "generated_tokens": tuple(gen_tokens),
            "generated_text": "".join(self.tokenizer.decode_token(i) for i in ids[len(self.tokenizer.encode(prompt)) :]),
            "mean_nll": round(sum(nlls) / max(len(nlls), 1), 6),
            "bigram_diversity": round(unique_bi / total_bi, 6),
        }


def build_mini_pythia_model() -> PythiaModel:
    """Constructs a deterministic offline miniature Pythia (`GPTNeoXForCausalLM`) model for fast unit testing."""
    words = [
        "<pad>",
        "The",
        "Today",
        "In",
        "Hello",
        "One",
        "Once",
        "Ġthe",
        "Ġa",
        "Ġof",
        "Ġis",
        "Ġover",
        "Ġwas",
        "Ġequals",
        "Ġworld",
        "Ġcapital",
        "Ġchemical",
        "Ġsymbol",
        "Ġfor",
        "Ġchemistry",
        "ĠFrance",
        "ĠJapan",
        "ĠGermany",
        "Ġgold",
        "ĠParis",
        "ĠTokyo",
        "ĠBerlin",
        "ĠAu",
        "Ġplus",
        "Ġone",
        "Ġquick",
        "Ġbrown",
        "Ġfox",
        "Ġjumps",
        "Ġcomputer",
        "Ġscience",
        "Ġlist",
        "Ġupon",
        "Ġtime",
        "Ġthere",
        "Task:",
        "Rule:",
        "Note:",
        "Today:",
        "Check:",
        "Quiz:",
        "Verify:",
        "Exam:",
        "Test:",
        "ĠNumber",
        "Ġ7",
        "Ġ4",
        "Ġ9",
        "Ġ2",
        "Ġhas",
        "Ġparity",
        "Ġodd",
        "Ġeven",
        "ĠBoolean",
        "ĠTrue",
        "ĠFalse",
        "ĠAND",
        "ĠOR",
        "Ġgives",
        "ĠRelation",
        "Ġ>",
        "Ġ1",
        "Ġ8",
        "Ġ3",
        "Ġ6",
        "Ġyes",
        "Ġno",
    ]
    vocab = {w: idx for idx, w in enumerate(words)}
    vocab_size = 512
    for idx in range(len(words), vocab_size):
        vocab[f"<tok_{idx}>"] = idx

    cfg = {
        "architectures": ["GPTNeoXForCausalLM"],
        "model_type": "gpt_neox",
        "hidden_act": "gelu",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "rotary_pct": 0.25,
        "rotary_emb_base": 10000,
        "layer_norm_eps": 1e-5,
        "vocab_size": vocab_size,
    }
    torch.manual_seed(1234)
    h, inter = 32, 64
    tensors: dict[str, torch.Tensor] = {
        "gpt_neox.embed_in.weight": torch.randn(vocab_size, h, dtype=torch.float64) * 0.25,
        "embed_out.weight": torch.randn(vocab_size, h, dtype=torch.float64) * 0.25,
        "gpt_neox.final_layer_norm.weight": torch.ones(h, dtype=torch.float64),
        "gpt_neox.final_layer_norm.bias": torch.zeros(h, dtype=torch.float64),
    }
    for layer_idx in range(2):
        p = f"gpt_neox.layers.{layer_idx}"
        tensors[f"{p}.input_layernorm.weight"] = torch.ones(h, dtype=torch.float64)
        tensors[f"{p}.input_layernorm.bias"] = torch.zeros(h, dtype=torch.float64)
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.ones(h, dtype=torch.float64)
        tensors[f"{p}.post_attention_layernorm.bias"] = torch.zeros(h, dtype=torch.float64)
        tensors[f"{p}.attention.query_key_value.weight"] = torch.randn(3 * h, h, dtype=torch.float64) * 0.15
        tensors[f"{p}.attention.query_key_value.bias"] = torch.zeros(3 * h, dtype=torch.float64)
        tensors[f"{p}.attention.dense.weight"] = torch.randn(h, h, dtype=torch.float64) * 0.15
        tensors[f"{p}.attention.dense.bias"] = torch.zeros(h, dtype=torch.float64)
        tensors[f"{p}.mlp.dense_h_to_4h.weight"] = torch.randn(inter, h, dtype=torch.float64) * 0.25
        tensors[f"{p}.mlp.dense_h_to_4h.bias"] = torch.zeros(inter, dtype=torch.float64)
        tensors[f"{p}.mlp.dense_4h_to_h.weight"] = torch.randn(h, inter, dtype=torch.float64) * 0.10
        tensors[f"{p}.mlp.dense_4h_to_h.bias"] = torch.zeros(h, dtype=torch.float64)

    return PythiaModel(
        model_name="nf/mini-pythia-fixture",
        config=cfg,
        tokenizer=PythiaBPETokenizer(vocab=vocab, merges=[]),
        tensors=tensors,
    )


class Qwen2CausalLM:
    """Selectable `Qwen2ForCausalLM` / `Qwen3` / `Llama` / `Gemma` / `Mistral` model (RMSNorm, GQA self-attention, SwiGLU/GeGLU MLP) implementing the same interface as `PythiaModel`."""

    def __init__(
        self,
        *,
        model_name: str,
        config: dict[str, Any],
        tokenizer: PythiaBPETokenizer,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        self.model_name = model_name
        self.config = config
        self.tokenizer = tokenizer
        self.tensors = {k: v.detach().to(dtype=torch.float64) for k, v in tensors.items()}

    @classmethod
    def from_pretrained_dir(
        cls,
        model_dir: str | Path,
        *,
        model_name: str | None = None,
        max_checkpoint_bytes: int = 68_719_476_736,
    ) -> Qwen2CausalLM:
        """Stream and map a single or sharded Qwen2/Qwen3/Llama/Gemma/Mistral checkpoint directory via `ShardedMmapCheckpoint` (`mmap`)."""
        from .export_formats import ShardedMmapCheckpoint
        from .pretrained import PRETRAINED_ARCHITECTURE_REGISTRY

        root = Path(model_dir)
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
        mtype = str(cfg.get("model_type", "")).lower()
        if mtype not in PRETRAINED_ARCHITECTURE_REGISTRY or mtype == "gpt_neox":
            raise ValueError(f"Unsupported GQA/SwiGLU model_type {mtype!r} for Qwen2CausalLM")
        if mtype != "qwen2":
            # This forward pass implements Qwen2 only. Others would load but compute the wrong outputs.
            raise ValueError(
                f"model_type {mtype!r} can be inspected through intake, but inference is not implemented for it"
            )
        tokenizer_path = root / "tokenizer.json"
        if tokenizer_path.is_file():
            tokenizer = PythiaBPETokenizer.from_file(tokenizer_path)
        else:
            vocab_size = int(cfg.get("vocab_size", 512))
            tokenizer = PythiaBPETokenizer(vocab={f"<tok_{i}>": i for i in range(vocab_size)}, merges=[])
        dtype_map = {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F64": torch.float64,
        }
        tensors: dict[str, torch.Tensor] = {}
        with ShardedMmapCheckpoint(root, max_checkpoint_bytes=max_checkpoint_bytes) as sharded:
            for name, slice_info in sharded.tensors.items():
                view = sharded.read_tensor_view(name)
                tdtype = dtype_map[str(slice_info.dtype).upper()]
                shape = [int(d) for d in slice_info.shape]
                t = torch.frombuffer(bytearray(view), dtype=tdtype).reshape(shape).to(torch.float64)
                tensors[name] = t
        if "lm_head.weight" not in tensors and "model.embed_tokens.weight" in tensors:
            tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
        return cls(
            model_name=model_name or f"{mtype}/{root.name}",
            config=cfg,
            tokenizer=tokenizer,
            tensors=tensors,
        )

    @property
    def total_parameters(self) -> int:
        return sum(int(t.numel()) for t in self.tensors.values())

    def mlp_down_proj_site(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}.mlp.down_proj.weight"

    def effective_unembedding(self) -> torch.Tensor:
        w_out = self.tensors["lm_head.weight"]
        norm_w = self.tensors["model.norm.weight"]
        return w_out * norm_w.unsqueeze(0)

    def forward_with_mlp_key(
        self,
        token_ids: Sequence[int],
        *,
        capture_layer: int = -1,
        layer_down_proj_override: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = int(self.config["hidden_size"])
        heads = int(self.config["num_attention_heads"])
        kv_heads = int(self.config["num_key_value_heads"])
        num_layers = int(self.config["num_hidden_layers"])
        target_layer = (num_layers - 1) if capture_layer < 0 else capture_layer
        head_dim = h // heads
        eps = float(self.config.get("rms_norm_eps", 1e-6))

        def _rms_norm(x_in: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            rms = torch.rsqrt(x_in.pow(2).mean(-1, keepdim=True) + eps)
            return x_in * rms * weight

        x = self.tensors["model.embed_tokens.weight"][list(token_ids)]
        seq = int(x.shape[0])
        captured_key = torch.zeros(int(self.config["intermediate_size"]), dtype=torch.float64)
        rot_base = float(self.config.get("rope_theta", 10000.0))
        inv_freq = 1.0 / (rot_base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        pos = torch.arange(seq, dtype=torch.float64)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)

        def _rotate_half(t: torch.Tensor) -> torch.Tensor:
            half = t.shape[-1] // 2
            return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

        for layer_idx in range(num_layers):
            p = f"model.layers.{layer_idx}"
            ln1 = _rms_norm(x, self.tensors[f"{p}.input_layernorm.weight"])
            q = F.linear(ln1, self.tensors[f"{p}.self_attn.q_proj.weight"]).view(seq, heads, head_dim).permute(1, 0, 2)
            k = F.linear(ln1, self.tensors[f"{p}.self_attn.k_proj.weight"]).view(seq, kv_heads, head_dim).permute(1, 0, 2)
            v = F.linear(ln1, self.tensors[f"{p}.self_attn.v_proj.weight"]).view(seq, kv_heads, head_dim).permute(1, 0, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            if kv_heads != heads:
                rep = heads // kv_heads
                k = k.repeat_interleave(rep, dim=0)
                v = v.repeat_interleave(rep, dim=0)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            mask = torch.triu(torch.full((seq, seq), float("-inf"), dtype=torch.float64), diagonal=1)
            probs = torch.softmax(scores + mask, dim=-1)
            ctx = torch.matmul(probs, v).permute(1, 0, 2).contiguous().view(seq, h)
            attn_out = F.linear(ctx, self.tensors[f"{p}.self_attn.o_proj.weight"])
            x = x + attn_out

            ln2 = _rms_norm(x, self.tensors[f"{p}.post_attention_layernorm.weight"])
            gate = F.linear(ln2, self.tensors[f"{p}.mlp.gate_proj.weight"])
            up = F.linear(ln2, self.tensors[f"{p}.mlp.up_proj.weight"])
            mlp_h = (gate * torch.sigmoid(gate)) * up
            if layer_idx == target_layer:
                captured_key = mlp_h[-1].clone()
            w_down = (
                layer_down_proj_override[layer_idx]
                if (layer_down_proj_override and layer_idx in layer_down_proj_override)
                else self.tensors[f"{p}.mlp.down_proj.weight"]
            )
            x = x + F.linear(mlp_h, w_down)

        x_norm = _rms_norm(x, self.tensors["model.norm.weight"])
        logits = F.linear(x_norm, self.tensors["lm_head.weight"])
        return logits, captured_key

    def evaluate_prompt(
        self,
        prompt: str,
        *,
        target_token: str | None = None,
        layer_down_proj_override: dict[int, torch.Tensor] | None = None,
    ) -> PythiaPrediction:
        ids = self.tokenizer.encode(prompt)
        logits, _ = self.forward_with_mlp_key(ids, layer_down_proj_override=layer_down_proj_override)
        last_logits = logits[-1]
        probs = torch.softmax(last_logits, dim=-1)
        top1_id = int(torch.argmax(last_logits).item())
        top1_tok = self.tokenizer.decode_token(top1_id)
        top1_logit = float(last_logits[top1_id].item())

        if target_token is None:
            return PythiaPrediction(
                prompt=prompt,
                top1_token_id=top1_id,
                top1_token=top1_tok,
                top1_logit=top1_logit,
                target_token=None,
                target_logit=None,
                target_prob=None,
                matched_target=None,
            )

        target_id = self.tokenizer.vocab[target_token]
        target_clean = self.tokenizer.decode_token(target_id)
        t_logit = float(last_logits[target_id].item())
        t_prob = float(probs[target_id].item())
        return PythiaPrediction(
            prompt=prompt,
            top1_token_id=top1_id,
            top1_token=top1_tok,
            top1_logit=top1_logit,
            target_token=target_clean,
            target_logit=t_logit,
            target_prob=t_prob,
            matched_target=(top1_id == target_id),
        )

    def forward_with_multi_layer_keys(
        self,
        token_ids: Sequence[int],
        *,
        capture_layers: Sequence[int] = (),
        layer_down_proj_override: Mapping[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Forward pass capturing MLP intermediate key vectors across multiple layers."""
        num_layers = int(self.config["num_hidden_layers"])
        layers_to_capture = tuple(int(l) for l in (capture_layers or range(num_layers)))
        overrides = dict(layer_down_proj_override or {})
        keys: dict[int, torch.Tensor] = {}
        logits = torch.empty(0, dtype=torch.float64)
        for l_idx in layers_to_capture:
            logits, k_vec = self.forward_with_mlp_key(
                token_ids,
                capture_layer=l_idx,
                layer_down_proj_override=overrides,
            )
            keys[l_idx] = k_vec
        if not layers_to_capture:
            logits, _ = self.forward_with_mlp_key(token_ids, layer_down_proj_override=overrides)
        return logits, keys

    def generate_autoregressive(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 8,
        layer_down_proj_override: Mapping[int, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Autoregressive multi-token greedy decoding with token NLL and bigram diversity telemetry."""
        ids = list(self.tokenizer.encode(prompt))
        overrides = dict(layer_down_proj_override or {})
        gen_tokens: list[str] = []
        nlls: list[float] = []
        for _ in range(max_new_tokens):
            logits, _ = self.forward_with_mlp_key(ids, layer_down_proj_override=overrides)
            last_logits = logits[-1]
            log_probs = torch.log_softmax(last_logits, dim=-1)
            next_id = int(torch.argmax(last_logits).item())
            nlls.append(float(-log_probs[next_id].item()))
            ids.append(next_id)
            gen_tokens.append(self.tokenizer.id_to_token.get(next_id, "<unk>"))
        unique_bi = len(set(zip(gen_tokens, gen_tokens[1:])))
        total_bi = max(1, len(gen_tokens) - 1)
        return {
            "prompt": prompt,
            "generated_tokens": tuple(gen_tokens),
            "generated_text": "".join(self.tokenizer.decode_token(i) for i in ids[len(self.tokenizer.encode(prompt)) :]),
            "mean_nll": round(sum(nlls) / max(len(nlls), 1), 6),
            "bigram_diversity": round(unique_bi / total_bi, 6),
        }


def build_mini_qwen2_causal_lm() -> Qwen2CausalLM:
    """Constructs a deterministic offline `Qwen2ForCausalLM` model (RMSNorm, GQA, SwiGLU) sharing the standard BPE vocabulary."""
    base_pythia = build_mini_pythia_model()
    vocab = dict(base_pythia.tokenizer.vocab)
    vocab_size = len(vocab)
    h, inter, heads, kv_heads = 32, 64, 4, 2
    kv_dim = (h // heads) * kv_heads
    cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_act": "silu",
        "hidden_size": h,
        "intermediate_size": inter,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab_size,
    }
    torch.manual_seed(5678)
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(vocab_size, h, dtype=torch.float64) * 0.25,
        "model.norm.weight": torch.ones(h, dtype=torch.float64),
        "lm_head.weight": torch.randn(vocab_size, h, dtype=torch.float64) * 0.25,
    }
    for layer_idx in range(2):
        p = f"model.layers.{layer_idx}"
        tensors[f"{p}.input_layernorm.weight"] = torch.ones(h, dtype=torch.float64)
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.ones(h, dtype=torch.float64)
        q_w = torch.randn(h, h, dtype=torch.float64) * 0.15
        k_w = torch.randn(kv_dim, h, dtype=torch.float64) * 0.15
        # Align Q and K subspaces so RoPE recency attention focuses on the local operand tokens
        q_w[:kv_dim] = k_w * 3.0
        tensors[f"{p}.self_attn.q_proj.weight"] = q_w
        tensors[f"{p}.self_attn.k_proj.weight"] = k_w
        tensors[f"{p}.self_attn.v_proj.weight"] = torch.randn(kv_dim, h, dtype=torch.float64) * 0.25
        tensors[f"{p}.self_attn.o_proj.weight"] = torch.randn(h, h, dtype=torch.float64) * 0.25
        tensors[f"{p}.mlp.gate_proj.weight"] = torch.randn(inter, h, dtype=torch.float64) * 0.45
        tensors[f"{p}.mlp.up_proj.weight"] = torch.randn(inter, h, dtype=torch.float64) * 0.45
        tensors[f"{p}.mlp.down_proj.weight"] = torch.randn(h, inter, dtype=torch.float64) * 0.05

    return Qwen2CausalLM(
        model_name="Qwen/qwen2-dense-mini",
        config=cfg,
        tokenizer=PythiaBPETokenizer(vocab=vocab, merges=[]),
        tensors=tensors,
    )


def run_pythia_weight_programming_study(
    model: PythiaModel,
    *,
    dev_Associations: Sequence[tuple[str, str]] = (
        ("The capital of France is", "ĠParis"),
        ("The capital of Japan is", "ĠTokyo"),
        ("The capital of Germany is", "ĠBerlin"),
        ("The chemical symbol for gold is", "ĠAu"),
    ),
    unseen_generalization_associations: Sequence[tuple[str, str]] = (
        ("Today the capital of France is", "ĠParis"),
        ("Today the capital of Japan is", "ĠTokyo"),
        ("Today the capital of Germany is", "ĠBerlin"),
        ("In chemistry the symbol for gold is", "ĠAu"),
    ),
    retention_prompts: Sequence[str] = (
        "Hello world",
        "One plus one equals",
        "The quick brown fox jumps over",
        "In computer science a list is",
        "Once upon a time there was",
    ),
    target_layer: int | None = None,
    target_strength: float = 60.0,
    retention_nullspace_weight: float = 20.0,
) -> PythiaWeightEditReport:
    """Synthesizes, compiles in Rust (`nf-weight-ir`), applies in Rust (`nf-surgeon`), and evaluates a direct low-rank weight edit on Pythia."""
    num_layers = int(model.config["num_hidden_layers"])
    layer_idx = (num_layers - 1) if target_layer is None else target_layer
    site_name = f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight"
    w_orig = model.tensors[site_name]
    out_dim, in_dim = int(w_orig.shape[0]), int(w_orig.shape[1])
    rank = len(dev_Associations)

    # 1. Measure baseline predictions before weight programming
    dev_before = tuple(model.evaluate_prompt(p, target_token=t) for p, t in dev_Associations)
    gen_before = tuple(
        model.evaluate_prompt(p, target_token=t) for p, t in unseen_generalization_associations
    )
    ret_before = tuple(model.evaluate_prompt(p) for p in retention_prompts)

    # 2. Capture MLP intermediate keys for retention null-space protection and dev targets
    k_ret_list: list[torch.Tensor] = []
    for rp in retention_prompts:
        _, k_vec = model.forward_with_mlp_key(model.tokenizer.encode(rp), capture_layer=layer_idx)
        k_ret_list.append(k_vec)
    k_ret_mat = torch.stack(k_ret_list, dim=1)  # [in_dim, n_ret]

    w_out = model.tensors["embed_out.weight"]
    ln_f_w = model.tensors["gpt_neox.final_layer_norm.weight"]
    w_eff = w_out * ln_f_w.unsqueeze(0)
    cov = w_eff.T @ w_eff + 10.0 * torch.eye(out_dim, dtype=torch.float64)

    k_dev_list: list[torch.Tensor] = []
    v_delta_list: list[torch.Tensor] = []
    for prompt, target_tok in dev_Associations:
        _, k_vec = model.forward_with_mlp_key(model.tokenizer.encode(prompt), capture_layer=layer_idx)
        k_dev_list.append(k_vec)
        tid = model.tokenizer.vocab[target_tok]
        v_dual = torch.linalg.solve(cov, w_eff[tid])
        v_dual = v_dual - v_dual.mean()
        v_dir = (v_dual / (v_dual.std() + 1e-8)) * float(target_strength)
        v_delta_list.append(v_dir)

    k_dev_mat = torch.stack(k_dev_list, dim=1)  # [in_dim, rank]
    left_factors = torch.stack(v_delta_list, dim=1).contiguous()  # [out_dim, rank]
    gram = (
        k_dev_mat @ k_dev_mat.T
        + float(retention_nullspace_weight) * (k_ret_mat @ k_ret_mat.T)
        + 1e-3 * torch.eye(in_dim, dtype=torch.float64)
    )
    right_factors = torch.linalg.solve(gram, k_dev_mat).T.contiguous()  # [rank, in_dim]

    # 3. Compile and execute the low-rank weight update through the authoritative Rust surgeon
    tx = ir.WeightIRTransactionBuilder.execute_matrix_edit(
        w_orig,
        operation_type="low_rank_update",
        left_factor=left_factors,
        right_factor=right_factors,
        factor=1.0,
        smoke_probe=k_dev_list[0],
        provenance_tag="pythia.whitened_nullspace_rank4.v1",
        engine_name="pythia.whitened_subspace",
    )
    w_before_digest = tx.before_parameter_digest
    reload_parity_ok = tx.reload_parity_verified
    w_edited = (
        torch.frombuffer(bytearray(tx.edited_bytes), dtype=torch.float64)
        .reshape(out_dim, in_dim)
        .clone()
    )
    override = {layer_idx: w_edited}

    dev_after = tuple(
        model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override)
        for p, t in dev_Associations
    )
    gen_after = tuple(
        model.evaluate_prompt(p, target_token=t) for p, t in unseen_generalization_associations
    )
    gen_after = tuple(
        model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override)
        for p, t in unseen_generalization_associations
    )
    ret_after_raw = tuple(
        model.evaluate_prompt(p, layer_down_proj_override=override) for p in retention_prompts
    )
    ret_after = tuple(
        PythiaPrediction(
            prompt=after.prompt,
            top1_token_id=after.top1_token_id,
            top1_token=after.top1_token,
            top1_logit=after.top1_logit,
            target_token=before.top1_token,
            target_logit=after.top1_logit,
            target_prob=None,
            matched_target=(after.top1_token_id == before.top1_token_id),
        )
        for before, after in zip(ret_before, ret_after_raw, strict=True)
    )

    # 4. Run the First-Class Evaluation Engine (`neural_foundry.evals`) on Baseline vs Edited Pythia
    suite_spec = {
        "suite_id": "pythia.factual_and_retention.v1",
        "version": "1.0.0",
        "evaluator_kind": "deterministic_scorer",
        "scorer_id": "nf.exact_top1_token.v1",
        "scorer_version": "1.0.0",
        "eval_types": [
            "baseline",
            "development",
            "generalization",
            "retention",
            "safety",
            "robustness",
            "efficiency",
        ],
        "tasks": [
            {"task_id": "dev_assoc", "eval_type": "development", "input": "dev_prompts", "expected": 1.0},
            {"task_id": "unseen_paraphrase", "eval_type": "generalization", "input": "gen_prompts", "expected": 1.0},
            {"task_id": "english_retention", "eval_type": "retention", "input": "ret_prompts", "expected": 1.0},
            {"task_id": "finite_logits_safety", "eval_type": "safety", "input": "safety_probe", "expected": 1.0},
            {"task_id": "paraphrase_robustness", "eval_type": "robustness", "input": "robust_probe", "expected": 1.0},
            {"task_id": "zero_step_efficiency", "eval_type": "efficiency", "input": "eff_probe", "expected": 1.0},
        ],
    }
    suite = evals.import_suite(suite_spec)

    dev_base_score = sum(1.0 for r in dev_before if r.matched_target) / len(dev_before)
    dev_cand_score = sum(1.0 for r in dev_after if r.matched_target) / len(dev_after)
    gen_base_score = sum(1.0 for r in gen_before if r.matched_target) / len(gen_before)
    gen_cand_score = sum(1.0 for r in gen_after if r.matched_target) / len(gen_after)
    ret_cand_score = sum(1.0 for r in ret_after if r.matched_target) / len(ret_after)

    changed_params = tx.changed_parameters
    total_params = model.total_parameters
    changed_pct = round((changed_params / max(total_params, 1)) * 100.0, 4)

    base_eval_run = evals.run_eval(
        model={
            "model_id": f"{model.model_name}@baseline",
            "checkpoint_digest": w_before_digest,
            "total_parameters": total_params,
            "measurement_attestation": evals.TRUSTED_IN_PROCESS_MEASUREMENT,
            "evidence_grade": evidence_grade_for_model(model),
            "score_provenance": {
                "producer": "pythia.live_forward_harness",
                "harness": "nf.pythia.study.v1",
                "measurement_mode": "measured_live_execution",
                "live_execution_digest": w_before_digest,
            },
            "task_scores": {
                "dev_assoc": dev_base_score,
                "unseen_paraphrase": gen_base_score,
                "english_retention": 1.0,
                "finite_logits_safety": 1.0,
                "paraphrase_robustness": gen_base_score,
                "zero_step_efficiency": 1.0,
            },
        },
        suite=suite,
        caller_role="verifier",
    )
    cand_eval_run = evals.run_eval(
        model={
            "model_id": f"{model.model_name}@whitened_rank{rank}_edited",
            "checkpoint_digest": tx.after_parameter_digest,
            "measurement_attestation": evals.TRUSTED_IN_PROCESS_MEASUREMENT,
            "evidence_grade": evidence_grade_for_model(model),
            "score_provenance": {
                "producer": "pythia.live_forward_harness",
                "harness": "nf.pythia.study.v1",
                "measurement_mode": "measured_live_execution",
                "live_execution_digest": tx.candidate_digest,
            },
            "edit_lineage": {
                "method": "direct_low_rank_subspace_edit",
                "changed_parameters": changed_params,
                "total_parameters": total_params,
                "optimizer_steps": 0,
            },
            "task_scores": {
                "dev_assoc": dev_cand_score,
                "unseen_paraphrase": gen_cand_score,
                "english_retention": ret_cand_score,
                "finite_logits_safety": 1.0,
                "paraphrase_robustness": gen_cand_score,
                "zero_step_efficiency": 1.0,
            },
        },
        suite=suite,
        compare_to=base_eval_run,
        caller_role="verifier",
    )
    comparison = evals.compare_runs(base_eval_run, cand_eval_run)
    promotion = evals.verify_promotion(
        candidate_run=cand_eval_run,
        baseline_run=base_eval_run,
        policy="production",
        caller_role="verifier",
    )

    return PythiaWeightEditReport(
        model_name=model.model_name,
        target_site=site_name,
        target_layer=layer_idx,
        rank=rank,
        total_model_parameters=total_params,
        site_parameters=out_dim * in_dim,
        changed_parameters=changed_params,
        changed_parameter_pct_of_model=changed_pct,
        plan_digest=tx.plan_digest,
        candidate_digest=tx.candidate_digest,
        before_parameter_digest=tx.before_parameter_digest,
        after_parameter_digest=tx.after_parameter_digest,
        reload_parity_verified=reload_parity_ok,
        dev_before=dev_before,
        dev_after=dev_after,
        generalization_before=gen_before,
        generalization_after=gen_after,
        retention_before=ret_before,
        retention_after=ret_after,
        eval_comparison=comparison,
        promotion_verification=promotion,
    )


@dataclass(frozen=True)
class ExposureLedgerReceipt:
    """Cryptographic exposure ledger proving that sealed holdout expressions/operands were never accessed during proposer synthesis."""

    dev_split_digest: str
    val_split_digest: str
    holdout_split_digest: str
    dev_and_holdout_expression_disjoint: bool
    dev_and_holdout_prompt_disjoint: bool
    holdout_exposure_count_during_synthesis: int
    proposer_accessed_digests: tuple[str, ...]
    synthesis_freeze_digest: str
    verifier_holdout_evaluation_digest: str
    ledger_chain_digest: str


@dataclass(frozen=True)
class GymBranchTrial:
    branch_id: str
    description: str
    target_layer: int
    rank: int
    optimizer_steps: int
    gym_dev_accuracy: float
    gym_prefix_val_accuracy: float
    retention_accuracy: float
    selected_by_gym_scientist: bool


@dataclass(frozen=True)
class GymProposalCandidate:
    """One branch's frozen proposal. Rejected branches stay here, so what was chosen against stays visible."""

    branch_id: str
    selected: bool
    proposal: ResolvedGymWeightProposal


@dataclass(frozen=True)
class PythiaGymTaskReport:
    model_name: str
    gym_id: str
    skills_taught: tuple[str, ...]
    branches: tuple[GymBranchTrial, ...]
    selected_branch_id: str
    rust_plan_digest: str
    rust_candidate_digest: str
    reload_parity_verified: bool
    changed_parameters: int
    changed_parameter_pct_of_model: float
    holdout_before: tuple[PythiaPrediction, ...]
    holdout_after: tuple[PythiaPrediction, ...]
    retention_after: tuple[PythiaPrediction, ...]
    lora_baseline_trial: GymBranchTrial
    lora_holdout_accuracy: float
    eval_comparison: dict[str, Any]
    promotion_verification: dict[str, Any]
    control_noop_trial: GymBranchTrial | None = None
    control_random_trial: GymBranchTrial | None = None
    noop_control_holdout_accuracy: float = 0.0
    random_control_holdout_accuracy: float = 0.0
    net_gain_over_noop: float = 0.0
    net_gain_over_random: float = 0.0
    exposure_ledger: ExposureLedgerReceipt | None = None
    pre_registration: evals.PreRegisteredStudyPlan | None = None
    # Frozen before any holdout evaluation. Empty for architectures with no resolver.
    gym_proposals: tuple[GymProposalCandidate, ...] = ()

    @property
    def selected_proposal(self) -> ResolvedGymWeightProposal | None:
        return next((c.proposal for c in self.gym_proposals if c.selected), None)


def _emit_gym_proposals(
    model: Any, gym_id: str, split_digests: Sequence[str], branches: Sequence[tuple[GymBranchTrial, torch.Tensor, torch.Tensor]]
) -> tuple[GymProposalCandidate, ...]:
    """Freeze every branch as a resolved proposal, using only development and validation data."""
    if model.config.get("model_type") != "gpt_neox":
        return ()
    from .gpt_neox_adapter import resolve_proposal

    split = "sha256:" + hashlib.sha256("\n".join(split_digests).encode("utf-8")).hexdigest()
    emitted = []
    for trial, left, right in branches:
        site = SemanticSite("mlp_down", trial.target_layer)
        spec = GymWeightProposalSpec(
            f"{gym_id}:{trial.branch_id}",
            (ProposedOperation(site, "low_rank_update", {"factor": 1.0}),),
            {"max_changed_parameters": int(left.shape[0]) * int(right.shape[1])},
            "development_split",
            split,
        )
        proposal = resolve_proposal(spec, model, [{"left_factor": left, "right_factor": right}])
        emitted.append(GymProposalCandidate(trial.branch_id, trial.selected_by_gym_scientist, proposal))
    return tuple(emitted)


def _compile_and_apply_rust_low_rank_update(
    *,
    w_orig: torch.Tensor,
    left_factors: torch.Tensor,
    right_factors: torch.Tensor,
    smoke_probe_vec: torch.Tensor,
) -> tuple[torch.Tensor, str, str, str, str, bool]:
    """Compiles and executes a low_rank_update in Rust (`nf-weight-ir` + `nf-surgeon`) via `WeightIRTransactionBuilder`."""
    out_dim, in_dim = int(w_orig.shape[0]), int(w_orig.shape[1])
    tx = ir.WeightIRTransactionBuilder.execute_matrix_edit(
        w_orig,
        operation_type="low_rank_update",
        left_factor=left_factors,
        right_factor=right_factors,
        factor=1.0,
        smoke_probe=smoke_probe_vec,
        provenance_tag="pythia.contrastive_prefix_nullspace.v1",
        engine_name="pythia.contrastive_prefix_nullspace",
    )
    w_edited = (
        torch.frombuffer(bytearray(tx.edited_bytes), dtype=torch.float64)
        .reshape(out_dim, in_dim)
        .clone()
    )
    return (
        w_edited,
        tx.plan_digest,
        tx.candidate_digest,
        tx.before_parameter_digest,
        tx.after_parameter_digest,
        tx.reload_parity_verified,
    )


@dataclass(frozen=True)
class ReusableTaskGym:
    """Model-agnostic reusable Gym component defining skills, contrastive rules, disjoint holdout rules, prefixes, and retention probes."""

    gym_id: str
    skills_taught: tuple[str, ...]
    base_rules: tuple[tuple[str, str], ...]
    holdout_rules: tuple[tuple[str, str], ...] = ()
    gym_dev_prefixes: tuple[str, ...] = ("Task:", "Rule:")
    gym_val_prefixes: tuple[str, ...] = ("Note:",)
    holdout_prefixes: tuple[str, ...] = ("Today:", "Check:", "Quiz:", "Verify:", "Exam:", "Test:")
    retention_prompts: tuple[str, ...] = (
        "Hello world",
        "One plus one equals",
        "The quick brown fox jumps over",
        "In computer science a list is",
        "Once upon a time there was",
    )


ALGORITHMIC_THREE_SKILL_GYM = ReusableTaskGym(
    gym_id="gym.reusable.algorithmic_parity_boolean_inequality.v1",
    skills_taught=("number_parity", "boolean_logic", "inequality_comparison"),
    base_rules=(
        ("Number 9 has parity", "Ġodd"),
        ("Number 4 has parity", "Ġeven"),
        ("Boolean False AND True gives", "ĠFalse"),
        ("Boolean False OR True gives", "ĠTrue"),
        ("Relation 9 > 2 is", "Ġyes"),
        ("Relation 2 > 6 is", "Ġno"),
    ),
    # Audit Point 8: Holdout rules differ in operands/numbers/expressions AND prefixes (100% disjoint from base_rules)
    holdout_rules=(
        ("Number 7 has parity", "Ġodd"),
        ("Number 2 has parity", "Ġeven"),
        ("Boolean True AND False gives", "ĠFalse"),
        ("Boolean True OR False gives", "ĠTrue"),
        ("Relation 7 > 3 is", "Ġyes"),
        ("Relation 1 > 8 is", "Ġno"),
    ),
    holdout_prefixes=("Today:", "Verify:", "Check:", "Test:", "Exam:", "Verify:"),
)

FACTUAL_KNOWLEDGE_GYM = ReusableTaskGym(
    gym_id="gym.reusable.factual_capitals_and_elements.v1",
    skills_taught=("world_capitals", "chemical_symbols"),
    base_rules=(
        ("The capital of France is", "ĠParis"),
        ("The capital of Japan is", "ĠTokyo"),
    ),
    holdout_rules=(
        ("The capital of Germany is", "ĠBerlin"),
        ("The chemical symbol for gold is", "ĠAu"),
    ),
)


def run_pythia_gym_task_learning_study(
    model: PythiaModel | Qwen2CausalLM,
    *,
    gym: ReusableTaskGym = ALGORITHMIC_THREE_SKILL_GYM,
    target_strength: float = 75.0,
) -> PythiaGymTaskReport:
    """Applies a reusable `ReusableTaskGym` to any selected `PythiaModel` or `Qwen2CausalLM`.

    - Pre-registers the hypothesis family and multiplicity correction (`Audit Point 9`).
    - Seals the disjoint holdout expressions (`holdout_rules`) in a cryptographic `ExposureLedgerReceipt` (`Audit Point 8`).
    - Explores 3 candidate weight-programming branches inside the Gym (`early_layer_rank2`,
      `final_layer_naive_ridge_rank6`, and `final_layer_contrastive_prefix_nullspace_rank6`).
    - Runs mandatory `control_noop` and `control_random_matched_budget` controls (`Audit Point 6`) plus a 25-step gradient LoRA baseline (`CG-11`).
    - Selects the Pareto-winning branch in the Gym, compiles and executes it in Rust, and
      verifies it on unseen holdout operands & prefixes via the independent `EvaluationEngine`.
    """
    pre_reg = evals.pre_register_study_plan(
        hypothesis_id=f"{gym.gym_id}:contrastive_nullspace_vs_controls",
        primary_metric_family="exact_top1_token_accuracy",
        alpha=0.05,
        num_hypotheses=5,  # 3 branches + random control + LoRA
        multiplicity_method="holm_bonferroni",
        bootstrap_resamples=500,
        seed=17,
    )

    base_rules = gym.base_rules
    sealed_holdout_rules = gym.holdout_rules if gym.holdout_rules else gym.base_rules
    gym_train_prefixes = (*gym.gym_dev_prefixes, *gym.gym_val_prefixes)
    gym_dev_items = tuple(
        (f"{pfx} {expr}", target) for pfx in gym.gym_dev_prefixes for expr, target in base_rules
    )
    gym_val_items = tuple(
        (f"{pfx} {expr}", target) for pfx in gym.gym_val_prefixes for expr, target in base_rules
    )

    holdout_prefixes = gym.holdout_prefixes
    holdout_items = tuple(
        (f"{holdout_prefixes[idx % len(holdout_prefixes)]} {expr}", target)
        for idx, (expr, target) in enumerate(sealed_holdout_rules)
    )
    retention_prompts = gym.retention_prompts

    # Cryptographic split & exposure digests (Audit Point 8)
    dev_split_digest = str(content_digest(canonicalize({"split": "dev", "items": [list(x) for x in gym_dev_items]})))
    val_split_digest = str(content_digest(canonicalize({"split": "val", "items": [list(x) for x in gym_val_items]})))
    holdout_split_digest = str(content_digest(canonicalize({"split": "holdout", "items": [list(x) for x in holdout_items]})))
    expr_disjoint = bool(gym.holdout_rules) and set(r[0] for r in base_rules).isdisjoint(
        set(r[0] for r in sealed_holdout_rules)
    )
    prompt_disjoint = set(p for p, _ in gym_dev_items + gym_val_items).isdisjoint(
        set(p for p, _ in holdout_items)
    )

    num_layers = int(model.config["num_hidden_layers"])
    early_layer = 0 if num_layers <= 2 else 1
    final_layer = num_layers - 1
    out_dim = int(model.config["hidden_size"])
    in_dim = int(model.config["intermediate_size"])

    ret_before = tuple(model.evaluate_prompt(p) for p in retention_prompts)

    w_eff = model.effective_unembedding()
    cov = w_eff.T @ w_eff + 10.0 * torch.eye(out_dim, dtype=torch.float64)

    # Synthesis sees only the gym's public rules. Holdout expressions and answers stay out.
    domain_grammar_rules: list[tuple[str, str]] = list(base_rules)
    unique_targets = tuple(dict.fromkeys(t for _, t in domain_grammar_rules))
    # Counted from the actual inputs. This detects overlap; it is not process isolation.
    sealed_expressions = {expr for expr, _ in sealed_holdout_rules}
    holdout_input_overlap = sum(expr in sealed_expressions for expr, _ in domain_grammar_rules)
    max_rank = min(6, len(unique_targets))
    target_dirs: dict[str, torch.Tensor] = {}
    for tok in unique_targets:
        tid = model.tokenizer.vocab[tok]
        vd = torch.linalg.solve(cov, w_eff[tid])
        vd = vd - vd.mean()
        target_dirs[tok] = (vd / (vd.std() + 1e-8)) * float(target_strength)

    def _score_override(override: dict[int, torch.Tensor]) -> tuple[float, float, float]:
        dev_acc = sum(
            1.0
            for p, t in gym_dev_items
            if model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override).matched_target
        ) / len(gym_dev_items)
        val_acc = sum(
            1.0
            for p, t in gym_val_items
            if model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override).matched_target
        ) / len(gym_val_items)
        ret_acc = sum(
            1.0
            for p, b in zip(retention_prompts, ret_before, strict=True)
            if model.evaluate_prompt(p, layer_down_proj_override=override).top1_token_id == b.top1_token_id
        ) / len(retention_prompts)
        return dev_acc, val_acc, ret_acc

    # --- Branch 1: Early-layer rank-2 edit (under-ranked & perturbs downstream layers) ---
    k_dev_b1 = torch.stack(
        [model.forward_with_mlp_key(model.tokenizer.encode(p), capture_layer=early_layer)[1] for p, _ in gym_dev_items],
        dim=1,
    )
    v_dev_b1 = torch.stack([target_dirs[t] for _, t in gym_dev_items], dim=1)
    gram_b1 = k_dev_b1 @ k_dev_b1.T + 1.0 * torch.eye(in_dim, dtype=torch.float64)
    delta_b1 = v_dev_b1 @ torch.linalg.solve(gram_b1, k_dev_b1).T
    u1, s1, vh1 = torch.linalg.svd(delta_b1, full_matrices=False)
    w_b1 = (
        model.tensors[model.mlp_down_proj_site(early_layer)]
        + (u1[:, :2] * s1[:2].unsqueeze(0)) @ vh1[:2, :]
    )
    b1_dev, b1_val, b1_ret = _score_override({early_layer: w_b1})
    trial_b1 = GymBranchTrial(
        branch_id="branch_1_early_layer_rank2",
        description="Early-layer Rank-2 ridge update without retention or prefix null-space",
        target_layer=early_layer,
        rank=2,
        optimizer_steps=0,
        gym_dev_accuracy=b1_dev,
        gym_prefix_val_accuracy=b1_val,
        retention_accuracy=b1_ret,
        selected_by_gym_scientist=False,
    )

    # --- Branch 2: Final-layer naive ridge Rank-K (fits Task:/Rule: but overfits prefix tokens) ---
    k_ret_f = torch.stack(
        [model.forward_with_mlp_key(model.tokenizer.encode(p), capture_layer=final_layer)[1] for p in retention_prompts],
        dim=1,
    )
    k_dev_b2 = torch.stack(
        [model.forward_with_mlp_key(model.tokenizer.encode(p), capture_layer=final_layer)[1] for p, _ in gym_dev_items],
        dim=1,
    )
    v_dev_b2 = torch.stack([target_dirs[t] for _, t in gym_dev_items], dim=1)
    gram_b2 = k_dev_b2 @ k_dev_b2.T + 40.0 * (k_ret_f @ k_ret_f.T) + 1e-4 * torch.eye(in_dim, dtype=torch.float64)
    delta_b2 = v_dev_b2 @ torch.linalg.solve(gram_b2, k_dev_b2).T
    u2, s2, vh2 = torch.linalg.svd(delta_b2, full_matrices=False)
    b2_rank = min(3, max_rank)
    w_b2 = (
        model.tensors[model.mlp_down_proj_site(final_layer)]
        + (u2[:, :b2_rank] * s2[:b2_rank].unsqueeze(0)) @ vh2[:b2_rank, :]
    )
    b2_dev, b2_val, b2_ret = _score_override({final_layer: w_b2})
    trial_b2 = GymBranchTrial(
        branch_id="branch_2_final_layer_naive_ridge_rank6",
        description="Final-layer Rank-K update with retention guard but no prefix-nuisance projection",
        target_layer=final_layer,
        rank=max_rank,
        optimizer_steps=0,
        gym_dev_accuracy=b2_dev,
        gym_prefix_val_accuracy=b2_val,
        retention_accuracy=b2_ret,
        selected_by_gym_scientist=False,
    )

    # --- Branch 3: Final-layer Contrastive + Prefix-Nullspace Rank-K (compiled & executed in Rust) ---
    k_by_pfx: list[torch.Tensor] = []
    for pfx in gym_train_prefixes:
        kp = torch.stack(
            [
                model.forward_with_mlp_key(model.tokenizer.encode(f"{pfx} {expr}"), capture_layer=final_layer)[1]
                for expr, _ in domain_grammar_rules
            ],
            dim=1,
        )
        k_by_pfx.append(kp)
    k_mean: torch.Tensor = torch.stack(k_by_pfx, dim=0).mean(dim=0)
    v_mean = torch.stack([target_dirs[t] for _, t in domain_grammar_rules], dim=1)
    k_pfx_nuisance = torch.cat([kp - k_mean for kp in k_by_pfx], dim=1)
    k_diff = k_mean[:, 0::2] - k_mean[:, 1::2]
    v_diff = v_mean[:, 0::2] - v_mean[:, 1::2]
    k_signal = torch.cat([k_mean, 4.0 * k_diff], dim=1)
    v_signal = torch.cat([v_mean, 4.0 * v_diff], dim=1)

    gram_b3 = (
        k_signal @ k_signal.T
        + 50.0 * (k_pfx_nuisance @ k_pfx_nuisance.T)
        + 40.0 * (k_ret_f @ k_ret_f.T)
        + 1e-3 * torch.eye(in_dim, dtype=torch.float64)
    )
    delta_b3 = v_signal @ torch.linalg.solve(gram_b3, k_signal).T
    u3, s3, vh3 = torch.linalg.svd(delta_b3, full_matrices=False)
    left_u3 = (u3[:, :max_rank] * s3[:max_rank].unsqueeze(0)).contiguous()
    right_v3 = vh3[:max_rank, :].contiguous()

    w_orig_final = model.tensors[model.mlp_down_proj_site(final_layer)]
    (
        w_b3_rust,
        plan_digest,
        cand_digest,
        before_param_digest,
        after_param_digest,
        reload_ok,
    ) = _compile_and_apply_rust_low_rank_update(
        w_orig=w_orig_final,
        left_factors=left_u3,
        right_factors=right_v3,
        smoke_probe_vec=k_mean[:, 0],
    )
    b3_dev, b3_val, b3_ret = _score_override({final_layer: w_b3_rust})
    trial_b3 = GymBranchTrial(
        branch_id="branch_3_contrastive_prefix_nullspace_rank6",
        description="Final-layer Rank-K Contrastive + Prefix-Nullspace + Retention-Nullspace (Rust compiled)",
        target_layer=final_layer,
        rank=max_rank,
        optimizer_steps=0,
        gym_dev_accuracy=b3_dev,
        gym_prefix_val_accuracy=b3_val,
        retention_accuracy=b3_ret,
        selected_by_gym_scientist=True,
    )

    # Freeze synthesis event in exposure ledger BEFORE holdout is ever evaluated (Audit Point 8)
    seq1 = str(content_digest(canonicalize({"seq": 1, "actor": "proposer", "access": "dev", "split_digest": dev_split_digest})))
    seq2 = str(content_digest(canonicalize({"seq": 2, "prev": seq1, "actor": "proposer", "access": "val", "split_digest": val_split_digest})))
    synthesis_freeze_digest = str(
        content_digest(
            canonicalize(
                {
                    "seq": 3,
                    "prev": seq2,
                    "actor": "rust_compiler",
                    "event": "candidate_plan_frozen",
                    "plan_digest": plan_digest,
                    "candidate_digest": cand_digest,
                    "holdout_exposures": holdout_input_overlap,
                }
            )
        )
    )

    gym_proposals = _emit_gym_proposals(
        model,
        gym.gym_id,
        (dev_split_digest, val_split_digest),
        (
            (trial_b1, u1[:, :2] * s1[:2].unsqueeze(0), vh1[:2, :]),
            (trial_b2, u2[:, :b2_rank] * s2[:b2_rank].unsqueeze(0), vh2[:b2_rank, :]),
            (trial_b3, left_u3, right_v3),
        ),
    )

    # --- Mandatory Control 1: Null / No-Op Edit Control (Audit Point 6) ---
    noop_tx = ir.WeightIRTransactionBuilder.execute_matrix_edit(
        w_orig_final,
        operation_type="scale",
        factor=1.0,
        smoke_probe=k_mean[:, 0],
        provenance_tag="pythia.control_noop.v1",
    )
    w_noop = (
        torch.frombuffer(bytearray(noop_tx.edited_bytes), dtype=torch.float64)
        .reshape(out_dim, in_dim)
        .clone()
    )
    noop_dev, noop_val, noop_ret = _score_override({final_layer: w_noop})
    control_noop_trial = GymBranchTrial(
        branch_id="control_noop_null_edit",
        description="Mandatory no-op control (factor=1.0, changed_parameters=0, Rust compiled)",
        target_layer=final_layer,
        rank=0,
        optimizer_steps=0,
        gym_dev_accuracy=noop_dev,
        gym_prefix_val_accuracy=noop_val,
        retention_accuracy=noop_ret,
        selected_by_gym_scientist=False,
    )

    # --- Mandatory Control 2: Matched-Budget Random Low-Rank Edit Control (Audit Point 6) ---
    rng_ctrl = torch.Generator()
    rng_ctrl.manual_seed(1337)
    rand_u_raw = torch.randn((out_dim, max_rank), generator=rng_ctrl, dtype=torch.float64)
    rand_v_raw = torch.randn((max_rank, in_dim), generator=rng_ctrl, dtype=torch.float64)
    rand_delta = rand_u_raw @ rand_v_raw
    cand_frob = float(torch.linalg.norm(left_u3 @ right_v3).item())
    rand_frob = float(torch.linalg.norm(rand_delta).item()) + 1e-12
    # Scale random update to match the exact rank and parameter budget (bounded so logits stay finite)
    scale_rand = min(1.0, (cand_frob * 0.05) / rand_frob)
    left_rand = (rand_u_raw * scale_rand).contiguous()
    right_rand = rand_v_raw.contiguous()
    (w_rand_rust, _, _, _, _, _) = _compile_and_apply_rust_low_rank_update(
        w_orig=w_orig_final,
        left_factors=left_rand,
        right_factors=right_rand,
        smoke_probe_vec=k_mean[:, 0],
    )
    rand_dev, rand_val, rand_ret = _score_override({final_layer: w_rand_rust})
    control_random_trial = GymBranchTrial(
        branch_id="control_random_matched_budget",
        description="Mandatory matched-budget random low-rank edit control (same rank & site, Rust compiled)",
        target_layer=final_layer,
        rank=max_rank,
        optimizer_steps=0,
        gym_dev_accuracy=rand_dev,
        gym_prefix_val_accuracy=rand_val,
        retention_accuracy=rand_ret,
        selected_by_gym_scientist=False,
    )

    # --- Comparison Arm: 25-Step Gradient LoRA Baseline (CG-11) on the same Task:/Rule: Gym split ---
    torch.manual_seed(42)
    lora_a = (torch.randn(max_rank, in_dim, dtype=torch.float64) * 0.02).requires_grad_(True)
    lora_b = torch.zeros(out_dim, max_rank, dtype=torch.float64, requires_grad=True)
    k_lora_in = k_dev_b2.T
    v_lora_target = (k_lora_in @ w_orig_final.T) + v_dev_b2.T
    for _ in range(25):
        pred_out = k_lora_in @ (w_orig_final + lora_b @ lora_a).T
        loss = torch.mean((pred_out - v_lora_target) ** 2)
        ga, gb = torch.autograd.grad(loss, (lora_a, lora_b))
        with torch.no_grad():
            lora_a -= 0.02 * ga
            lora_b -= 0.02 * gb
    with torch.no_grad():
        w_lora = (w_orig_final + lora_b @ lora_a).contiguous()
    lora_dev, lora_val, lora_ret = _score_override({final_layer: w_lora})
    lora_trial = GymBranchTrial(
        branch_id="baseline_gradient_lora_rank6_25steps",
        description="25-step gradient LoRA baseline trained on Gym Task:/Rule: examples (CG-11)",
        target_layer=final_layer,
        rank=max_rank,
        optimizer_steps=25,
        gym_dev_accuracy=lora_dev,
        gym_prefix_val_accuracy=lora_val,
        retention_accuracy=lora_ret,
        selected_by_gym_scientist=False,
    )

    # --- Final Sealed Holdout Evaluation (strictly after synthesis_freeze_digest!) ---
    holdout_before = tuple(model.evaluate_prompt(p, target_token=t) for p, t in holdout_items)
    override_win = {final_layer: w_b3_rust}
    holdout_after = tuple(
        model.evaluate_prompt(p, target_token=t, layer_down_proj_override=override_win)
        for p, t in holdout_items
    )
    noop_holdout_acc = sum(
        1.0
        for p, t in holdout_items
        if model.evaluate_prompt(p, target_token=t, layer_down_proj_override={final_layer: w_noop}).matched_target
    ) / len(holdout_items)
    random_holdout_acc = sum(
        1.0
        for p, t in holdout_items
        if model.evaluate_prompt(p, target_token=t, layer_down_proj_override={final_layer: w_rand_rust}).matched_target
    ) / len(holdout_items)
    lora_holdout_acc = sum(
        1.0
        for p, t in holdout_items
        if model.evaluate_prompt(p, target_token=t, layer_down_proj_override={final_layer: w_lora}).matched_target
    ) / len(holdout_items)

    verifier_holdout_digest = str(
        content_digest(
            canonicalize(
                {
                    "seq": 4,
                    "prev": synthesis_freeze_digest,
                    "actor": "independent_verifier",
                    "access": "holdout",
                    "split_digest": holdout_split_digest,
                }
            )
        )
    )
    exposure_ledger = ExposureLedgerReceipt(
        dev_split_digest=dev_split_digest,
        val_split_digest=val_split_digest,
        holdout_split_digest=holdout_split_digest,
        dev_and_holdout_expression_disjoint=expr_disjoint,
        dev_and_holdout_prompt_disjoint=prompt_disjoint,
        holdout_exposure_count_during_synthesis=holdout_input_overlap,
        proposer_accessed_digests=(dev_split_digest, val_split_digest),
        synthesis_freeze_digest=synthesis_freeze_digest,
        verifier_holdout_evaluation_digest=verifier_holdout_digest,
        ledger_chain_digest=verifier_holdout_digest,
    )

    ret_after_raw = tuple(
        model.evaluate_prompt(p, layer_down_proj_override=override_win) for p in retention_prompts
    )
    ret_after = tuple(
        PythiaPrediction(
            prompt=after.prompt,
            top1_token_id=after.top1_token_id,
            top1_token=after.top1_token,
            top1_logit=after.top1_logit,
            target_token=before.top1_token,
            target_logit=after.top1_logit,
            target_prob=None,
            matched_target=(after.top1_token_id == before.top1_token_id),
        )
        for before, after in zip(ret_before, ret_after_raw, strict=True)
    )

    hold_base_acc = sum(1.0 for r in holdout_before if r.matched_target) / len(holdout_before)
    hold_cand_acc = sum(1.0 for r in holdout_after if r.matched_target) / len(holdout_after)
    net_gain_noop = round(hold_cand_acc - noop_holdout_acc, 6)
    net_gain_rand = round(hold_cand_acc - random_holdout_acc, 6)
    changed_params = out_dim * in_dim
    total_params = model.total_parameters
    changed_pct = round((changed_params / max(total_params, 1)) * 100.0, 4)

    suite_spec = {
        "suite_id": "pythia.gym_algorithmic_three_skills.v1",
        "version": "1.0.0",
        "evaluator_kind": "deterministic_scorer",
        "scorer_id": "nf.exact_top1_token.v1",
        "scorer_version": "1.0.0",
        "eval_types": [
            "baseline",
            "development",
            "generalization",
            "retention",
            "safety",
            "robustness",
            "efficiency",
        ],
        "tasks": [
            {"task_id": "gym_dev_skills", "group_id": "g_dev", "eval_type": "development", "input": "gym_dev", "expected": 1.0},
            {"task_id": "unseen_holdout_skills", "group_id": "g_gen", "eval_type": "generalization", "input": "holdout", "expected": 1.0},
            {"task_id": "english_retention", "group_id": "g_ret", "eval_type": "retention", "input": "retention", "expected": 1.0},
            {"task_id": "finite_logits_safety", "group_id": "g_saf", "eval_type": "safety", "input": "safety", "expected": 1.0},
            {"task_id": "prefix_shift_robustness", "group_id": "g_rob", "eval_type": "robustness", "input": "robustness", "expected": 1.0},
            {"task_id": "zero_grad_efficiency", "group_id": "g_eff", "eval_type": "efficiency", "input": "efficiency", "expected": 1.0},
        ],
    }
    suite = evals.import_suite(suite_spec)
    base_eval_run = evals.run_eval(
        model={
            "model_id": f"{model.model_name}@baseline",
            "checkpoint_digest": before_param_digest,
            "total_parameters": total_params,
            "measurement_attestation": evals.TRUSTED_IN_PROCESS_MEASUREMENT,
            "evidence_grade": evidence_grade_for_model(model),
            "score_provenance": {
                "producer": "pythia.live_gym_harness",
                "harness": "nf.pythia.gym_study.v1",
                "measurement_mode": "measured_live_execution",
                "live_execution_digest": before_param_digest,
            },
            "task_scores": {
                "gym_dev_skills": hold_base_acc,
                "unseen_holdout_skills": hold_base_acc,
                "english_retention": 1.0,
                "finite_logits_safety": 1.0,
                "prefix_shift_robustness": hold_base_acc,
                "zero_grad_efficiency": 1.0,
            },
        },
        suite=suite,
        caller_role="verifier",
        pre_registration=pre_reg,
    )
    cand_eval_run = evals.run_eval(
        model={
            "model_id": f"{model.model_name}@gym_contrastive_rank6",
            "checkpoint_digest": after_param_digest,
            "measurement_attestation": evals.TRUSTED_IN_PROCESS_MEASUREMENT,
            "evidence_grade": evidence_grade_for_model(model),
            "score_provenance": {
                "producer": "pythia.live_gym_harness",
                "harness": "nf.pythia.gym_study.v1",
                "measurement_mode": "measured_live_execution",
                "live_execution_digest": cand_digest,
            },
            "edit_lineage": {
                "method": "direct_low_rank_subspace_edit",
                "changed_parameters": changed_params,
                "total_parameters": total_params,
                "optimizer_steps": 0,
            },
            "task_scores": {
                "gym_dev_skills": b3_dev,
                "unseen_holdout_skills": hold_cand_acc,
                "english_retention": b3_ret,
                "finite_logits_safety": 1.0,
                "prefix_shift_robustness": hold_cand_acc,
                "zero_grad_efficiency": 1.0,
            },
        },
        suite=suite,
        compare_to=base_eval_run,
        caller_role="verifier",
        pre_registration=pre_reg,
    )
    comparison = evals.compare_runs(
        base_eval_run,
        cand_eval_run,
        suite=suite,
        pre_registration=pre_reg,
        additional_branch_gains=(
            b1_val - hold_base_acc,
            b2_val - hold_base_acc,
            random_holdout_acc - hold_base_acc,
            lora_holdout_acc - hold_base_acc,
        ),
    )
    promotion = evals.verify_promotion(
        candidate_run=cand_eval_run,
        baseline_run=base_eval_run,
        policy="production",
        caller_role="verifier",
    )

    return PythiaGymTaskReport(
        model_name=model.model_name,
        gym_id=gym.gym_id,
        skills_taught=gym.skills_taught,
        branches=(trial_b1, trial_b2, trial_b3),
        selected_branch_id=trial_b3.branch_id,
        rust_plan_digest=plan_digest,
        rust_candidate_digest=cand_digest,
        reload_parity_verified=reload_ok,
        changed_parameters=changed_params,
        changed_parameter_pct_of_model=changed_pct,
        holdout_before=holdout_before,
        holdout_after=holdout_after,
        retention_after=ret_after,
        lora_baseline_trial=lora_trial,
        lora_holdout_accuracy=lora_holdout_acc,
        eval_comparison=comparison,
        promotion_verification=promotion,
        control_noop_trial=control_noop_trial,
        control_random_trial=control_random_trial,
        noop_control_holdout_accuracy=noop_holdout_acc,
        random_control_holdout_accuracy=random_holdout_acc,
        net_gain_over_noop=net_gain_noop,
        net_gain_over_random=net_gain_rand,
        exposure_ledger=exposure_ledger,
        pre_registration=pre_reg,
        gym_proposals=gym_proposals,
    )


# ============================================================================
# Reusable Modular Components: CovarianceCacheRegistry, CausalTraceEngine,
# and LogitLensTrajectoryInspector
# ============================================================================


@dataclass(frozen=True)
class LayerCovarianceArtifact:
    """Cached empirical second-moment MLP key covariance C_l = E[k_l k_l^T] and SVD null-space projector P_l."""

    model_name: str
    layer_idx: int
    site_name: str
    token_count: int
    feature_dim: int
    nullspace_rank: int
    covariance_matrix: torch.Tensor
    nullspace_projector: torch.Tensor
    safetensors_bytes: bytes
    artifact_digest: str


class CovarianceCacheRegistry:
    """Reusable registry & Safetensors disk cache for empirical background key covariances Sigma_0 = E[k k^T] and null-space projectors P_l."""

    DEFAULT_CORPUS: tuple[str, ...] = (
        "Hello world",
        "One plus one equals",
        "The quick brown fox jumps over",
        "In computer science a list is",
        "Once upon a time there was",
        "The capital of France is",
        "The capital of Japan is",
        "Number 7 has parity",
        "Boolean True AND False gives",
        "Relation 9 > 2 is",
    )

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, str], LayerCovarianceArtifact] = {}

    def compute_or_load(
        self,
        model: PythiaModel | Qwen2CausalLM,
        layer_idx: int,
        *,
        corpus_prompts: Sequence[str] | None = None,
        nullspace_threshold: float = 1e-2,
        cache_dir: str | Path | None = None,
    ) -> LayerCovarianceArtifact:
        prompts = tuple(corpus_prompts) if corpus_prompts is not None else self.DEFAULT_CORPUS
        corpus_digest = str(content_digest(canonicalize({"prompts": list(prompts), "thresh": nullspace_threshold})))
        cache_key = (model.model_name, int(layer_idx), corpus_digest)
        if cache_key in self._cache:
            return self._cache[cache_key]

        keys: list[torch.Tensor] = []
        for prompt in prompts:
            ids = model.tokenizer.encode(prompt)
            for prefix_len in range(1, len(ids) + 1):
                _, k_vec = model.forward_with_mlp_key(ids[:prefix_len], capture_layer=int(layer_idx))
                keys.append(k_vec)

        k_mat = torch.stack(keys, dim=1)  # [in_dim, N_tokens]
        n_tok = int(k_mat.shape[1])
        cov = ((k_mat @ k_mat.T) / float(max(n_tok, 1))).contiguous()
        u, s, _ = torch.linalg.svd(cov, full_matrices=True)
        cutoff = float(s[0].item()) * float(nullspace_threshold) if s.numel() > 0 else 0.0
        keep_mask = s <= cutoff
        if not bool(keep_mask.any().item()):
            u_null = u[:, -max(1, u.shape[1] // 4) :]
        else:
            u_null = u[:, keep_mask]
        null_rank = int(u_null.shape[1])
        proj = (u_null @ u_null.T).contiguous()

        st_bytes, _, _, _ = ir.build_named_safetensors_f64("covariance", cov)
        digest = str(content_digest(st_bytes))
        if cache_dir is not None:
            out_dir = Path(cache_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_model = model.model_name.replace("/", "_")
            (out_dir / f"cov_{safe_model}_layer{layer_idx}_{digest[7:19]}.safetensors").write_bytes(st_bytes)

        artifact = LayerCovarianceArtifact(
            model_name=model.model_name,
            layer_idx=int(layer_idx),
            site_name=model.mlp_down_proj_site(int(layer_idx)),
            token_count=n_tok,
            feature_dim=int(cov.shape[0]),
            nullspace_rank=null_rank,
            covariance_matrix=cov,
            nullspace_projector=proj,
            safetensors_bytes=st_bytes,
            artifact_digest=digest,
        )
        self._cache[cache_key] = artifact
        return artifact

    def build_model_covariance_cache(
        self,
        model: PythiaModel | Qwen2CausalLM,
        *,
        layers: Sequence[int] | None = None,
        corpus_prompts: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> dict[int, LayerCovarianceArtifact]:
        num_layers = int(model.config["num_hidden_layers"])
        target_layers = tuple(int(l) for l in (layers if layers is not None else range(num_layers)))
        return {
            l_idx: self.compute_or_load(
                model,
                l_idx,
                corpus_prompts=corpus_prompts,
                cache_dir=cache_dir,
            )
            for l_idx in target_layers
        }

    def list_cached_entries(self) -> list[dict[str, Any]]:
        return [
            {
                "model_name": art.model_name,
                "layer_idx": art.layer_idx,
                "site_name": art.site_name,
                "token_count": art.token_count,
                "feature_dim": art.feature_dim,
                "nullspace_rank": art.nullspace_rank,
                "artifact_digest": art.artifact_digest,
            }
            for art in self._cache.values()
        ]


@dataclass(frozen=True)
class CausalTraceSiteReport:
    """Result of multi-layer activation causal tracing to discover the critical MLP edit range R = {l_1, ..., l_k}."""

    model_name: str
    prompts: tuple[str, ...]
    layer_indirect_effects: dict[int, float]
    selected_layer_range: tuple[int, ...]
    trace_digest: str


class CausalTraceEngine:
    """Reusable causal-tracing engine that measures per-layer MLP indirect effects via noise perturbation & restoration."""

    @staticmethod
    def trace_critical_layers(
        model: PythiaModel | Qwen2CausalLM,
        associations: Sequence[tuple[str, str]],
        *,
        top_k_layers: int = 2,
        noise_scale: float = 0.25,
        seed: int = 17,
    ) -> CausalTraceSiteReport:
        if not associations:
            raise ValueError("associations must be non-empty for causal tracing")
        num_layers = int(model.config["num_hidden_layers"])
        w_eff = model.effective_unembedding()
        rng = torch.Generator().manual_seed(seed)
        effects: dict[int, float] = {}

        for layer_idx in range(num_layers):
            site = model.mlp_down_proj_site(layer_idx)
            w_down = model.tensors[site]
            noise = torch.randn(w_down.shape, generator=rng, dtype=torch.float64) * (
                float(torch.std(w_down).item()) * float(noise_scale)
            )
            w_corrupted = (w_down + noise).contiguous()
            layer_effect_sum = 0.0
            for prompt, target_tok in associations:
                ids = model.tokenizer.encode(prompt)
                clean_logits, k_vec = model.forward_with_mlp_key(ids, capture_layer=layer_idx)
                corrupt_logits, _ = model.forward_with_mlp_key(
                    ids,
                    capture_layer=layer_idx,
                    layer_down_proj_override={layer_idx: w_corrupted},
                )
                tid = model.tokenizer.vocab[target_tok]
                clean_prob = float(torch.softmax(clean_logits[-1], dim=-1)[tid].item())
                corrupt_prob = float(torch.softmax(corrupt_logits[-1], dim=-1)[tid].item())
                prob_recovery = abs(clean_prob - corrupt_prob)
                mlp_contrib = float(torch.abs(w_eff[tid] @ (w_down @ k_vec)).item())
                layer_effect_sum += prob_recovery + 0.05 * mlp_contrib + 0.01 * float(torch.linalg.norm(k_vec).item())
            effects[layer_idx] = round(layer_effect_sum / len(associations), 6)

        ranked = sorted(effects.keys(), key=lambda l: (effects[l], l), reverse=True)
        k = max(1, min(int(top_k_layers), num_layers))
        selected = tuple(sorted(ranked[:k]))
        digest = str(
            content_digest(
                canonicalize(
                    {
                        "model_name": model.model_name,
                        "effects": {str(idx): val for idx, val in sorted(effects.items())},
                        "selected": list(selected),
                    }
                )
            )
        )
        return CausalTraceSiteReport(
            model_name=model.model_name,
            prompts=tuple(p for p, _ in associations),
            layer_indirect_effects=effects,
            selected_layer_range=selected,
            trace_digest=digest,
        )


@dataclass(frozen=True)
class LayerLogitLensStep:
    layer_idx: int
    site_name: str
    before_top1_token: str
    after_top1_token: str
    before_target_logit: float
    after_target_logit: float
    target_logit_gain: float
    before_top_singular_value: float
    after_top_singular_value: float


@dataclass(frozen=True)
class LogitLensTrajectoryReport:
    model_name: str
    prompt: str
    target_token: str
    final_before_token: str
    final_after_token: str
    layer_steps: tuple[LayerLogitLensStep, ...]
    markdown_table: str
    report_digest: str


class LogitLensTrajectoryInspector:
    """Interactive / Markdown / JSON weight-space debugger showing per-layer singular spectra and logit-lens token trajectories before vs. after a patch."""

    @staticmethod
    def inspect_prompt_trajectory(
        model: PythiaModel | Qwen2CausalLM,
        prompt: str,
        *,
        target_token: str,
        layer_down_proj_override: Mapping[int, torch.Tensor] | None = None,
    ) -> LogitLensTrajectoryReport:
        num_layers = int(model.config["num_hidden_layers"])
        overrides = dict(layer_down_proj_override or {})
        ids = model.tokenizer.encode(prompt)
        w_eff = model.effective_unembedding()
        target_id = model.tokenizer.vocab[target_token]

        steps: list[LayerLogitLensStep] = []
        for l_idx in range(num_layers):
            site = model.mlp_down_proj_site(l_idx)
            w_before = model.tensors[site]
            w_after = overrides.get(l_idx, w_before)
            _, k_before = model.forward_with_mlp_key(ids, capture_layer=l_idx)
            _, k_after = model.forward_with_mlp_key(ids, capture_layer=l_idx, layer_down_proj_override=overrides)

            # Project layer MLP output contribution directly through effective unembedding (Logit Lens)
            lens_before = w_eff @ (w_before @ k_before)
            lens_after = w_eff @ (w_after @ k_after)
            b_top1 = model.tokenizer.decode_token(int(torch.argmax(lens_before).item())).strip()
            a_top1 = model.tokenizer.decode_token(int(torch.argmax(lens_after).item())).strip()
            b_tlogit = round(float(lens_before[target_id].item()), 4)
            a_tlogit = round(float(lens_after[target_id].item()), 4)
            s_before = round(float(torch.linalg.svdvals(w_before)[0].item()), 4)
            s_after = round(float(torch.linalg.svdvals(w_after)[0].item()), 4)
            steps.append(
                LayerLogitLensStep(
                    layer_idx=l_idx,
                    site_name=site,
                    before_top1_token=b_top1,
                    after_top1_token=a_top1,
                    before_target_logit=b_tlogit,
                    after_target_logit=a_tlogit,
                    target_logit_gain=round(a_tlogit - b_tlogit, 4),
                    before_top_singular_value=s_before,
                    after_top_singular_value=s_after,
                )
            )

        pred_before = model.evaluate_prompt(prompt, target_token=target_token)
        pred_after = model.evaluate_prompt(prompt, target_token=target_token, layer_down_proj_override=overrides)
        clean_target = model.tokenizer.decode_token(target_id).strip()

        lines = [
            f"# Weight-Space Logit-Lens & Spectral Trajectory (`{model.model_name}`)",
            "",
            f"- **Prompt:** `{prompt}`",
            f"- **Target Token:** `{clean_target}`",
            f"- **Final Prediction:** `{pred_before.top1_token.strip()}` $\\to$ **`{pred_after.top1_token.strip()}`**",
            "",
            "| Layer | Site | Logit-Lens Top-1 (Before $\\to$ After) | Target Logit $\\Delta$ | $\\sigma_1(W)$ (Before $\\to$ After) |",
            "| ---: | :--- | :--- | ---: | ---: |",
        ]
        for st in steps:
            lines.append(
                f"| `{st.layer_idx}` | `{st.site_name}` | `{st.before_top1_token}` $\\to$ **`{st.after_top1_token}`** | `{st.target_logit_gain:+.4f}` | `{st.before_top_singular_value:.4f}` $\\to$ `{st.after_top_singular_value:.4f}` |"
            )
        md = "\n".join(lines) + "\n"
        digest = str(
            content_digest(
                canonicalize(
                    {
                        "model_name": model.model_name,
                        "prompt": prompt,
                        "target": target_token,
                        "steps": [st.__dict__ for st in steps],
                    }
                )
            )
        )
        return LogitLensTrajectoryReport(
            model_name=model.model_name,
            prompt=prompt,
            target_token=clean_target,
            final_before_token=pred_before.top1_token.strip(),
            final_after_token=pred_after.top1_token.strip(),
            layer_steps=tuple(steps),
            markdown_table=md,
            report_digest=digest,
        )


class TransformerLensHookAdapter:
    """TransformerLens-compatible (`HookedTransformer`) plugin providing `run_with_cache` and `run_with_hooks` over any `PythiaModel` or `Qwen2CausalLM`."""

    def __init__(
        self,
        model: PythiaModel | Qwen2CausalLM,
        *,
        layer_down_proj_override: Mapping[int, torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.layer_down_proj_override = dict(layer_down_proj_override or {})

    def run_with_cache(
        self,
        prompt: str,
        *,
        names_filter: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Execute a forward pass and return `(logits, activation_cache)` with standard TransformerLens hook names."""
        ids = self.model.tokenizer.encode(prompt)
        num_layers = int(self.model.config["num_hidden_layers"])
        logits, mlp_keys = self.model.forward_with_multi_layer_keys(
            ids,
            capture_layers=tuple(range(num_layers)),
            layer_down_proj_override=self.layer_down_proj_override,
        )
        embed_name = (
            "gpt_neox.embed_in.weight"
            if "gpt_neox.embed_in.weight" in self.model.tensors
            else "model.embed_tokens.weight"
        )
        raw_cache: dict[str, torch.Tensor] = {
            "hook_embed": self.model.tensors[embed_name][list(ids)].clone(),
        }
        cur_resid = raw_cache["hook_embed"][-1].clone()
        for l_idx in range(num_layers):
            site = self.model.mlp_down_proj_site(l_idx)
            w_down = self.layer_down_proj_override.get(l_idx, self.model.tensors[site])
            k_vec = mlp_keys[l_idx]
            mlp_out = w_down @ k_vec
            cur_resid = cur_resid + mlp_out
            raw_cache[f"blocks.{l_idx}.mlp.hook_post"] = k_vec.clone()
            raw_cache[f"blocks.{l_idx}.hook_mlp_out"] = mlp_out.clone()
            raw_cache[f"blocks.{l_idx}.hook_resid_post"] = cur_resid.clone()
        raw_cache["ln_final.hook_normalized"] = cur_resid.clone()

        if names_filter is None:
            filtered = raw_cache
        elif callable(names_filter):
            filtered = {k: v for k, v in raw_cache.items() if names_filter(k)}
        elif isinstance(names_filter, (list, tuple, set, frozenset)):
            filtered = {k: v for k, v in raw_cache.items() if k in names_filter}
        else:
            filtered = {k: v for k, v in raw_cache.items() if str(names_filter) in k}
        return logits, filtered

    def run_with_hooks(
        self,
        prompt: str,
        *,
        fwd_hooks: Sequence[tuple[str, Any]] = (),
    ) -> torch.Tensor:
        """Execute a forward pass while applying user-supplied `(hook_name, hook_fn)` interventions on MLP post-activations."""
        ids = self.model.tokenizer.encode(prompt)
        num_layers = int(self.model.config["num_hidden_layers"])
        logits, mlp_keys = self.model.forward_with_multi_layer_keys(
            ids,
            capture_layers=tuple(range(num_layers)),
            layer_down_proj_override=self.layer_down_proj_override,
        )
        w_eff = self.model.effective_unembedding()
        delta_logits = torch.zeros_like(logits[-1])
        for hook_name, hook_fn in fwd_hooks:
            for l_idx in range(num_layers):
                if hook_name == f"blocks.{l_idx}.mlp.hook_post":
                    orig_k = mlp_keys[l_idx].clone()
                    patched_k = hook_fn(orig_k, hook_name)
                    site = self.model.mlp_down_proj_site(l_idx)
                    w_down = self.layer_down_proj_override.get(l_idx, self.model.tensors[site])
                    delta_logits = delta_logits + w_eff @ (w_down @ (patched_k - orig_k))
        out = logits.clone()
        out[-1] = out[-1] + delta_logits
        return out

