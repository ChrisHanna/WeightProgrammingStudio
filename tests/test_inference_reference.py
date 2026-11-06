"""Compare our Pythia forward pass and tokenizer with Hugging Face `transformers`.

Four levels, on every prompt in `PROMPTS`: tokenization, logits at all positions, hidden
activations at every layer, and base-versus-edit logit deltas for edits at several layers.

This is optional. It skips unless `transformers` and a real checkpoint are available, and it
never downloads anything. To run it:

    pip install transformers                 # or put it on PYTHONPATH
    NF_PYTHIA_14M_DIR=/path/to/pythia-14m \\
    NF_PYTHIA_31M_DIR=/path/to/pythia-31m \\
    python -m pytest tests/test_inference_reference.py -v -s

The directories default to /tmp/pythia-14m and /tmp/pythia-31m. Each needs config.json,
model.safetensors and tokenizer.json. `-s` prints the measured differences;
NF_INFERENCE_REPORT=path.json also writes them to a file.

Why there are two sets of tolerances. In a float64 model, `transformers` (checked on 5.22.1)
still does two steps in float32, both in `modeling_gpt_neox.py`:

  * `eager_attention_forward`: `softmax(attn_weights, dim=-1, dtype=torch.float32)`
  * `GPTNeoXRotaryEmbedding`: `inv_freq`, `freqs`, `cos` and `sin` are computed in float32

So stock output cannot agree with a true float64 pass beyond float32 noise. The `corrected`
reference changes only those two steps to float64, leaving every other line of the reference
running. Agreement with it is held to `CORRECTED_TOLERANCE`, near machine precision, which is
the real correctness check. `STOCK_TOLERANCE` is the looser bound for the unmodified library.
Both were set from the measurements recorded beside them.
"""

from __future__ import annotations

import json
import os
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip(
    "transformers",
    reason="optional reference comparison: `transformers` is not installed (see module docstring)",
)
modeling = pytest.importorskip("transformers.models.gpt_neox.modeling_gpt_neox")

from neural_foundry import pythia as pythia_module  # noqa: E402
from neural_foundry.pythia import PythiaModel  # noqa: E402

CHECKPOINTS = {
    "pythia-14m": ("NF_PYTHIA_14M_DIR", "/tmp/pythia-14m", 14_067_712),
    "pythia-31m": ("NF_PYTHIA_31M_DIR", "/tmp/pythia-31m", 30_494_720),
}

LEVELS = ("logits", "residual", "mlp", "edit_delta")

# Limits on the largest absolute difference, per level. Measured on 5.22.1, worst of both models:
#   corrected: logits 2.5e-11, residual 1.9e-12, mlp 1.3e-12, edit delta 2.9e-11
#   stock:     logits 3.4e-3,  residual 4.0e-4,  mlp 1.9e-4,  edit delta 3.8e-3
# The corrected limit rests on a noise floor: a random 1-ulp change to our own weights alone moved
# logits by 1.8e-11 to 3.2e-11. Injected bugs, a 0.01% change to layer_norm_eps or to the RoPE base,
# moved logits by 9e-5 and 1e-4, so 1e-10 separates rounding from error by about six orders.
CORRECTED_TOLERANCE = dict.fromkeys(LEVELS, 1e-10)
STOCK_TOLERANCE = {"logits": 1e-2, "residual": 1e-3, "mlp": 1e-3, "edit_delta": 1e-2}

# An edit must move logits by at least this much, or the delta comparison proves nothing.
MIN_EDIT_EFFECT = 1e-3
# The corrected tables must stay this close to stock ones: float32 rounding, not a formula change.
MAX_ROPE_TABLE_CHANGE = 1e-5
MAX_SOFTMAX_CHANGE = 1e-5

PROMPTS = (
    # Plain English
    "The capital of France is",
    "The chemical symbol for gold is",
    "Hello world",
    "One plus one equals",
    "The quick brown fox jumps over",
    "In computer science a list is",
    "Once upon a time there was",
    "She walked to the store and bought",
    "Water boils at a temperature of",
    "the",
    "A",
    # Punctuation and contractions
    "Hello, world!",
    "don't stop; it's 3.14 today.",
    "I'm sure we'll see they've gone, and you'd agree.",
    "\"quoted\" (parenthesised) [bracketed] {braced}",
    "Wait... what?! No -- really.",
    "rock'n'roll isn't dead'",
    "'s 't 're 've 'm 'll 'd",
    "user@example.com visited https://example.com/a?b=c#d",
    # Digits and code
    "abc123def 456ghi",
    "x = 1 + 2 * (3 - 4) / 5",
    "1,000,000 is 10^6 and 3.5e-7 is small",
    "def add(a, b):\n    return a + b\n",
    "for (int i = 0; i < n; ++i) { sum += a[i]; }",
    "<div class=\"box\">text</div>",
    "SELECT name FROM users WHERE id = 7;",
    "// comment\n/* block */\n# hash",
    # Whitespace
    "line one\nline two",
    "paragraph one\n\nparagraph two\n\n\n",
    "two  spaces",
    "   leading spaces",
    "trailing space ",
    "trailing spaces   ",
    "a\tb\tc",
    "mixed \t \n whitespace",
    "\n",
    " ",
    "wide" + " " * 30 + "gap",
    "windows\r\nline\r\nends",
    "non breaking spaces",
    "separator \x1c'll and \x85 next",  # \x1c is `str.isspace` but not whitespace to the reference
    # Non-ASCII
    "café naïve résumé",
    "é combining mark becomes one character",
    "日本語のテキスト",
    "Русский текст и 42 числа",
    "النص العربي",
    "한국어 문장입니다",
    "emoji 🙂 here 👩‍💻 too",
    "Ελληνικά, ٣٤ and ½ and Ⅷ",
    # Special tokens
    "<|endoftext|>Next document",
    "before<|endoftext|>after<|padding|>",
    "<|endoftext|",
    # Long enough for attention and RoPE precision to show
    "The quick brown fox jumps over the lazy dog and then keeps on running through the long "
    "grass of the meadow until evening falls over the quiet hills and everyone goes home.",
    "Numbers: " + ", ".join(str(i) for i in range(60)),
    "Repeat repeat repeat " * 25,
)


def _float64_rotary_forward(self, x: torch.Tensor, position_ids: torch.Tensor):
    """`GPTNeoXRotaryEmbedding.forward` with only its float32 casts changed to float64."""
    config = self.config
    head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
    base = config.rope_parameters["rope_theta"]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    freqs = position_ids[..., None].to(torch.float64) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling


def _float64_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """`eager_attention_forward` with only its softmax dtype changed to float64."""
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float64)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    return attn_output, attn_weights


class _corrected_reference:
    """Temporarily replace the reference's two float32 steps."""

    def __enter__(self) -> None:
        self._rotary = modeling.GPTNeoXRotaryEmbedding.forward
        self._eager = modeling.eager_attention_forward
        modeling.GPTNeoXRotaryEmbedding.forward = _float64_rotary_forward
        modeling.eager_attention_forward = _float64_eager_attention

    def __exit__(self, *_: object) -> None:
        modeling.GPTNeoXRotaryEmbedding.forward = self._rotary
        modeling.eager_attention_forward = self._eager


def _reference_trace(hf, ids: list[int]) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    """Logits, each layer's input plus the final-norm input, and each layer's GELU output."""
    residual: list[torch.Tensor] = []
    mlp: list[torch.Tensor] = []
    handles = []
    for layer in hf.gpt_neox.layers:
        handles.append(
            layer.register_forward_pre_hook(
                lambda _m, a, k: residual.append((a[0] if a else k["hidden_states"])[0].clone()),
                with_kwargs=True,
            )
        )
        handles.append(layer.mlp.act.register_forward_hook(lambda _m, _a, out: mlp.append(out[0].clone())))
    handles.append(hf.gpt_neox.final_layer_norm.register_forward_pre_hook(lambda _m, a: residual.append(a[0][0].clone())))
    try:
        with torch.no_grad():
            logits = hf(torch.tensor([ids]), use_cache=False).logits[0]
    finally:
        for handle in handles:
            handle.remove()
    return {"logits": logits, "residual": residual, "mlp": mlp}


def _our_trace(model: PythiaModel, ids: list[int], override=None) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    """Same trace from our pass. `pythia.F` is shimmed so production code needs no capture hooks."""
    residual: list[torch.Tensor] = []
    mlp: list[torch.Tensor] = []
    real = torch.nn.functional

    def layer_norm(x, *args, **kwargs):
        if not residual or residual[-1] is not x:
            residual.append(x)
        return real.layer_norm(x, *args, **kwargs)

    def gelu(x):
        out = real.gelu(x)
        mlp.append(out)
        return out

    shim = types.SimpleNamespace(layer_norm=layer_norm, gelu=gelu, linear=real.linear)
    saved = pythia_module.F
    pythia_module.F = shim
    try:
        with torch.no_grad():
            logits, _ = model.forward_with_mlp_key(ids, layer_down_proj_override=override)
    finally:
        pythia_module.F = saved
    return {"logits": logits, "residual": residual, "mlp": mlp}


def _edits(model: PythiaModel) -> dict[str, dict[int, torch.Tensor]]:
    """Deterministic rank-1 edits to dense_4h_to_h at the first, middle and last layer, and all three."""
    layers = int(model.config["num_hidden_layers"])
    chosen = (0, layers // 2, layers - 1)
    generator = torch.Generator().manual_seed(20261106)
    single: dict[int, torch.Tensor] = {}
    for layer in chosen:
        weight = model.tensors[model.mlp_down_proj_site(layer)]
        u = torch.randn(weight.shape[0], generator=generator, dtype=torch.float64)
        v = torch.randn(weight.shape[1], generator=generator, dtype=torch.float64)
        single[layer] = weight + 0.02 * torch.outer(u, v)
    edits = {f"layer{layer}": {layer: single[layer]} for layer in chosen}
    edits["combined"] = single
    return edits


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _measure(model: PythiaModel, hf, tokenized: list[list[int]]) -> dict[str, float]:
    """Largest difference per level over every prompt, against `hf` in its current state."""
    worst = dict.fromkeys(LEVELS, 0.0)
    worst["min_edit_effect"] = float("inf")
    edits = _edits(model)
    for ids in tokenized:
        ours = _our_trace(model, ids)
        ref = _reference_trace(hf, ids)
        assert len(ours["residual"]) == len(ref["residual"]) == len(hf.gpt_neox.layers) + 1
        assert len(ours["mlp"]) == len(ref["mlp"]) == len(hf.gpt_neox.layers)
        worst["logits"] = max(worst["logits"], _max_abs(ours["logits"], ref["logits"]))
        for name in ("residual", "mlp"):
            for a, b in zip(ours[name], ref[name], strict=True):
                worst[name] = max(worst[name], _max_abs(a, b))
        for edit in edits.values():
            edited = _our_trace(model, ids, override=edit)["logits"]
            originals = {layer: hf.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight.data.clone() for layer in edit}
            try:
                for layer, weight in edit.items():
                    hf.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight.data.copy_(weight)
                ref_edited = _reference_trace(hf, ids)["logits"]
            finally:
                for layer, weight in originals.items():
                    hf.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight.data.copy_(weight)
            our_delta = edited - ours["logits"]
            worst["edit_delta"] = max(worst["edit_delta"], _max_abs(our_delta, ref_edited - ref["logits"]))
            worst["min_edit_effect"] = min(worst["min_edit_effect"], float(our_delta.abs().max().item()))
    return worst


REPORT: dict[str, dict[str, object]] = {}


@pytest.fixture(scope="module", params=sorted(CHECKPOINTS))
def bundle(request: pytest.FixtureRequest) -> Iterator[types.SimpleNamespace]:
    name = request.param
    env, default, expected_parameters = CHECKPOINTS[name]
    root = Path(os.environ.get(env, default))
    for required in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (root / required).is_file():
            pytest.skip(f"{name} checkpoint not found: {root / required} is missing; set {env} (see module docstring)")
    model = PythiaModel.from_pretrained_dir(root, model_name=f"EleutherAI/{name}")
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=str(root / "tokenizer.json"))
    hf = transformers.GPTNeoXForCausalLM.from_pretrained(
        root, attn_implementation="eager", dtype=torch.float64
    ).eval()
    reference_ids = [tokenizer(p)["input_ids"] for p in PROMPTS]
    # Forward levels use reference ids so a tokenizer failure can't hide or fake a numeric one.
    tokenized = [ids for ids in reference_ids if ids]
    stock = _measure(model, hf, tokenized)
    with _corrected_reference():
        corrected = _measure(model, hf, tokenized)
    REPORT[name] = {"stock": stock, "corrected": corrected, "transformers": transformers.__version__}
    yield types.SimpleNamespace(
        name=name,
        model=model,
        hf=hf,
        reference_ids=reference_ids,
        tokenized=tokenized,
        stock=stock,
        corrected=corrected,
        expected_parameters=expected_parameters,
    )


def test_prompt_set_is_large_and_varied() -> None:
    assert len(set(PROMPTS)) == len(PROMPTS) >= 50
    assert any(not p.isascii() for p in PROMPTS) and any("\n" in p for p in PROMPTS)


def test_same_weights_and_parameter_count(bundle: types.SimpleNamespace) -> None:
    hf_params = sum(int(p.numel()) for p in bundle.hf.parameters())
    assert bundle.model.total_parameters == hf_params == bundle.expected_parameters
    state = dict(bundle.hf.state_dict())
    state["embed_out.weight"] = bundle.hf.get_output_embeddings().weight
    for name, tensor in bundle.model.tensors.items():
        assert torch.equal(tensor, state[name]), name


def test_level1_tokenization(bundle: types.SimpleNamespace) -> None:
    mismatches = [
        (prompt, ours, ref)
        for prompt, ref in zip(PROMPTS, bundle.reference_ids, strict=True)
        if (ours := bundle.model.tokenizer.encode(prompt)) != ref
    ]
    assert not mismatches, mismatches[:5]


@pytest.mark.parametrize("level", LEVELS)
def test_levels_2_to_4_against_corrected_reference(bundle: types.SimpleNamespace, level: str) -> None:
    assert bundle.corrected[level] <= CORRECTED_TOLERANCE[level], bundle.corrected


@pytest.mark.parametrize("level", LEVELS)
def test_levels_2_to_4_against_stock_reference(bundle: types.SimpleNamespace, level: str) -> None:
    assert bundle.stock[level] <= STOCK_TOLERANCE[level], bundle.stock


def test_top1_agrees_with_stock_reference(bundle: types.SimpleNamespace) -> None:
    for ids in bundle.tokenized:
        ours = _our_trace(bundle.model, ids)["logits"].argmax(-1)
        assert torch.equal(ours, _reference_trace(bundle.hf, ids)["logits"].argmax(-1)), ids


def test_edits_really_change_logits(bundle: types.SimpleNamespace) -> None:
    assert bundle.corrected["min_edit_effect"] >= MIN_EDIT_EFFECT, bundle.corrected


def test_corrections_only_remove_float32_rounding(bundle: types.SimpleNamespace) -> None:
    """Guard the guard: the corrected reference must not be a different computation."""
    rotary = bundle.hf.gpt_neox.rotary_emb
    x = torch.zeros(1, dtype=torch.float64)
    positions = torch.arange(max(len(ids) for ids in bundle.tokenized))[None]
    with torch.no_grad():
        stock_cos, stock_sin = modeling.GPTNeoXRotaryEmbedding.forward(rotary, x, positions)
    fixed_cos, fixed_sin = _float64_rotary_forward(rotary, x, positions)
    assert _max_abs(stock_cos, fixed_cos) <= MAX_ROPE_TABLE_CHANGE
    assert _max_abs(stock_sin, fixed_sin) <= MAX_ROPE_TABLE_CHANGE

    scores = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    query = torch.eye(64, dtype=torch.float64)[None, None].repeat(1, 4, 1, 1)
    module = bundle.hf.gpt_neox.layers[0].attention
    stock_out, _ = modeling.eager_attention_forward(module, query, query, scores, None, 1.0)
    fixed_out, _ = _float64_eager_attention(module, query, query, scores, None, 1.0)
    assert _max_abs(stock_out, fixed_out) <= MAX_SOFTMAX_CHANGE


@pytest.mark.parametrize(
    ("label", "patch"),
    [
        ("rope only", lambda: setattr(modeling.GPTNeoXRotaryEmbedding, "forward", _float64_rotary_forward)),
        ("softmax only", lambda: setattr(modeling, "eager_attention_forward", _float64_eager_attention)),
    ],
)
def test_attribution_of_each_float32_step(bundle: types.SimpleNamespace, label: str, patch: Callable[[], None]) -> None:
    """Record what each correction removes alone. Only both together may reach machine precision."""
    rotary, eager = modeling.GPTNeoXRotaryEmbedding.forward, modeling.eager_attention_forward
    worst = 0.0
    patch()
    try:
        for ids in bundle.tokenized:
            ours = _our_trace(bundle.model, ids)["logits"]
            worst = max(worst, _max_abs(ours, _reference_trace(bundle.hf, ids)["logits"]))
    finally:
        modeling.GPTNeoXRotaryEmbedding.forward, modeling.eager_attention_forward = rotary, eager
    REPORT[bundle.name].setdefault("attribution_logits", {})[label] = worst  # type: ignore[union-attr]
    assert worst > CORRECTED_TOLERANCE["logits"], f"{label} alone explained everything; update the docstring"


def test_report() -> None:
    """Runs last in this file. Prints measurements, and writes them if NF_INFERENCE_REPORT is set."""
    if not REPORT:
        pytest.skip("no checkpoint was measured")
    print("\n" + json.dumps(REPORT, indent=2, sort_keys=True))
    destination = os.environ.get("NF_INFERENCE_REPORT")
    if destination:
        Path(destination).write_text(json.dumps(REPORT, indent=2, sort_keys=True), encoding="utf-8")
