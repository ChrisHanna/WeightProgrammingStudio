"""Check `export_gguf_v3` output against the published GGUF v3 layout using a separate parser.

These tests are about the container only. No GGUF runtime is involved, so passing them does not
qualify a file for llama.cpp; `test_real_runtimes.py` holds the test that would.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from neural_foundry import export_formats, pythia  # noqa: E402
from neural_foundry.prd_adapters import PRACTICE_RETENTION_PROMPTS  # noqa: E402

F32, F16, Q4, Q8 = 0, 1, 2, 8
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Model:
    def __init__(self, tensors: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        self.tensors = {k: v.to(torch.float64) for k, v in tensors.items()}
        self.config = config or {}
        self.model_name = "toy"


def _read(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], bytes]:
    blob = path.read_bytes()
    assert blob[:4] == b"GGUF"
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", blob, 4)
    assert version == 3
    pos = 24

    def text() -> str:
        nonlocal pos
        (n,) = struct.unpack_from("<Q", blob, pos)
        pos += 8 + n
        return blob[pos - n : pos].decode()

    kv: dict[str, Any] = {}
    for _ in range(n_kv):
        key = text()
        (kind,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        if kind == 8:
            kv[key] = text()
        else:
            (kv[key],) = struct.unpack_from(SCALARS[kind], blob, pos)
            pos += struct.calcsize(SCALARS[kind])
    infos = {}
    for _ in range(n_tensors):
        name = text()
        (ndim,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        dims = struct.unpack_from(f"<{ndim}Q", blob, pos)
        pos += 8 * ndim
        kind, offset = struct.unpack_from("<IQ", blob, pos)
        pos += 12
        infos[name] = {"dims": dims, "type": kind, "offset": offset}
    alignment = kv.get("general.alignment", 32)
    base = pos + (-pos % alignment)
    assert not any(blob[pos:base])
    return kv, infos, blob[base:]


def _export(tmp_path: Path, model: Model, quantization: str = "f32") -> tuple[dict[str, Any], dict[str, dict[str, Any]], bytes, Path]:
    path = export_formats.export_gguf_v3(model, tmp_path / "m.gguf", quantization=quantization, behaviour_override="container layout test")
    return (*_read(path), path)


PYTHIA_CONFIG = {
    "model_type": "gpt_neox", "hidden_size": 128, "num_hidden_layers": 6, "intermediate_size": 512,
    "max_position_embeddings": 2048, "num_attention_heads": 4, "layer_norm_eps": 1e-05,
    "rotary_pct": 0.25, "use_parallel_residual": True,
}


def test_gptneox_metadata_uses_every_key_the_spec_requires(tmp_path: Path) -> None:
    kv, _, _, _ = _export(tmp_path, Model({"w": torch.ones(2, 2)}, PYTHIA_CONFIG))
    assert kv["general.architecture"] == "gptneox"
    assert kv["gptneox.context_length"] == 2048
    assert kv["gptneox.embedding_length"] == 128
    assert kv["gptneox.block_count"] == 6
    assert kv["gptneox.feed_forward_length"] == 512
    assert kv["gptneox.use_parallel_residual"] == 1
    assert kv["gptneox.rope.dimension_count"] == 8
    assert kv["gptneox.attention.head_count"] == 4
    assert kv["gptneox.attention.layer_norm_epsilon"] == pytest.approx(1e-05)
    assert not any("Hidden_size" in key for key in kv)


def test_hyperparameters_absent_from_the_config_are_not_invented(tmp_path: Path) -> None:
    kv, _, _, _ = _export(tmp_path, Model({"w": torch.ones(2, 2)}))
    assert set(kv) == {"general.architecture", "general.name", "general.alignment"}
    assert kv["general.architecture"] == "nfunknown"
    assert kv["general.name"] == "toy"


@pytest.mark.parametrize("model_type", ["gpt_neox", "Weird-Type_2", "___"])
def test_architecture_stays_within_the_allowed_characters(tmp_path: Path, model_type: str) -> None:
    kv, _, _, _ = _export(tmp_path, Model({"w": torch.ones(2, 2)}, {"model_type": model_type}))
    assert kv["general.architecture"].isalnum() and kv["general.architecture"] == kv["general.architecture"].lower()


def test_dimensions_are_stored_in_ggml_order_and_reload_row_major(tmp_path: Path) -> None:
    tensor = torch.arange(6.0).reshape(2, 3)
    model = Model({"w": tensor})
    _, infos, data, path = _export(tmp_path, model)
    assert infos["w"]["dims"] == (3, 2)
    assert struct.unpack("<6f", data[:24]) == tuple(range(6))
    reloaded, _ = export_formats.reload_and_verify_gguf_v3(path, model, [])
    assert torch.equal(reloaded.tensors["w"], model.tensors["w"])


@pytest.mark.parametrize(("quantization", "kind"), [("q8_0", Q8), ("q4_0", Q4)])
def test_block_types_are_used_only_for_rows_of_whole_blocks(tmp_path: Path, quantization: str, kind: int) -> None:
    model = Model({"good": torch.linspace(-1, 1, 128).reshape(2, 64), "split": torch.linspace(-1, 1, 64).reshape(32, 2)})
    kv, infos, _, path = _export(tmp_path, model, quantization)
    assert infos["good"]["type"] == kind and infos["good"]["dims"] == (64, 2)
    assert infos["split"]["type"] == F32
    assert kv["general.quantization_version"] == 2
    export_formats.reload_and_verify_gguf_v3(path, model, [])


@pytest.mark.parametrize("quantization", ["f32", "f16"])
def test_unquantized_files_carry_no_quantization_version(tmp_path: Path, quantization: str) -> None:
    kv, infos, _, _ = _export(tmp_path, Model({"w": torch.ones(2, 32)}), quantization)
    assert "general.quantization_version" not in kv
    assert infos["w"]["type"] == (F32 if quantization == "f32" else F16)


def test_only_row_block_files_declare_quantization_version(tmp_path: Path) -> None:
    kv, infos, _, _ = _export(tmp_path, Model({"w": torch.ones(32, 2)}), "q8_0")
    assert infos["w"]["type"] == F32 and "general.quantization_version" not in kv


def test_q4_0_bytes_decode_with_ggml_nibble_order(tmp_path: Path) -> None:
    values = torch.tensor([[float((i % 15) - 7) for i in range(32)]])
    _, infos, data, _ = _export(tmp_path, Model({"w": values}), "q4_0")
    assert infos["w"]["type"] == Q4 and len(data) == 18
    (scale,) = struct.unpack_from("<e", data, 0)
    decoded = [0.0] * 32
    for j, byte in enumerate(data[2:18]):  # dequantize_row_q4_0: low nibble j, high nibble j + 16
        decoded[j] = ((byte & 0x0F) - 8) * scale
        decoded[j + 16] = ((byte >> 4) - 8) * scale
    assert torch.max(torch.abs(torch.tensor(decoded) - values[0])) <= scale


def test_q8_0_bytes_decode_as_scale_then_signed_values(tmp_path: Path) -> None:
    values = torch.linspace(-1, 1, 32).reshape(1, 32)
    _, _, data, _ = _export(tmp_path, Model({"w": values}), "q8_0")
    assert len(data) == 34
    (scale,) = struct.unpack_from("<e", data, 0)
    decoded = [scale * q for q in struct.unpack_from("<32b", data, 2)]
    assert torch.max(torch.abs(torch.tensor(decoded) - values[0])) <= scale


def test_tensors_the_format_cannot_hold_are_refused_on_export(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="too long"):
        export_formats.export_gguf_v3(Model({"n" * 64: torch.ones(2, 2)}), tmp_path / "a.gguf")
    with pytest.raises(ValueError, match="dimensions"):
        export_formats.export_gguf_v3(Model({"w": torch.ones(1, 1, 1, 1, 1)}), tmp_path / "b.gguf")


def _string(text: str) -> bytes:
    return struct.pack("<Q", len(text.encode())) + text.encode()


def _file(kv: bytes, n_kv: int, *, name: str = "w", dims: tuple[int, ...] = (2, 2), kind: int = F32, data: bytes = struct.pack("<4f", 1, 2, 3, 4)) -> bytes:
    out = b"GGUF" + struct.pack("<IQQ", 3, 1, n_kv) + kv + _string(name) + struct.pack("<I", len(dims))
    out += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", kind, 0)
    return out + b"\0" * (-len(out) % 32) + data


def _kv(key: str, kind: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", kind) + payload


METADATA_ACCEPTED = {
    "uint8": _kv("k", 0, b"\x01"),
    "int16": _kv("k", 3, struct.pack("<h", -1)),
    "bool": _kv("k", 7, b"\x01"),
    "uint64": _kv("k", 10, struct.pack("<Q", 2**40)),
    "float64": _kv("k", 12, struct.pack("<d", 0.5)),
    "string_array": _kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _string("a") + _string("b")),
    "scalar_array": _kv("tokenizer.ggml.scores", 9, struct.pack("<IQ", 6, 2) + struct.pack("<2f", 0.0, 1.0)),
    "empty_array": _kv("k", 9, struct.pack("<IQ", 4, 0)),
}


@pytest.mark.parametrize("case", METADATA_ACCEPTED)
def test_reader_accepts_every_published_metadata_type(tmp_path: Path, case: str) -> None:
    path = tmp_path / "m.gguf"
    path.write_bytes(_file(METADATA_ACCEPTED[case], 1))
    export_formats.reload_and_verify_gguf_v3(path, Model({"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}), [])


REFUSED = {
    "duplicate_key": _file(_kv("k", 0, b"\x01") * 2, 2),
    "bool_not_0_or_1": _file(_kv("k", 7, b"\x02"), 1),
    "nested_array": _file(_kv("k", 9, struct.pack("<IQ", 9, 1) + struct.pack("<IQ", 4, 0)), 1),
    "array_length_huge": _file(_kv("k", 9, struct.pack("<IQ", 4, 2**62)), 1),
    "array_element_type_unknown": _file(_kv("k", 9, struct.pack("<IQ", 99, 1)), 1),
    "array_cut_short": _file(_kv("k", 9, struct.pack("<IQ", 12, 1000)), 1),
    "alignment_below_eight": _file(_kv("general.alignment", 4, struct.pack("<I", 4)), 1),
    "too_many_dimensions": _file(b"", 0, dims=(1, 1, 1, 2, 2)),
    "name_too_long": _file(b"", 0, name="n" * 65),
    "q8_row_not_whole_blocks": _file(b"", 0, dims=(2, 16), kind=Q8, data=b"\0" * 34),
}


@pytest.mark.parametrize("case", REFUSED)
def test_reader_refuses_what_the_format_forbids(tmp_path: Path, case: str) -> None:
    path = tmp_path / "m.gguf"
    path.write_bytes(REFUSED[case])
    with pytest.raises(ValueError):
        export_formats.reload_and_verify_gguf_v3(path, Model({"w": torch.ones(2, 2)}), [])


def test_file_with_row_major_dimensions_is_refused_rather_than_misread(tmp_path: Path) -> None:
    path = tmp_path / "old.gguf"
    path.write_bytes(_file(b"", 0, dims=(2, 3), data=struct.pack("<6f", 0, 1, 2, 3, 4, 5)))
    with pytest.raises(ValueError, match="shape"):
        export_formats.reload_and_verify_gguf_v3(path, Model({"w": torch.ones(2, 3)}), [])


def test_real_checkpoint_round_trips_through_the_container(tmp_path: Path) -> None:
    source = Path(os.environ.get("NF_PYTHIA_DIRS", "/tmp/pythia-14m").split(os.pathsep)[0])
    if not (source / "model.safetensors").is_file():
        pytest.skip(f"checkpoint not found at {source}")
    model = pythia.PythiaModel.from_pretrained_dir(source)
    path = export_formats.export_gguf_v3(model, tmp_path / "m.gguf", quantization="q8_0", behaviour_override="container layout test")
    kv, infos, _ = _read(path)
    assert kv["general.architecture"] == "gptneox" and kv["gptneox.block_count"] == 6
    assert kv["gptneox.rope.dimension_count"] == 8
    assert kv["general.quantization_version"] == 2
    assert all(i["dims"][0] % 32 == 0 for i in infos.values() if i["type"] == Q8)
    assert any(i["type"] == Q8 for i in infos.values())
    reloaded, _ = export_formats.reload_and_verify_gguf_v3(path, model, [])
    for name, tensor in model.tensors.items():
        assert reloaded.tensors[name].shape == tensor.shape


class Behaving(Model):
    """Top-1 is decided by the summed weights, so quantization flips it only where a test arranges that."""

    threshold = 0.0

    def evaluate_prompt(self, prompt: str, *, target_token: str | None = None) -> Any:
        total = float(sum(t.sum() for t in self.tensors.values()))
        token = int(total > self.threshold)
        return SimpleNamespace(top1_token_id=token, top1_logit=total, matched_target=(str(token) == target_token))


PROBES = tuple(f"probe {n}" for n in range(25))


def _sensitive() -> Behaving:
    # Most weights sit far below the block's largest value, so 4 bits rounds them away.
    values = torch.full((1, 32), 0.01)
    values[0, 0] = 1.0
    model = Behaving({"w": values})
    model.threshold = float(values.sum()) - 0.05
    return model


@pytest.mark.parametrize("quantization", ["f16", "q8_0", "q4_0", "nf_q4"])
def test_lossy_export_without_evidence_is_refused_and_writes_nothing(tmp_path: Path, quantization: str) -> None:
    path = tmp_path / "m.gguf"
    with pytest.raises(export_formats.GgufBehaviourChangeError, match="no behaviour_probes"):
        export_formats.export_gguf_v3(Model({"w": torch.ones(2, 32)}), path, quantization=quantization)
    assert list(tmp_path.iterdir()) == []


def test_refused_export_leaves_an_existing_file_alone(tmp_path: Path) -> None:
    path = tmp_path / "m.gguf"
    path.write_bytes(b"keep")
    with pytest.raises(export_formats.GgufBehaviourChangeError):
        export_formats.export_gguf_v3(_sensitive(), path, quantization="q4_0", behaviour_probes=PROBES)
    assert path.read_bytes() == b"keep"


def test_too_few_probes_cannot_pass_the_gate(tmp_path: Path) -> None:
    model = Behaving({"w": torch.ones(2, 32)})
    with pytest.raises(export_formats.GgufBehaviourChangeError, match="minimum"):
        export_formats.export_gguf_v3(model, tmp_path / "m.gguf", quantization="q8_0", behaviour_probes=PROBES[:3])


def test_lost_task_accuracy_is_refused_even_when_agreement_is_allowed_to_fall(tmp_path: Path) -> None:
    items = [(p, "1") for p in PROBES]
    with pytest.raises(export_formats.GgufBehaviourChangeError, match="task accuracy fell"):
        export_formats.export_gguf_v3(_sensitive(), tmp_path / "m.gguf", quantization="q4_0", behaviour_probes=PROBES, task_items=items, min_top1_agreement=0.0)


def test_passing_gate_is_recorded_in_file_and_receipt(tmp_path: Path) -> None:
    model = Behaving({"w": torch.ones(2, 32)})
    path = export_formats.export_gguf_v3(model, tmp_path / "m.gguf", quantization="q8_0", behaviour_probes=PROBES)
    assert list(tmp_path.iterdir()) == [path]
    kv, _, _ = _read(path)
    assert kv["nf.behaviour_status"] == "verified" and kv["nf.probe_count"] == 25 and kv["nf.top1_agreement"] == 1.0
    _, receipt = export_formats.reload_and_verify_gguf_v3(path, model, PROBES)
    assert receipt.behaviour_status == "verified" and receipt.behaviour_override is None
    assert receipt.gain_preserved_after_rebuild is True


def test_override_of_a_failed_gate_is_recorded_and_marks_the_receipt(tmp_path: Path) -> None:
    model = _sensitive()
    path = export_formats.export_gguf_v3(model, tmp_path / "m.gguf", quantization="q4_0", behaviour_probes=PROBES, behaviour_override="size matters more here")
    kv, _, _ = _read(path)
    assert kv["nf.behaviour_status"] == "override_failed_gate"
    assert kv["nf.behaviour_override"] == "size matters more here"
    assert kv["nf.top1_agreement"] < 1.0
    _, receipt = export_formats.reload_and_verify_gguf_v3(path, model, PROBES)
    assert receipt.behaviour_status == "override_failed_gate"
    assert receipt.behaviour_override == "size matters more here"
    assert receipt.export_behaviour is not None and receipt.export_behaviour["top1_agreement"] < 1.0
    assert receipt.gain_preserved_after_rebuild is False


def test_failed_gate_stays_marked_even_when_reload_probes_all_match(tmp_path: Path) -> None:
    model = Behaving({"w": torch.ones(2, 32)})
    path = export_formats.export_gguf_v3(model, tmp_path / "m.gguf", quantization="q8_0", behaviour_probes=PROBES[:3], behaviour_override="few prompts on purpose")
    _, receipt = export_formats.reload_and_verify_gguf_v3(path, model, PROBES)
    assert receipt.behavioral_match_rate == 1.0
    assert receipt.gain_preserved_after_rebuild is False


def test_unmeasured_override_does_not_claim_preservation_even_when_reload_probes_all_match(tmp_path: Path) -> None:
    model = Behaving({"w": torch.ones(2, 32)})
    path = export_formats.export_gguf_v3(model, tmp_path / "m.gguf", quantization="q8_0", behaviour_override="no probes on purpose")
    _, receipt = export_formats.reload_and_verify_gguf_v3(path, model, PROBES)
    assert receipt.behaviour_status == "override_unverified"
    assert receipt.behavioral_match_rate == 1.0
    assert receipt.gain_preserved_after_rebuild is False


@pytest.mark.parametrize("reason", ["", "   "])
def test_override_needs_a_reason(tmp_path: Path, reason: str) -> None:
    with pytest.raises(ValueError, match="reason"):
        export_formats.export_gguf_v3(Model({"w": torch.ones(2, 32)}), tmp_path / "m.gguf", quantization="q8_0", behaviour_override=reason)


def test_f32_without_probes_records_no_behaviour_claim(tmp_path: Path) -> None:
    model = Model({"w": torch.ones(2, 2)})
    path = export_formats.export_gguf_v3(model, tmp_path / "m.gguf")
    assert not any(key.startswith("nf.") for key in _read(path)[0])
    _, receipt = export_formats.reload_and_verify_gguf_v3(path, model, [])
    assert receipt.behaviour_status == "not_recorded"


def test_real_checkpoint_4bit_export_is_refused_on_many_prompts(tmp_path: Path) -> None:
    source = Path(os.environ.get("NF_PYTHIA_DIRS", "/tmp/pythia-14m").split(os.pathsep)[0])
    if not (source / "model.safetensors").is_file():
        pytest.skip(f"checkpoint not found at {source}")
    model = pythia.PythiaModel.from_pretrained_dir(source)
    probes = list(dict.fromkeys(PRACTICE_RETENTION_PROMPTS))
    assert len(probes) >= export_formats.GGUF_MIN_BEHAVIOUR_PROBES
    for quantization in ("q4_0", "nf_q4"):
        with pytest.raises(export_formats.GgufBehaviourChangeError, match="top-1 agreement"):
            export_formats.export_gguf_v3(model, tmp_path / "m.gguf", quantization=quantization, behaviour_probes=probes)
