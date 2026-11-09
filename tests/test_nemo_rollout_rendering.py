from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.xfail(strict=True, reason="Known gap: stock NeMo rollout rendering is not routed through our token-ID path")
def test_stock_nemo_completions_mode_uses_our_token_id_path() -> None:
    python = os.environ.get("NF_NEMO_GYM_PYTHON")
    if not python:
        pytest.skip("set NF_NEMO_GYM_PYTHON to the pinned NeMo interpreter")
    code = '''
from types import SimpleNamespace
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
config = VLLMModelConfig.model_fields
assert config['use_completions_api'].default is False
assert config['render_chat_template'].default is False
fake = SimpleNamespace(config=SimpleNamespace(model='test', extra_body=None, return_token_id_information=False),
                       _apply_sampling_overrides=lambda value:value)
body = VLLMModel._build_completion_body_from_chat_body(fake, {}, '<rendered chat>')
assert isinstance(body['prompt'], list), 'stock NeMo still forwards rendered text, not token IDs'
assert body.get('add_special_tokens') is False
'''
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr