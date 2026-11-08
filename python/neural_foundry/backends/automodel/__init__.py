"""Optional NeMo AutoModel full-SFT backend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .. import BackendFactory, registry
from .config import AUTOMODEL_PYTHON, AUTOMODEL_VERSIONS, AutoModelBackendConfig


def _available(config: AutoModelBackendConfig | None = None) -> bool:
	python = config.python_executable if config is not None else os.environ.get("NF_AUTOMODEL_PYTHON", sys.executable)
	if not python or not Path(python).is_file() or not (Path(python).parent / "automodel").is_file():
		return False
	code = ("import importlib.metadata as m,json,sys; "
			"names=('nemo-automodel','torch','triton','transformers','torchao','torchdata','datasets'); "
			"print(json.dumps({'versions':{x:m.version(x) for x in names},'python':list(sys.version_info[:3])}))")
	try:
		result = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=15,
								check=False, env={**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1"})
		return result.returncode == 0 and json.loads(result.stdout) == {
			"versions": AUTOMODEL_VERSIONS, "python": list(AUTOMODEL_PYTHON),
		}
	except (OSError, ValueError, subprocess.SubprocessError):
		return False


def _create(config: AutoModelBackendConfig):
	from .adapter import automodel_sft_trainer

	return automodel_sft_trainer(config)


registry.register(BackendFactory("nemo_automodel.sft", "1", {}, _create, _available))