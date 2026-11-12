"""Launch the operator-qualified WSL worker without putting secrets in arguments."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[3]
UNIT = "neural-foundry-local-training.service"


def systemd_argument(value: str) -> str:
    if "\n" in value or "\x00" in value:
        raise ValueError("Invalid service path")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--database-port", type=int, default=55432)
    parser.add_argument("--api-port", type=int, default=8787)
    parser.add_argument("--install-user-service", action="store_true")
    parser.add_argument("--replace-user-service", action="store_true")
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.database_port, args.api_port)):
        raise SystemExit("Ports must be between 1 and 65535.")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    interpreter = Path(config["python"])
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise SystemExit("The configured operator interpreter is unavailable.")
    if config.get("qualification", {}).get("complete") is not True:
        raise SystemExit("Complete the explicitly approved qualification before starting a worker.")
    if args.install_user_service:
        directory = Path.home() / ".config" / "systemd" / "user"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / UNIT
        command = [str(interpreter), str(Path(__file__).resolve()), "--config", str(config_path),
                   "--database-port", str(args.database_port), "--api-port", str(args.api_port)]
        content = (
            "[Unit]\nDescription=Neural Foundry qualified local training worker\n"
            "After=network.target\n\n[Service]\nType=simple\n"
            "WorkingDirectory=" + str(ROOT).replace("%", "%%") + "\n"
            "ExecStart=" + " ".join(systemd_argument(value) for value in command) + "\n"
            "Restart=on-failure\nRestartSec=10\nTimeoutStopSec=30\n"
            "KillMode=control-group\nUMask=0077\nNoNewPrivileges=true\n\n"
            "[Install]\nWantedBy=default.target\n"
        )
        if path.exists() and args.replace_user_service:
            previous = path.read_text()
            if "Description=Neural Foundry qualified local training worker" not in previous or systemd_argument(str(Path(__file__).resolve())) not in previous:
                raise SystemExit("Refusing to replace a service not owned by this launcher.")
        with path.open("w" if args.replace_user_service else "x") as stream:
            stream.write(content)
        subprocess.run(["systemd-analyze", "--user", "verify", str(path)], check=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", UNIT], check=True)
        subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT], check=True)
        print("Installed and started " + UNIT + ". No model job was submitted.")
        return
    secrets = ROOT / "artifacts" / "control-plane-local" / "secrets"
    password = (secrets / "postgres_password").read_text().strip()
    environment = dict(os.environ)
    environment.update({
        "PATH": "/usr/lib/wsl/lib:" + str(interpreter.parent) + ":" + environment.get("PATH", os.defpath),
        "PYTHONPATH": str(ROOT / "python"),
        "CP_DATABASE_URL": f"postgresql://control_plane:{quote(password, safe='')}@127.0.0.1:{args.database_port}/control_plane",
        "CP_OPERATOR_TOKEN_FILE": str(secrets / "operator_token"),
        "CP_ARTIFACT_API_URL": f"http://127.0.0.1:{args.api_port}",
        "CP_WORKER_ID": "local-training-" + config["id"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })
    os.chdir(ROOT)
    os.execve(str(interpreter), [str(interpreter), "-m",
        "neural_foundry.control_plane.local_runtime", "worker", "--config", str(config_path)], environment)


if __name__ == "__main__":
    main()
