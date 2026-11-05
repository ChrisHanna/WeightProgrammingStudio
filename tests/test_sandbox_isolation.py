"""Real local OCI tests. Missing prerequisites fail, never qualify through skips."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

BINARY = Path(__file__).resolve().parents[1] / "target/debug/examples/fixture_sandbox"


def invoke(root: Path, source: str = "", **options: Any) -> dict[str, Any]:
    root.mkdir(mode=0o700, exist_ok=True)
    request = {"root": str(root), "enabled": True, "source": source, **options}
    result = subprocess.run(
        [str(BINARY)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
        env={**os.environ, "NF_HOST_SECRET": "fixture-controller-secret"},
    )
    reply: dict[str, Any] = json.loads(result.stdout)
    assert "error" not in reply, reply
    return dict(reply["ok"])


def output(receipt: dict[str, Any]) -> str:
    assert receipt["cleanup_confirmed"] is True
    assert receipt["outputs_eligible"] is False
    assert receipt["promotion_status"] == "not_evaluated"
    return bytes.fromhex(receipt["report"]["stdout_hex"]).decode()


def test_sandbox_isolation_symlink_controller_root_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    result = subprocess.run(
        [str(BINARY)],
        input=json.dumps({"root": str(alias), "enabled": True, "qualify_only": True}),
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    assert json.loads(result.stdout)["error"]["code"] == "SCOPE_VIOLATION"
    assert not list(real.iterdir())


def test_sandbox_isolation_default_off_before_code_or_runtime(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    result = subprocess.run(
        [str(BINARY)],
        input=json.dumps(
            {"root": "/not/a/controller/root", "source": f"open({str(marker)!r},'w').write('bad')"}
        ),
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    assert json.loads(result.stdout)["error"]["message"] == "UNTRUSTED_GYM_EXECUTION_DISABLED"
    assert not marker.exists()


def test_sandbox_isolation_actual_uid_caps_network_guardian_and_cgroup(tmp_path: Path) -> None:
    receipt = invoke(
        tmp_path / "private",
        """import ctypes,json,os,signal,socket
from pathlib import Path
s=Path('/proc/self/status').read_text()
assert os.getuid()==65534 and 'CapEff:\\t0000000000000000' in s
assert 'NoNewPrivs:\\t1' in s and 'Seccomp:\\t2' in s
assert 'NF_HOST_SECRET' not in os.environ
assert not Path('/var/run/docker.sock').exists() and not Path('/dev/nvidia0').exists()
assert Path('/sys/fs/cgroup/pids.max').read_text().strip()=='16'
assert Path('/sys/fs/cgroup/memory.max').read_text().strip()=='134217728'
for target in [0,os.getppid(),1]:
 try:
  if target==0: os.setuid(0)
  else: os.kill(target,signal.SIGKILL)
  raise AssertionError('privilege escape')
 except PermissionError: pass
libc=ctypes.CDLL(None,use_errno=True)
assert libc.mount(b'none',b'/out',b'tmpfs',0,None)==-1
assert ctypes.get_errno() in [1,13]
assert libc.unshare(0x10000000)==-1
sock=socket.socket();sock.settimeout(.1)
try: sock.connect(('198.51.100.1',9));raise AssertionError('network escape')
except OSError: pass
print(json.dumps({'safe':True,'result':2+2}))
""",
    )
    assert receipt["status"] == "completed", receipt
    assert json.loads(output(receipt)) == {"safe": True, "result": 4}
    assert receipt["report"]["measurement_scope"] == "private_container_cgroup_v2"
    assert receipt["report"]["memory_peak_bytes"] > 0


def test_sandbox_isolation_mount_symlink_archive_hidden_asset_escape(tmp_path: Path) -> None:
    secret = tmp_path / "controller-secret"
    secret.write_text("hidden-host-fixture")
    source = f"""import io,os,tarfile
from pathlib import Path
secret={str(secret)!r}
assert not Path(secret).exists()
os.symlink(secret,'/out/link')
try: Path('/out/link').read_bytes();raise AssertionError('symlink escaped')
except FileNotFoundError: pass
for target in ['/job/main.py','/guard/policy.json','/etc/nf-escape','/proc/1/root/etc/nf-escape']:
 try: open(target,'wb');raise AssertionError('write escaped')
 except OSError: pass
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as archive:
 info=tarfile.TarInfo('../../etc/nf-tar-escape');info.size=1
 archive.addfile(info,io.BytesIO(b'x'))
buf.seek(0)
try:
 with tarfile.open(fileobj=buf) as archive: archive.extractall('/out',filter='fully_trusted')
 raise AssertionError('archive escaped')
except OSError: pass
assert not Path('/etc/nf-tar-escape').exists()
print('denied')
"""
    receipt = invoke(tmp_path / "private", source)
    assert receipt["status"] == "completed", receipt
    assert output(receipt) == "denied\n"
    assert secret.read_text() == "hidden-host-fixture"


def test_sandbox_isolation_combined_output_limit(tmp_path: Path) -> None:
    receipt = invoke(
        tmp_path / "private",
        "import os\nos.write(1,b'x'*1000000)",
        limits={"wall_ms": 2000, "cpu_ms": 2000, "output_bytes": 128},
    )
    assert receipt["status"] == "output_exceeded", receipt
    assert len(output(receipt)) <= 128


def test_sandbox_isolation_wall_and_cumulative_cpu_limits(tmp_path: Path) -> None:
    wall = invoke(
        tmp_path / "wall",
        "import time\ntime.sleep(60)",
        limits={"wall_ms": 200, "cpu_ms": 2000, "output_bytes": 128},
    )
    assert wall["status"] == "timed_out", wall
    output(wall)
    cpu = invoke(
        tmp_path / "cpu",
        "while True: pass",
        limits={"wall_ms": 3000, "cpu_ms": 200, "output_bytes": 128},
    )
    assert cpu["status"] == "cpu_exceeded", cpu
    output(cpu)
    assert cpu["report"]["cpu_ms"] >= 200


def test_sandbox_isolation_memory_limit_is_not_a_skip(tmp_path: Path) -> None:
    receipt = invoke(tmp_path / "private", "x=bytearray(256*1024*1024)\nprint('unsafe')")
    assert receipt["status"] in {"failed", "incomplete"}, receipt
    assert receipt["cleanup_confirmed"] is True
    assert receipt["outputs_eligible"] is False
    assert receipt.get("report") is None or "unsafe" not in output(receipt)


def test_sandbox_isolation_process_limit_and_escaped_session_cleanup(tmp_path: Path) -> None:
    receipt = invoke(
        tmp_path / "private",
        """import json,os,time
children=[]
for i in range(64):
 try: pid=os.fork()
 except BlockingIOError: break
 if pid==0:
  os.setsid();time.sleep(60);os._exit(0)
 children.append(pid)
assert 0<len(children)<16
print(json.dumps({'children':len(children)}),flush=True)
os._exit(0)
""",
    )
    assert receipt["status"] == "completed", receipt
    assert 0 < json.loads(output(receipt))["children"] < 16
    ps = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^/{receipt['job']}$", "--format", "{{.ID}}"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert not ps.stdout.strip()


def test_sandbox_isolation_file_size_and_total_writable_storage(tmp_path: Path) -> None:
    receipt = invoke(
        tmp_path / "private",
        """import errno,os
from pathlib import Path
try: Path('/out/big').write_bytes(b'x'*131072);raise AssertionError('file limit ignored')
except OSError as e: assert e.errno==errno.EFBIG
assert Path('/out/big').stat().st_size<=65536
for i in range(300):
 try: Path('/out/f'+str(i)).write_bytes(b'x'*65536)
 except OSError as e:
  assert e.errno==errno.ENOSPC
  break
else: raise AssertionError('tmpfs limit ignored')
print('bounded')
""",
    )
    assert receipt["status"] == "completed", receipt
    assert output(receipt) == "bounded\n"


def test_sandbox_isolation_cancel_does_not_publish(tmp_path: Path) -> None:
    root = tmp_path / "private"
    receipt = invoke(
        root,
        "import time\nprint('partial checkpoint',flush=True)\ntime.sleep(60)",
        cancel_after_ms=2000,
        limits={"wall_ms": 5000, "cpu_ms": 5000, "output_bytes": 4096},
    )
    assert receipt["status"] == "cancelled", receipt
    assert receipt["cleanup_confirmed"] is True
    assert receipt["report"] is None and receipt["outputs_eligible"] is False
    state = json.loads((root / receipt["job"] / "pre_cleanup.json").read_bytes())["State"]
    assert state["StartedAt"] != "0001-01-01T00:00:00Z"


def test_sandbox_isolation_real_controller_crash_recovery(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    result = subprocess.run(
        [str(BINARY)],
        input=json.dumps(
            {
                "root": str(root),
                "enabled": True,
                "source": "import time\ntime.sleep(60)",
                "limits": {"wall_ms": 5000, "cpu_ms": 5000, "output_bytes": 128},
                "crash_after_ms": 2000,
            }
        ),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 86, result.stderr
    pending = [p for p in root.glob("*/intent.json") if not p.with_name("completion.json").exists()]
    assert len(pending) == 1
    assert pending[0].with_name("configured.json").exists()
    recovered = invoke(root, recover=True)
    # Qualification reconciles stale jobs before reserving its own probe.
    assert recovered["recovered"] == 0
    evidence = json.loads(pending[0].with_name("recovery.json").read_bytes())
    assert evidence["schema_version"] == 3
    assert evidence["cleanup_confirmed"] is True
    assert evidence["status"] == "incomplete"
    assert evidence["outputs_eligible"] is False
    assert evidence["report"] is None
    storage = evidence["storage"]
    assert storage["workspace_reclaimed"] is True
    assert storage["accounting_finalized"] is True
    assert storage["capacity_guaranteed"] is False
    assert storage["retained_bytes"] > 0
    assert storage["root_retained_bytes"] >= storage["retained_bytes"]
    journal = pending[0].parent
    assert not (journal / "job").exists()
    assert not (journal / "guard").exists()
    assert not (journal / "reservation.bin").exists()
    assert not (journal / "receipt.tmp").exists()
    assert invoke(root, recover=True)["recovered"] == 0
