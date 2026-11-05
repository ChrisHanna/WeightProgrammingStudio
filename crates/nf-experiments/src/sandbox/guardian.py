"""Fixed OCI guardian; never imported into the coordinator or gym.

Only the Rust launcher's actual capability preflight enables this profile.
Untrusted Python runs as a separate non-root process, never an editor callback.
"""

import json
import os
import selectors
import signal
import subprocess
import time
from pathlib import Path


def cpu_usec():
    fields = dict(line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
    return int(fields["usage_usec"])


def clock():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def main():
    # This file is only invoked by the fixed Docker entrypoint. Root is confined
    # to this private PID/mount/network namespace; the untrusted child is nobody.
    if os.environ.get("NF_CONTAINER_GUARDIAN") != "v1" or os.getuid() != 0:
        raise SystemExit("guardian requires its qualified private container")
    policy = json.loads(Path("/guard/policy.json").read_bytes())
    start = clock()
    initial_cpu = cpu_usec()
    maximum = policy["output_bytes"]
    with Path("/job/input.json").open("rb") as job_input:
        child = subprocess.Popen(
            ["/usr/local/bin/python3", "-I", "-S", "-B", "/job/main.py"],
            stdin=job_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/out",
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent", "TMPDIR": "/tmp"},
            user=65534,
            group=65534,
            extra_groups=[],
            start_new_session=True,
            close_fds=True,
        )
    selector = selectors.DefaultSelector()
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    for name in outputs:
        stream = getattr(child, name)
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    status = "completed"
    cleanup_deadline = None
    while child.poll() is None or selector.get_map():
        elapsed = (clock() - start) * 1000
        cpu_ms = (cpu_usec() - initial_cpu) // 1000
        if cleanup_deadline is None and (
            elapsed >= policy["wall_ms"] or cpu_ms >= policy["cpu_ms"]
        ):
            status = "timed_out" if elapsed >= policy["wall_ms"] else "cpu_exceeded"
            cleanup_deadline = clock() + 0.1
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child.poll() is not None and cleanup_deadline is None:
            cleanup_deadline = clock() + 0.1
            # A normal leader exit cannot leave a model/tool process behind.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if cleanup_deadline is not None and clock() >= cleanup_deadline:
            break
        for key, _ in selector.select(0.01):
            data = os.read(key.fileobj.fileno(), 4096)
            if not data:
                selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            name = key.data
            room = maximum - sum(map(len, outputs.values()))
            outputs[name].extend(data[: max(0, room)])
            if len(data) > room and status == "completed":
                status = "output_exceeded"
                cleanup_deadline = clock() + 0.1
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    try:
        exit_code = child.wait(timeout=0.1)
    except subprocess.TimeoutExpired:
        exit_code = None
        status = "incomplete"
    if status == "completed" and exit_code != 0:
        status = "failed"
    # Final samples close the race where the child exits between monitor ticks.
    # An over-budget successful exit must not masquerade as completed work.
    if status == "completed" and (clock() - start) * 1000 >= policy["wall_ms"]:
        status = "timed_out"
    if status == "completed" and (cpu_usec() - initial_cpu) // 1000 >= policy["cpu_ms"]:
        status = "cpu_exceeded"
    # Escaped sessions still live in this container's PID namespace. Exiting
    # the primary guardian causes docker-init/PID1 exit and kernel tree cleanup.
    report = {
        "schema_version": 1,
        "kind": "sandbox_execution",
        "status": status,
        "exit_code": exit_code,
        "stdout_hex": bytes(outputs["stdout"]).hex(),
        "stderr_hex": bytes(outputs["stderr"]).hex(),
        "wall_ms": int((clock() - start) * 1000),
        "cpu_ms": (cpu_usec() - initial_cpu) // 1000,
        "memory_peak_bytes": int(Path("/sys/fs/cgroup/memory.peak").read_text()),
        "measurement_scope": "private_container_cgroup_v2",
    }
    print(json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
