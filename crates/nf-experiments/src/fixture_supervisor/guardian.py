"""Static native fixture guardian, composed with boundary.py by the controller.

No generated Python, dynamic runtime, filesystem, network, GPU or ledger access.
Prepare freezes bytes and probes kernel enforcement, but NEVER starts work.
Controller must durably persist the returned identity and recheck lease/fence
before sending exactly that identity as authorization. No fsync claim is made.
"""
import ctypes
import errno
import hashlib
import json
import os
import platform
import resource
import selectors
import signal
import stat
import sys
import time

PROFILE = "static_elf_fd_cpu_linux_v2"
MAX_OUTPUT = 65536


def now():
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def boot_id():
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
        return stream.read(64).strip()


def validate(policy):
    if sys.platform != "linux" or os.geteuid() == 0:
        raise ValueError("requires non-root Linux")
    if policy["profile"] != PROFILE:
        raise ValueError("hostile gyms and other profiles require a qualified sandbox")
    if policy["boot_id"] != boot_id():
        raise ValueError("boot changed; elapsed authority cannot be restarted")
    bounds = {"memory_bytes": (32 << 20, 1 << 40), "cpu_seconds": (1, 86400),
              "output_bytes": (1, MAX_OUTPUT), "grace_ms": (1, 1000),
              "reap_ms": (1, 1000), "deadline_boottime_ns": (1, (1 << 63) - 1)}
    for key, (low, high) in bounds.items():
        value = policy[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError("invalid finite limit: " + key)
    if not now() < policy["deadline_boottime_ns"] <= now() + 86400 * 10**9:
        raise ValueError("expired or unsupported deadline")
    if platform.machine() not in ("x86_64", "aarch64"):
        raise ValueError("unsupported seccomp architecture")
    args = policy["argv"]
    if not isinstance(args, list) or not args or len(args) > 64:
        raise ValueError("invalid trusted argv")
    if any(not isinstance(arg, str) or "\0" in arg for arg in args):
        raise ValueError("invalid trusted argument")
    if not os.path.isabs(args[0]):
        raise ValueError("executable must be absolute")
    payload = bytes.fromhex(policy.get("input_hex", ""))
    if len(payload) > 4096:
        raise ValueError("input exceeds bounded fixture protocol")
    binding = policy["binding"]
    if set(binding) != {"contract", "reservation", "attempt", "lease", "fencing_epoch", "nonce", "lease_deadline_boottime_ns"}:
        raise ValueError("closed launch binding required")
    for key in ("contract", "reservation", "attempt", "lease"):
        if not isinstance(binding[key], str) or not 1 <= len(binding[key]) <= 128:
            raise ValueError("bounded nonempty launch identity required")
    if (type(binding["fencing_epoch"]) is not int or not 1 <= binding["fencing_epoch"] < 2**64
            or type(binding["lease_deadline_boottime_ns"]) is not int
            or policy["deadline_boottime_ns"] > binding["lease_deadline_boottime_ns"]
            or not isinstance(binding["nonce"], str) or len(binding["nonce"]) != 64
            or any(c not in "0123456789abcdef" for c in binding["nonce"])):
        raise ValueError("invalid fence, lease deadline or 256-bit launch nonce")


class Worker:
    """Nonblocking fork/exec: stalled executable I/O cannot stall the watchdog."""

    def __init__(self, executable, policy):
        out_r, out_w = os.pipe2(os.O_CLOEXEC)
        err_r, err_w = os.pipe2(os.O_CLOEXEC)
        in_r, in_w = os.pipe2(os.O_CLOEXEC)
        parent = os.getpid()
        try:
            self.pid = os.fork()
        except BaseException:
            for fd in (out_r, out_w, err_r, err_w, in_r, in_w):
                os.close(fd)
            raise
        if self.pid == 0:
            try:
                os.dup2(in_r, 0)
                os.dup2(out_w, 1)
                os.dup2(err_w, 2)
                for fd in (in_r, in_w, out_r, out_w, err_r, err_w):
                    os.close(fd)
                os.chdir("/")
                child_boundary(parent, policy, executable)
                os.execve(executable, policy["argv"],
                          {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"})
            except BaseException:
                os._exit(125)
        os.close(out_w)
        os.close(err_w)
        os.close(in_r)
        payload = bytes.fromhex(policy.get("input_hex", ""))
        try:
            if payload:
                os.write(in_w, payload)  # <= PIPE_BUF, initially empty, never waits for a reader.
        except BrokenPipeError:
            pass
        finally:
            os.close(in_w)
        self.stdout = os.fdopen(out_r, "rb", buffering=0)
        self.stderr = os.fdopen(err_r, "rb", buffering=0)
        self.returncode = None
        self.usage = None

    def poll(self):
        if self.returncode is None:
            pid, status, usage = os.wait4(self.pid, os.WNOHANG)
            if pid:
                self.returncode = os.waitstatus_to_exitcode(status)
                self.usage = usage
        return self.returncode

    def send(self, signum):
        # Only this single-threaded parent reaps. An unreaped PID cannot be reused.
        if self.poll() is None:
            os.kill(self.pid, signum)

    def terminate(self):
        self.send(signal.SIGTERM)

    def kill(self):
        self.send(signal.SIGKILL)


def run(policy):
    validate(policy)
    # The guardian is trusted but not unbudgeted. Keep a separate finite
    # address-space ceiling that still permits lowering the child's AS limit.
    guardian_memory = policy["memory_bytes"] + (64 << 20)
    resource.setrlimit(resource.RLIMIT_AS, (guardian_memory, guardian_memory))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    qualified_credentials = qualify_runtime()
    executable = freeze_executable(policy)
    # Probe the exact kernel boundary with TRUSTED code only. No worker executes
    # before the kernel filter/FD closure and immutable image are qualified.
    probe = os.fork()
    if probe == 0:
        try:
            child_boundary(os.getppid(), policy, executable)
            os._exit(0)
        except BaseException:
            os._exit(125)
    _, probe_status, _ = os.wait4(probe, 0)
    if probe_status != 0:
        os.close(executable)
        raise ValueError("kernel seccomp/close_range/resource boundary probe failed")
    with open("/proc/self/stat", encoding="ascii") as stream:
        start_ticks = int(stream.read().rsplit(")", 1)[1].split()[19])
    identity = {"policy": policy, "guardian_pid": os.getpid(),
                "guardian_start_ticks": start_ticks,
                "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()}
    print(json.dumps({"kind": "prepared", "identity": identity}, separators=(",", ":")), flush=True)
    deadline = policy["deadline_boottime_ns"]  # Never recomputed after dispatch.
    events = {"supervisor_started_ns": now()}
    result = {"identity": identity, "qualified_credentials": qualified_credentials,
              "profile": PROFILE, "boot_id": policy["boot_id"],
              "deadline_boottime_ns": deadline, "status": "incomplete",
              "cleanup_confirmed": False, "exit_code": None,
              "memory_scope": "kernel_RLIMIT_AS_per_process_not_RSS_or_GPU",
              "events": events, "stdout_hex": "", "stderr_hex": ""}
    selector = selectors.DefaultSelector()
    os.set_blocking(0, False)
    selector.register(0, selectors.EVENT_READ, "control")
    child = None
    control_buffer = bytearray()
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    stop_at = None
    kill_at = None
    end_at = None

    def stop(reason):
        nonlocal stop_at, kill_at, end_at
        if stop_at is None:
            result["status"] = reason
            stop_at = now()
            events["stop_observed_ns"] = stop_at
            kill_at = stop_at + policy["grace_ms"] * 10**6
            end_at = kill_at + policy["reap_ms"] * 10**6
            if child is not None:
                child.terminate()  # SIGTERM is the cooperative-stop protocol.
                events["cancel_sent_ns"] = now()

    def observe_pending_control():
        if 0 in selector.get_map():
            try:
                pending = os.read(0, 16)
            except BlockingIOError:
                return
            stop("control_lost" if not pending else
                 "cancelled" if pending.strip(b"S") == b"" else "invalid_control")

    try:
        while True:
            current = now()
            if current >= deadline:
                stop("timed_out")
            if stop_at is not None and child is None:
                # Preparation's trusted probe was synchronously reaped and no
                # worker was authorized. This is known cleanup, not lost work.
                result["cleanup_confirmed"] = True
                events["cleanup_confirmed_ns"] = now()
                break
            if child is not None:
                if stop_at is not None and current >= kill_at and "kill_sent_ns" not in events:
                    child.kill()
                    events["kill_sent_ns"] = now()
                code = child.poll()  # Reaps only this child; never signals a stale PID/group.
                streams_open = any(key.data != "control" for key in selector.get_map().values())
                if code is not None and not streams_open:
                    # Observe a pending stop/EOF before recording completion.
                    observe_pending_control()
                    result["exit_code"] = code
                    result["cleanup_confirmed"] = True
                    events["cleanup_confirmed_ns"] = now()
                    if stop_at is None:
                        result["status"] = "completed" if code == 0 else "failed"
                        if now() >= deadline:
                            result["status"] = "timed_out"
                    break
                if end_at is not None and current >= end_at:
                    result["exit_code"] = code
                    break  # UNKNOWN cleanup retains the complete reservation.
            for key, _ in selector.select(0.005):
                if key.data == "control":
                    data = os.read(0, 16385)
                    if not data:
                        stop("control_lost")
                        selector.unregister(0)
                    elif data.strip(b"S") == b"":
                        stop("cancelled")  # Caller sent only AFTER durable stop event.
                    elif child is None and stop_at is None:
                        control_buffer.extend(data)
                        if len(control_buffer) > 16384:
                            stop("invalid_control")
                            continue
                        if b"\n" not in control_buffer:
                            continue
                        line, _, pending = control_buffer.partition(b"\n")
                        authorization = json.loads(line)
                        control_buffer.clear()
                        if authorization != {"authorize": identity}:
                            stop("invalid_control")
                            continue
                        # Pipes may coalesce authorize + immediate stop in one
                        # read. Preserve the stop instead of parsing trailing
                        # control bytes as JSON or accidentally starting work.
                        if pending:
                            stop("cancelled" if pending.strip(b"S") == b"" else "invalid_control")
                            continue
                        if now() >= deadline:
                            stop("timed_out")
                            continue
                        observe_pending_control()  # Hashing may have outlived controller authority.
                        if stop_at is not None:
                            continue
                        child = Worker(executable, policy)
                        os.close(executable)
                        executable = None
                        events["worker_started_ns"] = now()
                        for name in outputs:
                            stream = getattr(child, name)
                            os.set_blocking(stream.fileno(), False)
                            selector.register(stream, selectors.EVENT_READ, name)
                    else:
                        stop("invalid_control")
                else:
                    data = os.read(key.fileobj.fileno(), 4096)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    else:
                        room = policy["output_bytes"] - sum(map(len, outputs.values()))
                        outputs[key.data].extend(data[:room])
                        if len(data) > room:
                            stop("output_exceeded")
    except Exception as error:
        result["status"] = "supervisor_error"
        result["error"] = type(error).__name__  # No unbounded/path-bearing exception text.
    finally:
        if executable is not None:
            os.close(executable)
        if child is not None and child.returncode is None:
            child.kill()
            until = end_at if end_at is not None else now() + policy["reap_ms"] * 10**6
            while child.poll() is None and now() < until:
                time.sleep(0.001)
            result["exit_code"] = child.returncode
            # Error/forced tail does not prove complete stream handling.
        for name in outputs:
            result[name + "_hex"] = bytes(outputs[name]).hex()
            if child is not None:
                stream = getattr(child, name)
                if not stream.closed:
                    stream.close()
        selector.close()
    guardian_usage = resource.getrusage(resource.RUSAGE_SELF)
    all_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["guardian_cpu_used_ns"] = int((guardian_usage.ru_utime + guardian_usage.ru_stime) * 10**9)
    result["guardian_peak_rss_bytes"] = guardian_usage.ru_maxrss * 1024
    result["all_children_cpu_used_ns"] = int((all_children.ru_utime + all_children.ru_stime) * 10**9)
    result["accounting_scope"] = "guardian_snapshot_plus_all_reaped_children_including_probe; controller_separate; serialization_and_delivery_tail_unmeasured"
    result["cpu_used_ns"] = None if child is None or child.usage is None else int(
        (child.usage.ru_utime + child.usage.ru_stime) * 10**9)
    result["peak_rss_bytes"] = None if child is None or child.usage is None else child.usage.ru_maxrss * 1024
    if result["status"] == "completed" and result["cpu_used_ns"] >= policy["cpu_seconds"] * 10**9:
        result["status"] = "cpu_exceeded"  # Never hide a measured kernel-limit overshoot.
    returned = now()
    events["receipt_created_ns"] = returned
    cleanup = events.get("cleanup_confirmed_ns")
    result["cleanup_overshoot_ns"] = None if cleanup is None else max(0, cleanup - deadline)
    result["cancellation_latency_ns"] = None if cleanup is None or stop_at is None else cleanup - stop_at
    result["receipt_overshoot_ns"] = max(0, returned - deadline)
    return result


def main():
    try:
        if len(sys.argv) != 2 or len(sys.argv[1].encode()) > 16384:
            raise ValueError("bounded policy required")
        result = run(json.loads(sys.argv[1]))
    except Exception as error:
        result = {"status": "unsupported", "cleanup_confirmed": False,
                  "error": type(error).__name__, "requirement": str(error)[:256]}
    # This is a bounded observation, NEVER publication authorization or evidence
    # of physical fsync completion. A lost/truncated receipt is incomplete.
    print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
