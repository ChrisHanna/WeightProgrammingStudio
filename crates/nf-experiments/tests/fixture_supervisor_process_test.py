"""Bounded real-process tests. Adversarial native code runs ONLY after qualification.

No hostile Python is executed. This is component evidence, not T038 acceptance.
"""
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "src/fixture_supervisor"
PYTHON = str(Path(sys.executable).resolve())
SCRIPT = (SUPERVISOR / "boundary.py").read_text() + "\n" + (SUPERVISOR / "guardian.py").read_text()


def clock():
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def line(process):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(3):
            raise AssertionError("guardian response exceeded 3 seconds")
        return json.loads(process.stdout.readline(180000))


class SupervisorProcesses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="nf-t038-static-")
        cls.binary = Path(cls.directory.name) / "worker"
        subprocess.run(["cc", "-static", "-O2", "-pthread", "-o", str(cls.binary),
                        str(ROOT / "tests/fixture_supervisor_static_worker.c")],
                       check=True, capture_output=True, timeout=30)
        cls.digest = hashlib.sha256(cls.binary.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.send_signal(signal.SIGCONT)
                process.kill()
            process.wait(timeout=2)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream and not stream.closed:
                    stream.close()

    def policy(self, mode="positive", **changes):
        deadline = clock() + 2_000_000_000
        result = {
            "profile": "static_elf_fd_cpu_linux_v2",
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "deadline_boottime_ns": deadline,
            "memory_bytes": 96 << 20, "cpu_seconds": 1,
            "output_bytes": 4096, "grace_ms": 40, "reap_ms": 200,
            "executable_sha256": self.digest,
            "argv": [str(self.binary), mode, str(os.getpid())], "input_hex": "",
            "binding": {"contract": "contract-1", "reservation": "reservation-1",
                        "attempt": "attempt-1", "lease": "lease-1", "fencing_epoch": 7,
                        "nonce": secrets.token_hex(32), "lease_deadline_boottime_ns": deadline},
        }
        result.update(changes)
        return result

    def launch(self, request, authorize=True, pass_fds=(), script=SCRIPT):
        process = subprocess.Popen([PYTHON, "-I", "-S", "-B", "-c", script, json.dumps(request)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, pass_fds=pass_fds)
        self.processes.append(process)
        prepared = line(process)
        if prepared.get("kind") == "prepared":
            self.assertEqual(prepared["identity"]["policy"], request)
            self.assertEqual(prepared["identity"]["guardian_pid"], process.pid)
            # Simulated caller persistence, then recheck; real storage integration
            # belongs to the coordinator and is intentionally not claimed here.
            if authorize:
                self.authorize(process, prepared["identity"])
        return process, prepared

    def authorize(self, process, identity):
        process.stdin.write(json.dumps({"authorize": identity}).encode() + b"\n")
        process.stdin.flush()

    def receipt(self, process):
        report = line(process)
        process.wait(timeout=3)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(process.stderr.read(4096), b"")
        return report

    def completed(self, mode="positive", **changes):
        process, prepared = self.launch(self.policy(mode, **changes))
        self.assertEqual(prepared.get("kind"), "prepared", prepared)
        return self.receipt(process)

    def test_positive_static_thread_input_and_costs(self):
        report = self.completed(input_hex=b"trusted-input".hex())
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["cleanup_confirmed"])
        self.assertEqual(bytes.fromhex(report["stdout_hex"]), b"thread\ntrusted-input")
        self.assertGreater(report["guardian_cpu_used_ns"], 0)
        self.assertGreater(report["guardian_peak_rss_bytes"], 0)
        self.assertGreater(report["all_children_cpu_used_ns"], report["cpu_used_ns"])
        self.assertIn("controller_separate", report["accounting_scope"])
        self.assertIn("tail_unmeasured", report["accounting_scope"])

    def test_native_attacks_cannot_disable_guardian_or_reacquire_control(self):
        with open("/dev/null", "rb") as source:
            descriptor = fcntl.fcntl(source, fcntl.F_DUPFD_CLOEXEC, 200)
        try:
            process, prepared = self.launch(self.policy("attack"), pass_fds=(descriptor,))
        finally:
            os.close(descriptor)
        self.assertEqual(prepared.get("kind"), "prepared", prepared)
        report = self.receipt(process)
        self.assertEqual(report["status"], "completed", report)
        self.assertEqual(bytes.fromhex(report["stdout_hex"]), b"denied\n")

    def test_prepare_does_not_start_worker_or_reset_deadline(self):
        request = self.policy(deadline_boottime_ns=clock() + 350_000_000)
        process, prepared = self.launch(request, authorize=False)
        self.assertEqual(prepared["kind"], "prepared")
        report = self.receipt(process)
        self.assertEqual(report["status"], "timed_out")
        self.assertNotIn("worker_started_ns", report["events"])
        self.assertEqual(report["deadline_boottime_ns"], request["deadline_boottime_ns"])

    def test_every_launch_identity_field_is_bound_before_authorization(self):
        for key in ("contract", "reservation", "attempt", "lease", "fencing_epoch", "nonce", "lease_deadline_boottime_ns"):
            with self.subTest(key=key):
                process, prepared = self.launch(self.policy(), authorize=False)
                self.assertEqual(prepared["kind"], "prepared")
                identity = prepared["identity"]
                value = identity["policy"]["binding"][key]
                identity["policy"]["binding"][key] = value + 1 if isinstance(value, int) else value + "0"
                self.authorize(process, identity)
                report = self.receipt(process)
                self.assertEqual(report["status"], "invalid_control")
                self.assertNotIn("worker_started_ns", report["events"])

    def test_sealed_bytes_survive_source_mutation_after_prepare(self):
        copy = Path(self.directory.name) / "mutable-copy"
        copy.write_bytes(self.binary.read_bytes())
        request = self.policy(argv=[str(copy), "positive"])
        try:
            process, prepared = self.launch(request, authorize=False)
            self.assertEqual(prepared["kind"], "prepared")
            copy.write_bytes(b"not ELF anymore")
            self.authorize(process, prepared["identity"])
            self.assertEqual(self.receipt(process)["status"], "completed")
        finally:
            copy.unlink()

    def test_python_dynamic_elf_bad_digest_and_expired_identity_fail_closed(self):
        for change in ({"argv": [PYTHON, "-c", "raise SystemExit(90)"],
                        "executable_sha256": hashlib.sha256(Path(PYTHON).read_bytes()).hexdigest()},
                       {"executable_sha256": "0" * 64}, {"boot_id": "other"},
                       {"deadline_boottime_ns": 1}, {"profile": "hostile_gym"}):
            with self.subTest(change=change):
                _, failure = self.launch(self.policy(**change))
                self.assertEqual(failure["status"], "unsupported")
                self.assertFalse(failure["cleanup_confirmed"])
                self.assertIn("requirement", failure)

    def test_missing_kernel_primitive_fails_before_prepared(self):
        script = SCRIPT.replace("    install_filter(executable, os.getpid())",
                                "    raise ValueError('injected missing seccomp primitive')")
        _, failure = self.launch(self.policy(), script=script)
        self.assertEqual(failure["status"], "unsupported")
        self.assertIn("boundary probe failed", failure["requirement"])

    def test_unsafe_credentials_and_mutable_runtime_are_rejected(self):
        # Trusted harness fault injection; no hostile executable starts.
        for script in (SCRIPT.replace('if len(set(uids)) != 1 or uids[0] == 0',
                                      'if True or len(set(uids)) != 1 or uids[0] == 0'),
                       SCRIPT.replace('if info.st_uid == os.getuid() or info.st_mode & 0o022:',
                                      'if True:')):
            _, failure = self.launch(self.policy(), script=script)
            self.assertEqual(failure["status"], "unsupported")
            self.assertNotIn("kind", failure)

    def test_deadline_kills_uncooperative_worker_and_live_threads(self):
        for mode in ("sleep", "leader_exit"):
            with self.subTest(mode=mode):
                request = self.policy(mode, deadline_boottime_ns=clock() + 350_000_000)
                process, prepared = self.launch(request)
                self.assertEqual(prepared["kind"], "prepared")
                report = self.receipt(process)
                self.assertEqual(report["status"], "timed_out", report)
                self.assertTrue(report["cleanup_confirmed"])
                self.assertIn("kill_sent_ns", report["events"])
                self.assertGreater(report["cleanup_overshoot_ns"], 0)

    def test_cancel_control_eof_and_no_authorization_after_cancel(self):
        for action in ("cancel", "eof"):
            process, prepared = self.launch(self.policy("sleep"))
            self.assertEqual(prepared["kind"], "prepared")
            time.sleep(0.08)
            if action == "cancel":
                process.stdin.write(b"SS")
                process.stdin.flush()
            else:
                process.stdin.close()
            report = self.receipt(process)
            self.assertEqual(report["status"], "cancelled" if action == "cancel" else "control_lost")
            self.assertTrue(report["cleanup_confirmed"])

    def test_output_memory_and_cpu_limits(self):
        report = self.completed("flood", output_bytes=513)
        self.assertEqual(report["status"], "output_exceeded")
        self.assertEqual(len(bytes.fromhex(report["stdout_hex"])), 513)
        report = self.completed("memory")
        self.assertEqual(report["exit_code"], 42)
        self.assertTrue(report["cleanup_confirmed"])
        report = self.completed("cpu", deadline_boottime_ns=clock() + 4_000_000_000,
                                binding={**self.policy()["binding"], "lease_deadline_boottime_ns": clock() + 5_000_000_000})
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["exit_code"], -signal.SIGKILL)
        self.assertGreater(report["cpu_used_ns"], 800_000_000)

    def test_guardian_death_cannot_produce_a_receipt(self):
        process, prepared = self.launch(self.policy("sleep"))
        self.assertEqual(prepared["kind"], "prepared")
        time.sleep(0.08)
        process.kill()
        process.wait(timeout=2)
        self.assertEqual(process.stdout.read(), b"")
        self.assertEqual(process.returncode, -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
