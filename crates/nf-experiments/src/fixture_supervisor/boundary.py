"""Linux static-ELF FD sandbox. No host paths, network, devices or processes.

The kernel/admin and the controller/guardian runtime are trusted. This is not a
sandbox for generated Python, dynamic ELF, accelerators, or ledger-owning work.
Unknown syscalls return EPERM. Unsupported primitives fail before preparation.
"""
import ctypes
import errno
import fcntl
import hashlib
import os
import platform
import resource
import signal
import stat
import struct
import sys


def credentials(pid="self"):
    with open(f"/proc/{pid}/status", encoding="ascii") as stream:
        fields = dict(line.rstrip().split(":", 1) for line in stream if ":" in line)
    uids = list(map(int, fields["Uid"].split()))
    gids = list(map(int, fields["Gid"].split()))
    if len(set(uids)) != 1 or uids[0] == 0 or len(set(gids)) != 1:
        raise ValueError("requires uniform non-root real/effective/saved/fs credentials")
    for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if int(fields[key], 16):
            raise ValueError("requires empty inherited/permitted/effective/bounding/ambient capabilities")
    return {"uid": uids[0], "gid": gids[0],
            "groups": sorted(map(int, fields["Groups"].split()))}


def protected_path(path):
    """Trusted admin-owned deployment, not just an executable hash assertion.

    Resolve symlinks, then check every ancestor. Admin replacement is outside the
    threat model; writable-by-controller runtime or library deployment is refused.
    """
    path = os.path.realpath(path)
    while True:
        info = os.stat(path)
        if info.st_uid == os.getuid() or info.st_mode & 0o022:
            raise ValueError("runtime path is not protected from controller credentials")
        if path == "/":
            return
        path = os.path.dirname(path)


def qualify_runtime():
    own = credentials()
    if credentials(os.getppid()) != own:
        raise ValueError("controller and guardian credential qualification differs")
    if os.getresuid() != (own["uid"],) * 3 or os.getresgid() != (own["gid"],) * 3:
        raise ValueError("unsafe saved credentials")
    protected_path(sys.executable)
    # Python was invoked with -I -S; no user site/environment modules. Pin the
    # trust of loaded stdlib/extension paths and mapped interpreter dependencies.
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and not path.startswith("<"):
            protected_path(path)
    with open("/proc/self/maps", encoding="ascii") as stream:
        for line in stream:
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].startswith("/"):
                protected_path(parts[5].strip())
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(3, 0, 0, 0, 0) != 0:
        raise ValueError("PR_SET_DUMPABLE protection unavailable")
    return own


def freeze_executable(policy):
    """Hash the actual sealed bytes, not a mutable source inode or path."""
    source = os.open(policy["argv"][0], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    sealed = None
    try:
        info = os.fstat(source)
        if not stat.S_ISREG(info.st_mode) or not 64 <= info.st_size <= 256 << 20:
            raise ValueError("requires bounded native ELF file")
        sealed = os.memfd_create("nf-fixture", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        remaining = info.st_size
        while remaining:
            data = os.read(source, min(65536, remaining))
            if not data:
                raise ValueError("executable changed during copy")
            view = memoryview(data)
            while view:
                count = os.write(sealed, view)
                view = view[count:]
            remaining -= len(data)
        fcntl.fcntl(sealed, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        seals = fcntl.fcntl(sealed, fcntl.F_GET_SEALS)
        if seals & 15 != 15:
            raise ValueError("immutable memfd seals unavailable")
        with os.fdopen(os.dup(sealed), "rb") as stream:
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            header = stream.read(64)
            machine = {"x86_64": 62, "aarch64": 183}.get(platform.machine())
            if (header[:7] != b"\x7fELF\x02\x01\x01" or
                    struct.unpack_from("<HH", header, 16) != (2, machine)):
                raise ValueError("requires native little-endian ELF64 ET_EXEC")
            offset = struct.unpack_from("<Q", header, 32)[0]
            size, count = struct.unpack_from("<HH", header, 54)
            if size != 56 or not 1 <= count <= 128 or offset + size * count > info.st_size:
                raise ValueError("invalid bounded ELF program headers")
            stream.seek(offset)
            for _ in range(count):
                kind = struct.unpack_from("<I", stream.read(size))[0]
                if kind in (2, 3):  # PT_DYNAMIC / PT_INTERP
                    raise ValueError("dynamic loader/dependencies are not immutable; static ELF required")
        if digest != policy["executable_sha256"]:
            raise ValueError("sealed executable differs from approved digest")
        os.lseek(sealed, 0, os.SEEK_SET)
        result, sealed = sealed, None
        return result
    finally:
        os.close(source)
        if sealed is not None:
            os.close(sealed)


def install_filter(executable, pid):
    """Allow only local computation, anonymous memory, threads and inherited FDs.

    No open*, ptrace, process_vm_*, kill, pidfds, io_uring, socket, mount, bpf,
    perf, namespace, credential or limit-changing syscall can bypass this list.
    execveat is restricted to the one sealed descriptor and AT_EMPTY_PATH.
    """
    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                    ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]
    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]
    if platform.machine() == "x86_64":
        arch, clone, mmap, execveat, tgkill = 0xC000003E, 56, 9, 322, 234
        allowed = [0, 1, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 17, 19, 20,
                   24, 25, 28, 34, 35, 39, 60, 63, 96, 97, 98, 102, 104, 107, 108, 110,
                   131, 158, 186, 202, 204, 218, 228, 230, 231, 262, 273, 302,
                   318, 334]
        # newfstatat/read-only prlimit are intentionally NOT allowed: path lookup
        # and cross-process metadata are not needed by this static fixture ABI.
        allowed.remove(262)
        allowed.remove(302)
    elif platform.machine() == "aarch64":
        arch, clone, mmap, execveat, tgkill = 0xC00000B7, 220, 222, 281, 131
        allowed = [57, 62, 63, 64, 65, 66, 67, 73, 80, 93, 94, 96, 98, 99, 101,
                   113, 115, 124, 132, 134, 135, 139, 160, 163, 165, 169, 172, 173,
                   174, 175, 176, 177, 178, 179, 214, 215, 216, 226, 233,
                   278, 293]
    else:
        raise ValueError("unsupported seccomp architecture")
    deny, allow = 0x00050000 | errno.EPERM, 0x7FFF0000
    ins = [(0x20, 0, 0, 4), (0x15, 1, 0, arch), (0x06, 0, 0, 0x80000000)]

    def match(number, checks):
        # Reload syscall number; successful argument loads never flow into the
        # next syscall comparison. Check both halves of each 64-bit argument.
        body = []
        for argument, value, mask in checks:
            for half in (0, 1):
                body += [(0x20, 0, 0, 16 + argument * 8 + half * 4),
                         (0x54, 0, 0, (mask >> (half * 32)) & 0xFFFFFFFF),
                         (0x15, 1, 0, (value >> (half * 32)) & 0xFFFFFFFF),
                         (0x06, 0, 0, deny)]
        body += [(0x06, 0, 0, allow)]
        ins.extend([(0x20, 0, 0, 0), (0x15, 0, len(body), number)] + body)

    full = (1 << 64) - 1
    match(execveat, [(0, executable, full), (4, 0x1000, full)])
    # Anonymous mmap only. No file/device backing or externally shared memory.
    match(mmap, [(3, 0x20, 0x20)])
    # Thread-only clone, with all unreviewed flags denied (including namespace,
    # ptrace, pidfd, CLONE_PARENT, and low-byte exit signals).
    required = 0x10000 | 0x100 | 0x800 | 0x400
    permitted = required | 0x200 | 0x40000 | 0x80000 | 0x100000 | 0x200000 | 0x1000000
    match(clone, [(0, required, required), (0, 0, full ^ permitted)])
    match(tgkill, [(0, pid, full)])  # Kernel verifies tid belongs to this tgid.
    # glibc falls back to inspectable clone when clone3 is unavailable.
    ins += [(0x20, 0, 0, 0), (0x15, 0, 1, 435),
            (0x06, 0, 0, 0x00050000 | errno.ENOSYS)]
    for number in allowed:
        ins += [(0x20, 0, 0, 0), (0x15, 0, 1, number), (0x06, 0, 0, allow)]
    ins += [(0x06, 0, 0, deny)]
    filters = (Filter * len(ins))(*(Filter(*item) for item in ins))
    program = Program(len(filters), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise ValueError("kernel no_new_privs/seccomp filter unavailable")


def child_boundary(parent, policy, executable):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) or os.getppid() != parent:
        os._exit(125)
    for kind, limit in ((resource.RLIMIT_AS, policy["memory_bytes"]),
                        (resource.RLIMIT_CPU, policy["cpu_seconds"]),
                        (resource.RLIMIT_FSIZE, policy["output_bytes"]),
                        (resource.RLIMIT_CORE, 0), (resource.RLIMIT_NOFILE, 64)):
        resource.setrlimit(kind, (limit, limit))
    os.setpgid(0, 0)
    # Only pipes 0/1/2 and the read-only, sealed ELF descriptor survive. close_range
    # is mandatory, including FDs above RLIMIT_NOFILE. Install policy LAST.
    for low, high in ((3, executable - 1), (executable + 1, 0xFFFFFFFF)):
        if low <= high and libc.syscall(436, ctypes.c_uint(low), ctypes.c_uint(high), 0):
            raise ValueError("kernel close_range unavailable")
    install_filter(executable, os.getpid())
