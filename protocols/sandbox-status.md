# T026: qualified local CPU calculation sandbox

## Scope and authority

`nf_experiments::sandbox::QualifiedDocker` is a private-field, non-deserializable Rust capability. `qualify(false, root)` returns `PERMISSION_DENIED: UNTRUSTED_GYM_EXECUTION_DISABLED` before inspecting the root or launching a process. Explicit controller opt-in performs actual Docker and guest preflight; merely finding a binary, socket, image, descriptor or mock cannot mint the capability. No Python code is imported or evaluated in the coordinator.

This is an isolated calculation backend, not gym approval, a coordinator lease, resource reservation, released model adapter, public API permission or accepted checkpoint. Generic study preflight remains fail-closed. The later intake/validation/approval/binding tasks must enforce frozen contract and gym identities before study dispatch. Generated gyms remain disabled in product capability metadata. No callback or live Torch/model/coordinator handle enters this container. Returned stdout/stderr are untrusted proposed bytes; `outputs_eligible=false` and `promotion_status=not_evaluated` even after normal completion.

## Fixed supported profile

The only backend is the existing local Linux Docker daemon at `unix:///var/run/docker.sock`, explicitly selected independently of inherited context/DOCKER_HOST. The Docker executable is the operator-trusted system `/usr/bin/docker`. No remote daemon, GPU, extra device, arbitrary image/runtime command, broad host mount, network destination, dependency install or privileged mode is accepted. An absent/unqualified backend returns a typed error; there is no plain-subprocess or in-process fallback.

The immutable CPU runtime is official `python@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c` (Python3.12.12 slim-bookworm). Deployment/setup may explicitly fetch this ordinary runtime image; execution always uses `--pull=never`. The image is not bundled into the wheel and contains no model weights. Python/Debian notices remain in the upstream image; Docker and that runtime are external system dependencies. A changed image/guardian/profile requires a new passing capability proof.

The actual configured container and guest must prove:

- Private PID, mount, IPC, cgroup and network domains; read-only image, default seccomp and no-new-privileges. No privileged mode or host PID/IPC/network namespace.
- Dropped capabilities, with only KILL/SETUID/SETGID for the fixed trusted guardian in this private container. The guardian launches a distinct Python interpreter as UID/GID65534, no supplementary groups, permitted/effective capabilities zero. No-new-privileges blocks reacquiring the bounding-set capabilities. The worker cannot signal or become the root guardian.
- Only controller-created `/job` (source/input) and `/guard` (fixed guardian/policy) snapshots mounted read-only. The controller's private journal, host credential paths, model store, Docker socket and hidden datasets are never mounted. Neither file names nor mounts are selected by the worker.
- Writable `/tmp` and `/out` are private noexec/nosuid/nodev tmpfs, 8MiB each. `/out` owner65534/group0/mode0770 allows its worker owner and trusted guardian group only. Shared memory is limited to64KiB; POSIX message queues have a zero-byte hard allowance. The image root is read-only. There is no arbitrary file-output publication or host output bind.
- Kernel cgroupv2 memory+swap allowance128MiB, process limit16 and aggregate one-CPU rate quota. These include the guardian, descendants and tmpfs pages, not only leader RSS. Kernel CPU hard limit6s/process is additional protection, not falsely reported as a whole-job CPU measurement.
- Hard RLIMIT file size64KiB, descriptors64 and core size0. Source/input are each bounded64KiB; aggregate captured stdout+stderr1–65536bytes. Runtime wall/CPU requests are100–5000ms. Unsupported limits fail before worker launch.

The trusted guardian reads the private cgroup's cumulative CPU usage and uses kernel BOOTTIME deadlines, captures both streams with a combined byte bound and terminates the worker group on a breach. It is a different UID and inaccessible to worker signal/ptrace control. Escaped sessions remain in the container's PID namespace: when docker-init/PID1 exits, the kernel tears down all members. Rust separately bounds startup/control waits, can cancel attached work and removes the exact owned container. Overshoot from sampling, scheduling and bounded cleanup is measured, not silently reported as the requested ceiling. Startup/daemon/controller work is outside guest measurements and must be conservatively charged by the study dispatcher; the low-level backend is not an accounting grant.

## Integrity, cleanup and crash recovery

The caller provides an existing absolute, physical, owner-held0700 controller root. Symlink components and unsafe Docker mount delimiters fail. Random128-bit job names and created-only snapshots prevent worker-chosen targets. Rust hashes actual source/input bytes, writes/fsyncs a private intent before creation and retains observed/configured/exit/cleanup/completion records. Actual Docker configuration, immutable image and exact read-only mount sources/destinations are checked before code starts.

Cleanup requires the exact persisted unpredictable name, image and ownership label; no unrelated process/container is selected. Absence is established by a successful local-daemon query, never by assuming a failed inspect means empty. An uncertain/error cleanup leaves an unfinished intent or `cleanup_confirmed=false`, never an eligible output. `recover_pending()` reconciles bounded private intent records after controller death, removes only matching owned containers and writes incomplete/non-promotable recovery evidence. A failed cleanup remains retryable rather than being marked recovered. Receipts measure the guest cgroup CPU/peak memory and guardian wall interval explicitly; missing reports remain missing, not zero.

## Actual checks and limitations

`cargo build -p nf-experiments --example fixture_sandbox` builds the genuine Rust controller test interface. `python -m pytest -q -k sandbox_isolation` runs real containers; missing Docker/image/cgroup/security prerequisites fail rather than skip. Tests cover default-off denial, forged/symlink roots, actual UID/caps/syscalls, guardian signal denial, absent host secrets/devices/socket, network/mount/symlink/archive escapes, output/CPU/wall/memory/process/file/storage limits, escaped-session tree cleanup, cancellation/partial output denial and real controller crash/recovery.

These are software isolation checks for this exact Linux OCI profile. They do not prove kernel/runtime invulnerability, anonymization, model quality, independence or scientific acceptance. The OS, Docker daemon/client, approved image and fixed guardian are trusted; an adversarial host root, Docker operator or same-UID controller-file attacker is outside this boundary. Hidden data must still be omitted from each proposal scope, not merely filtered after reading it.

## Resolved setup over-block

The earlier automatic recovery identified `chmod.*/usr/.*`: a combined scratch permission step and later Docker interpreter path matched a protected-system-permission rule. No `/usr` permission change was intended or performed. The permission-changing step was removed entirely; snapshot files were created through file tools and actual Docker checks then ran under the unchanged security policy. No blocked action was respelled, no policy/trust-root/sysctl changed, and no new owner permission was needed for Docker itself. Unprivileged namespace creation remains unavailable on this host; the qualified local OCI backend supplies the enforced namespaces instead.
