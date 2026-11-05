/* Dedicated adversarial fixture: execute ONLY behind the qualified FD sandbox. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

static void denied(long result, const char *name) {
    if (result != -1 || errno != EPERM) {
        dprintf(2, "NOT_DENIED:%s:%ld:%d\n", name, result, errno);
        _exit(90);
    }
}
static void *thread(void *unused) {
    (void)unused;
    write(1, "thread\n", 7);
    return NULL;
}
static void *linger(void *unused) {
    (void)unused;
    for (;;) pause();
    return NULL;
}
int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "positive";
    if (!strcmp(mode, "positive")) {
        pthread_t t;
        if (pthread_create(&t, NULL, thread, NULL) || pthread_join(t, NULL)) return 91;
        char input[4096];
        ssize_t size = read(0, input, sizeof(input));
        if (size > 0) write(1, input, size);
        return 0;
    }
    if (!strcmp(mode, "attack")) {
        pid_t guardian = getppid();
        pid_t controller = argc > 2 ? atoi(argv[2]) : guardian;
        for (int i = 0; i < 2; i++) {
            pid_t target = i ? controller : guardian;
            denied(kill(target, SIGSTOP), "SIGSTOP");
            denied(kill(target, SIGKILL), "SIGKILL");
            denied(syscall(SYS_tgkill, target, target, SIGSTOP), "tgkill");
            denied(ptrace(PTRACE_ATTACH, target, NULL, NULL), "ptrace");
            char byte = 0;
            struct iovec local = {&byte, 1}, remote = {(void *)1, 1};
            denied(process_vm_writev(target, &local, 1, &remote, 1, 0), "process_vm_writev");
            denied(process_vm_readv(target, &local, 1, &remote, 1, 0), "process_vm_readv");
            for (int fd = 0; fd < 16; fd++) {
                char path[128];
                snprintf(path, sizeof(path), "/proc/%d/fd/%d", target, fd);
                denied(open(path, O_WRONLY), "proc-control-fd");
            }
            char memory[128];
            snprintf(memory, sizeof(memory), "/proc/%d/mem", target);
            denied(open(memory, O_RDWR), "proc-memory");
        }
        denied(syscall(SYS_pidfd_open, guardian, 0), "pidfd_open");
        denied(syscall(SYS_pidfd_getfd, 0, 0, 0), "pidfd_getfd");
        denied(syscall(SYS_pidfd_send_signal, 0, SIGSTOP, NULL, 0), "pidfd_send_signal");
        denied(syscall(SYS_io_uring_setup, 1, NULL), "io_uring");
        denied(open("/etc/passwd", O_RDONLY), "host-filesystem");
        denied(fork(), "fork");
        denied(setsid(), "setsid");
        struct rlimit limit = {1UL << 30, 1UL << 30};
        denied(setrlimit(RLIMIT_AS, &limit), "setrlimit");
        /* Probe high inherited descriptors; no control handles may survive. */
        struct stat info;
        if (fstat(200, &info) != -1 || errno != EBADF) return 92;
        write(1, "denied\n", 7);
        return 0;
    }
    if (!strcmp(mode, "sleep")) {
        signal(SIGTERM, SIG_IGN);
        for (;;) pause();
    }
    if (!strcmp(mode, "leader_exit")) {
        signal(SIGTERM, SIG_IGN);
        pthread_t t;
        if (pthread_create(&t, NULL, linger, NULL)) return 91;
        syscall(SYS_exit, 0); /* thread group must remain supervised */
    }
    if (!strcmp(mode, "flood")) {
        char data[4096]; memset(data, 'x', sizeof(data));
        for (;;) write(1, data, sizeof(data));
    }
    if (!strcmp(mode, "cpu")) {
        volatile unsigned long value = 0;
        for (;;) value++;
    }
    if (!strcmp(mode, "memory")) {
        void *value = mmap(NULL, 256UL << 20, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return value == MAP_FAILED && errno == ENOMEM ? 42 : 90;
    }
    return 93;
}
