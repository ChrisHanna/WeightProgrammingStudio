/* Fixed FD-only f64 fixture ABI v1. No labels, scores, paths, or ledger authority.
 * Native little-endian ELF64 only (also checked by the qualified guardian).
 * All hashing/validation/allocation is outside the measured forward interval.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static int transfer(int fd, void *bytes, size_t count, int writing) {
    unsigned char *p = bytes;
    while (count) {
        ssize_t n = writing ? write(fd, p, count) : read(fd, p, count);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += n;
        count -= (size_t)n;
    }
    return 0;
}
static int clock_ns(uint64_t *value) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return -1;
    *value = (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
    return 0;
}
/* Arithmetic is a fixed fixture adapter; never accept labels or scores. */
static int arithmetic(int timed) {
    uint32_t count;
    double weights[4][42], inputs[32][2];
    unsigned char contexts[32][32], output[32][808], extra;
    if (transfer(0, &count, sizeof count, 0) || !count || count > 32
        || transfer(0, weights, sizeof weights, 0)) return 2;
    for (unsigned cell = 0; cell < count; ++cell)
        if (transfer(0, contexts[cell], 32, 0) || transfer(0, inputs[cell], 16, 0)) return 2;
    if (read(0, &extra, 1) != 0) return 2;
    for (unsigned arm = 0; arm < 4; ++arm)
        for (unsigned i = 0; i < 42; ++i) if (!isfinite(weights[arm][i])) return 3;
    for (unsigned cell = 0; cell < count; ++cell) {
        volatile double *x = inputs[cell];
        if (!isfinite(x[0]) || !isfinite(x[1])) return 3;
        memcpy(output[cell], timed ? "NFARI002" : "NFARI001", 8);
        memcpy(output[cell] + 8, contexts[cell], 32);
        unsigned offset = 40;
        for (unsigned repeat = 0; repeat < (timed ? 4u : 1u); ++repeat) {
            for (unsigned arm = 0; arm < 4; ++arm) {
                volatile double *w = weights[arm];
                volatile double hidden[6], y[5];
                uint64_t start = 0, end = 0;
                if (timed && clock_ns(&start)) return 4;
                for (unsigned row = 0; row < 6; ++row) {
                    double v = (0.0 + w[row * 2] * x[0]) + w[row * 2 + 1] * x[1];
                    hidden[row] = v > 0.0 ? v : 0.0;
                }
                for (unsigned head = 0; head < 5; ++head) {
                    double sum = 0.0;
                    for (unsigned i = 0; i < 6; ++i) sum += w[12 + head * 6 + i] * hidden[i];
                    y[head] = sum;
                }
                if (timed && (clock_ns(&end) || end < start)) return 4;
                if (timed) {
                    uint64_t elapsed = end - start;
                    memcpy(output[cell] + offset, &elapsed, 8);
                    offset += 8;
                }
                for (unsigned head = 0; head < 5; ++head) {
                    double value = y[head];
                    if (!isfinite(value)) return 4;
                    memcpy(output[cell] + offset, &value, 8);
                    offset += 8;
                }
            }
        }
        if (offset != (timed ? 808u : 200u)
            || transfer(1, output[cell], offset, 1)) return 6;
    }
    return 0;
}
int main(void) {
    unsigned char header[8], contexts[32][32], output[32][432], extra;
    uint32_t count;
    double weights[4][4], inputs[32][2];
    if (transfer(0, header, sizeof header, 0)) return 2;
    if (!memcmp(header, "NFARI001", 8)) return arithmetic(0);
    if (!memcmp(header, "NFARI002", 8)) return arithmetic(1);
    if (memcmp(header, "NFLIN001", 8)
        || transfer(0, &count, sizeof count, 0) || !count || count > 32
        || transfer(0, weights, sizeof weights, 0)) return 2;
    for (unsigned cell = 0; cell < count; ++cell)
        if (transfer(0, contexts[cell], 32, 0) || transfer(0, inputs[cell], 16, 0)) return 2;
    if (read(0, &extra, 1) != 0) return 2;
    for (unsigned a = 0; a < 4; ++a)
        for (unsigned i = 0; i < 4; ++i) if (!isfinite(weights[a][i])) return 3;
    for (unsigned cell = 0; cell < count; ++cell) {
    double *input = inputs[cell];
    for (unsigned i = 0; i < 2; ++i) if (!isfinite(input[i])) return 3;
    memcpy(output[cell], header, 8);
    memcpy(output[cell] + 8, contexts[cell], 32);
    unsigned offset = 40;
    for (unsigned repeat = 0; repeat < 4; ++repeat) {
        for (unsigned arm = 0; arm < 4; ++arm) {
            uint64_t start, end;
            double y[2];
            /* volatile loads/stores prevent hoisting across the actual clocks;
             * -ffp-contract=off preserves the registered Rust kernel ordering. */
            volatile double *w = weights[arm], *x = input, *out = y;
            if (clock_ns(&start)) return 4;
            out[0] = (0.0 + w[0] * x[0]) + w[1] * x[1];
            out[1] = (0.0 + w[2] * x[0]) + w[3] * x[1];
            if (clock_ns(&end) || end < start || !isfinite(y[0]) || !isfinite(y[1])) return 4;
            uint64_t elapsed = end - start;
            memcpy(output[cell] + offset, &elapsed, 8);
            memcpy(output[cell] + offset + 8, y, 16);
            offset += 24;
        }
    }
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) || usage.ru_maxrss < 0) return 5;
    uint64_t rss = (uint64_t)usage.ru_maxrss * 1024;
    memcpy(output[cell] + offset, &rss, 8);
    }
    return transfer(1, output, (size_t)count * 432, 1) ? 6 : 0;
}
