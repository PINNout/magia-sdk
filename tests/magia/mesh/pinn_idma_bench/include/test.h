// PINN iDMA transfer benchmark parameters.
// SPDX-License-Identifier: Apache-2.0
#ifndef PINN_IDMA_BENCH_TEST_H
#define PINN_IDMA_BENCH_TEST_H

// Largest transfer, in bytes. 128 KiB = four 128x128 FP16 weight matrices.
#define MAX_BYTES (128 * 1024)

// Offset inside a tile's L1 used as source / destination, clear of the
// first bytes that other tests use for parameters.
#define L1_OFF 0x1000

#endif
