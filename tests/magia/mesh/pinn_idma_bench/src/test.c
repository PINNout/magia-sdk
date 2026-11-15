// PINN iDMA transfer benchmark for MAGIA.
// SPDX-License-Identifier: Apache-2.0
//
// Runs on tile 0 only (the other tiles return immediately; their L1 is used
// as a passive source / destination). Times iDMA copies of 4, 32 and 128 KiB:
//
//   L2 -> my L1            weights arriving from shared memory
//   my L1 -> L2            gradients leaving to shared memory
//   my L1 -> neighbour L1  push to the tile next to me
//   neighbour L1 -> my L1  pull from the tile next to me
//   my L1 -> far L1        push to the opposite corner of the mesh
//   far L1 -> my L1        pull from the opposite corner
//
// 32 KiB is one 128x128 FP16 weight matrix; all weights of the 4x128
// ModifiedMlp are about 288 KiB.
//
// Cycle counts come from the mcycle CSR (valid on GVSoC); sentinel_start/end
// are also emitted for RTL testbenches.

#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "eventunit.h"
#include "utils/performance_utils.h"

#define WAIT_MODE WFE

#include "test.h"

// Shared-memory (L2) buffer: lives in this program's .bss.
static volatile uint8_t l2_buf[MAX_BYTES];

// dir 0: AXI address -> my L1 (OBI).   dir 1: my L1 (OBI) -> AXI address.
static uint32_t time_copy(uint32_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len)
{
    sentinel_start();
    uint32_t c0 = perf_get_cycles();
    idma_memcpy_1d(dir, axi_addr, obi_addr, len);
    if (dir == 0)
        eu_idma_wait_a2o(WAIT_MODE);
    else
        eu_idma_wait_o2a(WAIT_MODE);
    uint32_t c1 = perf_get_cycles();
    sentinel_end();
    return c1 - c0;
}

static void report(const char *what, uint32_t len, uint32_t cyc)
{
    uint32_t bpc = cyc ? (len * 100u) / cyc : 0; /* bytes per cycle x100 */
    printf("[IDMA] %s bytes=%d cycles=%d bytes_per_cycle_x100=%d\n", what, len, cyc, bpc);
}

int main(void)
{
    uint32_t hartid = get_hartid();
    if (hartid != 0) return 0;

    idma_init();
    eu_init();
    eu_idma_init(0);
    perf_init();

    uint32_t my_l1  = get_l1_base(hartid) + L1_OFF;
    uint32_t near_l1 = get_l1_base(GET_ID(0, 1)) + L1_OFF;                               /* x+1 */
    uint32_t far_l1  = get_l1_base(GET_ID(MESH_Y_TILES - 1, MESH_X_TILES - 1)) + L1_OFF; /* corner */
    uint32_t l2      = (uint32_t)l2_buf;

    /* Give the source something to carry; timing does not depend on it. */
    volatile uint32_t *p = (volatile uint32_t *)my_l1;
    for (uint32_t i = 0; i < MAX_BYTES / 4; i += 64) p[i] = 0x3C003C00u + i;

    printf("PINN iDMA bench: mesh %dx%d, tile %d, L2 buffer at 0x%x\n", MESH_X_TILES, MESH_Y_TILES,
           hartid, l2);

    static const uint32_t sizes[] = {4 * 1024, 32 * 1024, 128 * 1024};
    for (uint32_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        uint32_t n = sizes[i];
        report("L2   -> my L1  ", n, time_copy(0, l2, my_l1, n));
        report("my L1 -> L2    ", n, time_copy(1, l2, my_l1, n));
        report("my L1 -> near  ", n, time_copy(1, near_l1, my_l1, n));
        report("near -> my L1  ", n, time_copy(0, near_l1, my_l1, n));
        report("my L1 -> far   ", n, time_copy(1, far_l1, my_l1, n));
        report("far  -> my L1  ", n, time_copy(0, far_l1, my_l1, n));
    }

    printf("PINN iDMA bench done\n");
    return 0;
}
