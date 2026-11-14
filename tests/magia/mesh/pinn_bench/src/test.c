// PINN layer micro-benchmarks for MAGIA.
// SPDX-License-Identifier: Apache-2.0
//
// Times, on ONE tile (hartid 0; the other tiles return immediately):
//
//   A. RedMulE GEMM  Y = X * W + Y  in FP16 at the three shapes a PINN
//      training step needs for a hidden layer of width WIDTH and a batch
//      slice of B points held by the tile:
//        forward / derivative  Z      = X    * W      [B x W] * [W x W]
//        backward, data        dX     = dZ   * W^T    [B x W] * [W x W]  (same shape)
//        backward, weights     dW     = X^T  * dZ     [W x B] * [B x W]
//   B. A plain-core transpose of the matrices the backward pass needs
//      transposed (W and X), because RedMulE takes operands as stored.
//
// Timing: on GVSoC read the "cycles" column (mcycle CSR). On RTL / Verilator
// the SDK says that CSR is not usable, so each measurement is also wrapped in
// sentinel_start()/sentinel_end(), which make the testbench print the cycle
// count; they appear in the log in the same order as the lines printed here.
//
// Matrix contents are arbitrary small positive FP16 numbers: the FP datapath
// timing does not depend on the values.

#include <stdint.h>

#include "tile.h"
#include "redmule.h"
#include "eventunit.h"
#include "utils/performance_utils.h"

#define WAIT_MODE WFE

#include "test.h"

// Fill n FP16 elements with values in [0.25, 0.37).
static void fill_fp16(uint32_t addr, uint32_t n, uint32_t seed)
{
    volatile uint16_t *p = (volatile uint16_t *)addr;
    for (uint32_t i = 0; i < n; i++) p[i] = (uint16_t)(0x3400u | ((i * 7u + seed) & 0x1FFu));
}

static uint32_t time_gemm(uint32_t x, uint32_t w, uint32_t y, uint16_t m, uint16_t n, uint16_t k)
{
    sentinel_start();
    uint32_t c0 = perf_get_cycles();
    redmule_gemm(x, w, y, m, n, k);
    eu_redmule_wait(WAIT_MODE);
    uint32_t c1 = perf_get_cycles();
    sentinel_end();
    return c1 - c0;
}

// dst[cols x rows] = transpose(src[rows x cols]), 16-bit elements, on the core.
static uint32_t time_transpose(uint32_t src, uint32_t dst, uint32_t rows, uint32_t cols)
{
    const uint16_t *s = (const uint16_t *)src;
    uint16_t       *d = (uint16_t *)dst;
    sentinel_start();
    uint32_t c0 = perf_get_cycles();
    for (uint32_t i = 0; i < rows; i++)
        for (uint32_t j = 0; j < cols; j++) d[j * rows + i] = s[i * cols + j];
    uint32_t c1 = perf_get_cycles();
    sentinel_end();
    return c1 - c0;
}

static void report_gemm(const char *what, uint32_t m, uint32_t n, uint32_t k, uint32_t cyc)
{
    uint32_t macs = m * n * k;
    // MACs per cycle x100, guarded for RTL where the CSR may read 0.
    uint32_t mpc = cyc ? (macs * 100u) / cyc : 0;
    printf("[GEMM] %s M=%d N=%d K=%d macs=%d cycles=%d macs_per_cycle_x100=%d\n", what, m, n, k,
           macs, cyc, mpc);
}

static void report_tr(const char *what, uint32_t rows, uint32_t cols, uint32_t cyc)
{
    uint32_t elems = rows * cols;
    uint32_t cpe   = (cyc * 100u) / elems;
    printf("[TRANSPOSE] %s rows=%d cols=%d elems=%d cycles=%d cycles_per_elem_x100=%d\n", what,
           rows, cols, elems, cyc, cpe);
}

int main(void)
{
    uint32_t hartid = get_hartid();
    if (hartid != 0) return 0;

    redmule_init();
    eu_init();
    eu_redmule_init(0);
    perf_init();

    uint32_t l1 = get_l1_base(hartid);
    uint32_t x  = l1 + OFF_X;
    uint32_t w  = l1 + OFF_W;
    uint32_t y  = l1 + OFF_Y;
    uint32_t t  = l1 + OFF_T;

    fill_fp16(x, MAX_B * WIDTH, 1);
    fill_fp16(w, WIDTH * WIDTH, 2);
    fill_fp16(y, MAX_B * WIDTH, 3);

    printf("PINN bench: WIDTH=%d, FP16, tile %d\n", WIDTH, hartid);

    static const uint16_t batch[] = {16, 64, 256};
    for (uint32_t i = 0; i < sizeof(batch) / sizeof(batch[0]); i++) {
        uint16_t b = batch[i];

        // Forward / derivative, and backward-data: [B x W] * [W x W].
        report_gemm("fwd  Z=X*W   ", b, WIDTH, WIDTH, time_gemm(x, w, y, b, WIDTH, WIDTH));

        // Backward-weights: [W x B] * [B x W]. X is reused as the [W x B]
        // operand and Y as the [B x W] one; the result is accumulated into the
        // scratch region T, so W stays intact for the next iteration.
        report_gemm("bwd  dW=Xt*dZ", WIDTH, b, WIDTH, time_gemm(x, y, t, WIDTH, b, WIDTH));

        // What the backward pass must transpose first: activations X [B x W].
        report_tr("X  (acts)   ", b, WIDTH, time_transpose(x, t, b, WIDTH));
    }

    // Weights W [W x W], needed transposed once per layer per step.
    report_tr("W  (weights)", WIDTH, WIDTH, time_transpose(w, t, WIDTH, WIDTH));

    printf("PINN bench done\n");
    return 0;
}
