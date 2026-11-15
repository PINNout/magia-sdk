// PINN per-element micro-benchmarks on Spatz: CV32 side.
// SPDX-License-Identifier: Apache-2.0
//
// Runs on tile 0 only. For each kernel it times "dispatch + run + wait" from
// the CV32 at two sizes (2048 and 32768 elements = batch slices of 16 and 256
// at width 128). The difference between the two sizes divided by the
// difference in elements is the per-element cost without dispatch overhead.
//
// Order: kernels that use only vector instructions already exercised by other
// tests run first; the strided-load transpose and the tanh with a vector
// divide run last, so if the simulator lacks one of those instructions the
// earlier numbers are still printed.
//
// Cycle counts come from the mcycle CSR (valid on GVSoC). sentinel_start/end
// are also emitted for RTL testbenches.

#include <stdint.h>

#include "tile.h"
#include "eventunit.h"
#include "utils/performance_utils.h"
#include "pinn_spatz_params.h"
#include "pinn_spatz_bench_task_bin.h"

#define L1          (get_l1_base(get_hartid()))
#define PARAMS_ADDR (L1 + 0x000)
#define N_MAX       (MAX_B * WIDTH)        /* 32768 elements = 64 KiB */
#define X_ADDR      (L1 + 0x1000)
#define U_ADDR      (X_ADDR + N_MAX * 2)
#define V_ADDR      (U_ADDR + N_MAX * 2)
#define D_ADDR      (V_ADDR + N_MAX * 2)

static void fill_fp16(uint32_t addr, uint32_t n, uint32_t seed)
{
    volatile uint16_t *p = (volatile uint16_t *)addr;
    /* values in [0.25, 0.37) */
    for (uint32_t i = 0; i < n; i++) p[i] = (uint16_t)(0x3400u | ((i * 7u + seed) & 0x1FFu));
}

static uint32_t run(uint32_t task, uint32_t len, uint32_t rows, uint32_t cols)
{
    volatile pinn_spatz_params_t *p = (volatile pinn_spatz_params_t *)PARAMS_ADDR;
    p->a                            = X_ADDR;
    p->b                            = U_ADDR;
    p->c                            = V_ADDR;
    p->dst                          = D_ADDR;
    p->len                          = len;
    p->rows                         = rows;
    p->cols                         = cols;

    sentinel_start();
    uint32_t c0 = perf_get_cycles();
    spatz_run_task_with_params(task, PARAMS_ADDR);
    eu_spatz_wait(WFE);
    uint32_t c1 = perf_get_cycles();
    sentinel_end();

    if (spatz_get_exit_code() != 0) printf("  !! Spatz exit code 0x%x\n", spatz_get_exit_code());
    return c1 - c0;
}

static void bench(const char *name, uint32_t task)
{
    const uint32_t n_small = 16 * WIDTH, n_big = MAX_B * WIDTH;
    printf("[SPATZ] %s: starting\n", name);
    uint32_t c_small = run(task, n_small, 16, WIDTH);
    uint32_t c_big   = run(task, n_big, MAX_B, WIDTH);
    uint32_t slope   = ((c_big - c_small) * 100u) / (n_big - n_small);
    printf("[SPATZ] %s elems=%d cycles=%d | elems=%d cycles=%d | cycles_per_elem_x100=%d "
           "out[0]=0x%04x\n",
           name, n_small, c_small, n_big, c_big, slope, *(volatile uint16_t *)D_ADDR);
}

int main(void)
{
    if (get_hartid() != 0) return 0;

    eu_init();
    eu_spatz_init(0);
    perf_init();

    fill_fp16(X_ADDR, N_MAX, 1);
    fill_fp16(U_ADDR, N_MAX, 2);
    fill_fp16(V_ADDR, N_MAX, 3);

    printf("PINN Spatz bench: WIDTH=%d FP16\n", WIDTH);
    spatz_init(SPATZ_BINARY_START);

    /* Expected out[0] for a correct result with this input data:
     *   gate 0x3403, tanh_poly 0x33a4, tanh_div 0x33d8 (true tanh: 0x33d8),
     *   transpose 0x3401 (= x[0]). */
    bench("gate x*u+(1-x)*v ", PINN_GATE_SPATZ_TASK);
    bench("sum of squares   ", PINN_SUMSQ_SPATZ_TASK);
    bench("tanh polynomial  ", PINN_TANH_POLY_SPATZ_TASK);
    bench("transpose strided", PINN_TRANSPOSE_SPATZ_TASK);
    bench("tanh rational/div", PINN_TANH_DIV_SPATZ_TASK);

    spatz_clk_dis();
    printf("PINN Spatz bench done\n");
    return 0;
}
