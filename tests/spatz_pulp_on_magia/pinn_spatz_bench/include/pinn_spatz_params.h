// PINN Spatz micro-benchmark parameters.
// SPDX-License-Identifier: Apache-2.0
#ifndef PINN_SPATZ_PARAMS_H_
#define PINN_SPATZ_PARAMS_H_

#include <stdint.h>

// One parameter block shared by all tasks; each task uses the fields it needs.
typedef struct {
    uintptr_t a;   /* fp16 input  (x)                         */
    uintptr_t b;   /* fp16 input  (u), gate only              */
    uintptr_t c;   /* fp16 input  (v), gate only              */
    uintptr_t dst; /* fp16 output                             */
    uint32_t len;  /* number of elements                      */
    uint32_t rows; /* transpose only: source is [rows x cols] */
    uint32_t cols;
} pinn_spatz_params_t;

#define WIDTH 128 /* hidden width of the PINN */
#define MAX_B 256 /* largest batch slice      */

#endif
