// PINN per-element kernels on Spatz, FP16, LMUL=8 (128 elements per iteration
// at VLEN=256). Written in the style of multi_task/vec_add_spatz_task.c and
// nn_is_bench/nn_act_spatz_task.c.
// SPDX-License-Identifier: Apache-2.0
//
//   pinn_gate_spatz_task       out = x*u + (1-x)*v      (ModifiedMlp gating)
//   pinn_sumsq_spatz_task      128 partial sums of x^2  (loss / grad-norm)
//   pinn_tanh_poly_spatz_task  tanh, odd degree-7 polynomial, no division
//                              (max abs error 2.8e-2: timing reference only)
//   pinn_transpose_spatz_task  dst[cols x rows] = src[rows x cols]^T using
//                              strided vector loads
//   pinn_tanh_div_spatz_task   tanh, rational x(945+105x^2+x^4)/(945+420x^2+15x^4)
//                              (max abs error 2.4e-3 in FP16), one vector divide

#include "tile.h"
#include "pinn_spatz_params.h"

static inline volatile pinn_spatz_params_t *get_params(void)
{
    return (volatile pinn_spatz_params_t *)mmio32(SPATZ_DATA);
}

int pinn_gate_spatz_task(void)
{
    register _Float16 NEG1 asm("fs0") = -1.0f;
    volatile pinn_spatz_params_t *p   = get_params();
    _Float16 *X                       = (_Float16 *)p->a;
    _Float16 *U                       = (_Float16 *)p->b;
    _Float16 *V                       = (_Float16 *)p->c;
    _Float16 *D                       = (_Float16 *)p->dst;
    size_t avl                        = p->len;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(X));
        asm volatile("vle16.v v8, (%0)" ::"r"(U));
        asm volatile("vle16.v v16, (%0)" ::"r"(V));
        /* out = v + x*(u - v) */
        asm volatile("vfmul.vf v24, v16, %0" ::"f"(NEG1));
        asm volatile("vfadd.vv v24, v24, v8");
        asm volatile("vfmul.vv v24, v24, v0");
        asm volatile("vfadd.vv v24, v24, v16");
        asm volatile("vse16.v v24, (%0)" ::"r"(D));
        X += vl;
        U += vl;
        V += vl;
        D += vl;
    }
    return 0;
}

int pinn_sumsq_spatz_task(void)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
    volatile pinn_spatz_params_t *p   = get_params();
    _Float16 *X                       = (_Float16 *)p->a;
    _Float16 *D                       = (_Float16 *)p->dst;
    size_t avl                        = p->len; /* multiple of 128 */
    size_t vl;

    /* v16 = 0 (accumulator), built only from instructions other tests use. */
    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle16.v v16, (%0)" ::"r"(X));
    asm volatile("vfmul.vf v16, v16, %0" ::"f"(ZERO));

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(X));
        asm volatile("vfmul.vv v8, v0, v0");
        asm volatile("vfadd.vv v16, v16, v8");
        X += vl;
    }
    /* 128 partial sums; folding them to one number is a fixed small cost. */
    asm volatile("vse16.v v16, (%0)" ::"r"(D));
    return 0;
}

int pinn_tanh_poly_spatz_task(void)
{
    register _Float16 LIM asm("fs0")  = 2.5f;
    register _Float16 NLIM asm("fs1") = -2.5f;
    register _Float16 C0 asm("fs2")   = 0.96875f;
    register _Float16 C1 asm("fs3")   = -0.2381591796875f;
    register _Float16 C2 asm("fs4")   = 0.03936767578125f;
    register _Float16 C3 asm("fs5")   = -0.0025768280029296875f;
    register _Float16 ONE asm("fs6")  = 1.0f;
    register _Float16 NONE asm("fs7") = -1.0f;
    volatile pinn_spatz_params_t *p   = get_params();
    _Float16 *X                       = (_Float16 *)p->a;
    _Float16 *D                       = (_Float16 *)p->dst;
    size_t avl                        = p->len;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(X));
        asm volatile("vfmin.vf v0, v0, %0" ::"f"(LIM));
        asm volatile("vfmax.vf v0, v0, %0" ::"f"(NLIM));
        asm volatile("vfmul.vv v8, v0, v0"); /* s = x^2 */
        /* Horner in s: ((C3*s + C2)*s + C1)*s + C0, then * x */
        asm volatile("vfmul.vf v16, v8, %0" ::"f"(C3));
        asm volatile("vfadd.vf v16, v16, %0" ::"f"(C2));
        asm volatile("vfmul.vv v16, v16, v8");
        asm volatile("vfadd.vf v16, v16, %0" ::"f"(C1));
        asm volatile("vfmul.vv v16, v16, v8");
        asm volatile("vfadd.vf v16, v16, %0" ::"f"(C0));
        asm volatile("vfmul.vv v16, v16, v0");
        asm volatile("vfmin.vf v16, v16, %0" ::"f"(ONE));
        asm volatile("vfmax.vf v16, v16, %0" ::"f"(NONE));
        asm volatile("vse16.v v16, (%0)" ::"r"(D));
        X += vl;
        D += vl;
    }
    return 0;
}

int pinn_transpose_spatz_task(void)
{
    volatile pinn_spatz_params_t *p = get_params();
    const _Float16 *S               = (const _Float16 *)p->a;
    _Float16 *D                     = (_Float16 *)p->dst;
    size_t rows                     = p->rows;
    size_t cols                     = p->cols;
    size_t stride                   = cols * sizeof(_Float16); /* bytes between column elements */

    /* Row j of dst is column j of src: read it with a strided load. */
    for (size_t j = 0; j < cols; j++) {
        const _Float16 *s = S + j;
        size_t avl        = rows;
        size_t vl;
        for (; avl > 0; avl -= vl) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
            asm volatile("vlse16.v v0, (%0), %1" ::"r"(s), "r"(stride));
            asm volatile("vse16.v v0, (%0)" ::"r"(D));
            s += vl * cols;
            D += vl;
        }
    }
    return 0;
}

int pinn_tanh_div_spatz_task(void)
{
    register _Float16 LIM asm("fs0")  = 4.0f;
    register _Float16 NLIM asm("fs1") = -4.0f;
    register _Float16 K105 asm("fs2") = 105.0f;
    register _Float16 K945 asm("fs3") = 945.0f;
    register _Float16 K15 asm("fs4")  = 15.0f;
    register _Float16 K420 asm("fs5") = 420.0f;
    register _Float16 ONE asm("fs6")  = 1.0f;
    register _Float16 NONE asm("fs7") = -1.0f;
    volatile pinn_spatz_params_t *p   = get_params();
    _Float16 *X                       = (_Float16 *)p->a;
    _Float16 *D                       = (_Float16 *)p->dst;
    size_t avl                        = p->len;
    size_t vl;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(X));
        asm volatile("vfmin.vf v0, v0, %0" ::"f"(LIM));
        asm volatile("vfmax.vf v0, v0, %0" ::"f"(NLIM));
        asm volatile("vfmul.vv v8, v0, v0"); /* s = x^2 */
        /* num = x * ((s + 105)*s + 945) */
        asm volatile("vfadd.vf v16, v8, %0" ::"f"(K105));
        asm volatile("vfmul.vv v16, v16, v8");
        asm volatile("vfadd.vf v16, v16, %0" ::"f"(K945));
        asm volatile("vfmul.vv v16, v16, v0");
        /* den = (15*s + 420)*s + 945 */
        asm volatile("vfmul.vf v24, v8, %0" ::"f"(K15));
        asm volatile("vfadd.vf v24, v24, %0" ::"f"(K420));
        asm volatile("vfmul.vv v24, v24, v8");
        asm volatile("vfadd.vf v24, v24, %0" ::"f"(K945));
        asm volatile("vfdiv.vv v16, v16, v24");
        asm volatile("vfmin.vf v16, v16, %0" ::"f"(ONE));
        asm volatile("vfmax.vf v16, v16, %0" ::"f"(NONE));
        asm volatile("vse16.v v16, (%0)" ::"r"(D));
        X += vl;
        D += vl;
    }
    return 0;
}
