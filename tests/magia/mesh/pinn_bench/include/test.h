// PINN layer micro-benchmark parameters.
// SPDX-License-Identifier: Apache-2.0
#ifndef PINN_BENCH_TEST_H
#define PINN_BENCH_TEST_H

// Hidden width of the PINN (config.arch.hidden_dim).
#define WIDTH 128

// Largest batch slice a tile is asked to hold in this benchmark.
#define MAX_B 256

// L1 layout (bytes from the tile's L1 base). FP16 = 2 bytes per element.
//   X  : MAX_B x WIDTH   activations            64 KiB
//   W  : WIDTH x WIDTH   weights                32 KiB
//   Y  : MAX_B x WIDTH   bias in / result out   64 KiB
//   T  : MAX_B x WIDTH   transpose destination  64 KiB
#define OFF_X 0
#define OFF_W (OFF_X + MAX_B * WIDTH * 2)
#define OFF_Y (OFF_W + WIDTH * WIDTH * 2)
#define OFF_T (OFF_Y + MAX_B * WIDTH * 2)

#endif
