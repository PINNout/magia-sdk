# PINN per-element micro-benchmarks on Spatz (FP16 vectors).
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME pinn_spatz_bench)

# Spatz tasks -> pinn_spatz_bench_task_bin.h
add_spatz_task(
    TEST_NAME       ${TEST_NAME}
    FIRST_TASK_NAME pinn_gate_spatz_task
    TASK_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/spatz_task/pinn_spatz_tasks.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# CV32 executable with the Spatz binary embedded
add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
