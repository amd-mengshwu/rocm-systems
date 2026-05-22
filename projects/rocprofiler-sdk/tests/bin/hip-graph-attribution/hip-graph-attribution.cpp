/*
Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

template <typename T>
void
check(T result, char const* const func, const char* const file, int const line)
{
    if(result)
    {
        fprintf(stderr,
                "Hip error at %s:%d code=%d(%s) \"%s\" \n",
                file,
                line,
                static_cast<unsigned int>(result),
                hipGetErrorName(result),
                func);
        exit(EXIT_FAILURE);
    }
}
#define checkHipErrors(val) check((val), #val, __FILE__, __LINE__)

__global__ void
kernel_a(float* x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    x[idx] += 1.0f;
}

__global__ void
kernel_b(float* x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    x[idx] *= 2.0f;
}

int
main(int argc, char** argv)
{
    int iterations = (argc > 1) ? std::atoi(argv[1]) : 50;

    constexpr size_t N     = 1024;
    float*           d_buf = nullptr;
    checkHipErrors(hipMalloc(&d_buf, N * sizeof(float)));

    hipGraph_t graph;
    checkHipErrors(hipGraphCreate(&graph, 0));

    // Topology (5 nodes):
    //   a0 (kernel_a, root)
    //     |-> b1 (kernel_b)
    //     `-> a2 (kernel_a)      [b1 and a2 run in parallel after a0]
    //   a3 (kernel_a, depends on both b1 and a2)
    //   b4 (kernel_b, depends on a3)
    // Three distinct nodes call kernel_a (a0, a2, a3);
    // two distinct nodes call kernel_b (b1, b4).

    auto make_kernel_node = [&](hipGraphNode_t* node, hipGraphNode_t* deps, size_t ndeps,
                                void (*fn)(float*)) {
        hipKernelNodeParams kp{};
        void*               args[] = {&d_buf};
        kp.func                    = reinterpret_cast<void*>(fn);
        kp.gridDim                 = dim3(8, 1, 1);
        kp.blockDim                = dim3(128, 1, 1);
        kp.kernelParams            = args;
        kp.extra                   = nullptr;
        kp.sharedMemBytes          = 0;
        checkHipErrors(hipGraphAddKernelNode(node, graph, deps, ndeps, &kp));
    };

    hipGraphNode_t a0, b1, a2, a3, b4;
    make_kernel_node(&a0, nullptr, 0, kernel_a);
    make_kernel_node(&b1, &a0, 1, kernel_b);
    make_kernel_node(&a2, &a0, 1, kernel_a);
    hipGraphNode_t deps_for_a3[] = {b1, a2};
    make_kernel_node(&a3, deps_for_a3, 2, kernel_a);
    make_kernel_node(&b4, &a3, 1, kernel_b);

    // Two separate executable graphs from the same source -- to verify per-exec
    // distinctness of graph_exec_id.
    hipGraphExec_t exec_a, exec_b;
    checkHipErrors(hipGraphInstantiate(&exec_a, graph, nullptr, nullptr, 0));
    checkHipErrors(hipGraphInstantiate(&exec_b, graph, nullptr, nullptr, 0));

    hipStream_t stream;
    checkHipErrors(hipStreamCreate(&stream));

    for(int i = 0; i < iterations; ++i)
    {
        checkHipErrors(hipGraphLaunch(exec_a, stream));
        checkHipErrors(hipGraphLaunch(exec_b, stream));
    }
    checkHipErrors(hipStreamSynchronize(stream));

    // Failure-mode test: destroy exec_a, then attempt launch on the destroyed
    // handle -- must fail and must NOT emit a GRAPH_LAUNCH record. The subsequent
    // launch on exec_b must still be attributed correctly (TLS state cleaned up).
    checkHipErrors(hipGraphExecDestroy(exec_a));
    hipError_t failed = hipGraphLaunch(exec_a, stream);  // expected: not hipSuccess
    (void) failed;  // not asserted; the test asserts the trace artifacts via validate.py
    checkHipErrors(hipGraphLaunch(exec_b, stream));      // one extra valid launch
    checkHipErrors(hipStreamSynchronize(stream));

    std::fprintf(stderr,
                 "[hip-graph-attribution] iterations=%d execs=2 nodes_per_launch=5 "
                 "distinct_kernels=2 valid_launches=%d failed_launches=1\n",
                 iterations,
                 iterations * 2 + 1);

    checkHipErrors(hipGraphExecDestroy(exec_b));
    checkHipErrors(hipGraphDestroy(graph));
    checkHipErrors(hipStreamDestroy(stream));
    checkHipErrors(hipFree(d_buf));
    return 0;
}
