// Copyright (c) 2025 The Khronos Group, Inc.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
#include <utility>
using namespace sycl;  // (optional) avoids need for "sycl::" before SYCL names

constexpr size_t N = 100;
constexpr size_t Iter = 10;

int main() {
  // Create a queue to work on
  queue myQueue({property::queue::in_order{}});

  // Create some 1D arrays of float
  float* AOld = malloc_device<float>(N, myQueue);
  float* ANew = malloc_device<float>(N, myQueue);

    // Simple 1D stencil function to be used by a parallel kernel 
    auto step = [=](sycl::id<1> index) {
      ANew[index + 1] = AOld[index] / 2 + AOld[index + 2] / 2;
    };

  khr::command_graph myGraph{myQueue};

  myGraph.begin_recording(myQueue);

  // Unroll iterative 1D solver by a factor of two to enable
  // recording into an immutable graph
  //
  // Record an asynchronous kernel to compute even timesteps
  myQueue.submit([&](handler& cgh) {
    cgh.parallel_for(range<1>{N - 2}, step);
  });

  std::swap(AOld, ANew);

  // Record an asynchronous kernel to compute odd timesteps
  myQueue.submit([&](handler& cgh) {
    cgh.parallel_for(range<1>{N - 2}, step);
  });

  myGraph.end_recording();

  khr::command_graph<khr::graph_state::executable> myExecGraph =
      myGraph.finalize();

  // Initialize data on device
  myQueue.submit([&](handler& cgh) {
    cgh.single_task([=]() {
      ANew[0] = ANew[N - 1] = AOld[0] = AOld[N - 1] = 100.0f;
      for (size_t i = 1; i < N - 1; i++) AOld[i] = 0.0f;
    });
  });

  // Replay linear graph to calculate Iter timesteps
  for (size_t i = 0; i < Iter; i += 2) myQueue.khr_graph(myExecGraph);

  myQueue.wait();

  std::cout << "Success!" << std::endl;

  free(AOld, myQueue);
  free(ANew, myQueue);
  return 0;
}
