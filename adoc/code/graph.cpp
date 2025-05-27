#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;  // (optional) avoids need for "sycl::" before SYCL names

constexpr size_t N = 100;
constexpr size_t Iter = 10;

int main() {
  // Create a queue to work on.
  queue myQueue({property::queue::in_order{}});

  // Create some 1D arrays of float
  float* A_old = malloc_device<float>(N, myQueue);
  float* A_new = malloc_device<float>(N, myQueue);
    
  {
    khr::command_graph myGraph{myQueue};

    myGraph.begin_recording(myQueue);

    // Record an asynchronous kernel to compute even iterations
    myQueue.submit([&](handler& cgh) {
      // Enqueue a parallel kernel iterating on a N 1D iteration space
      cgh.parallel_for(range<1>{N-2}, [=](id<1> index) {
        A_new[index+1] = A_old[index] / 2 + A_old[index+2] / 2;
      });
    });

    std::swap(A_old, A_new);

    // Record an asynchronous kernel to compute odd iterations
    myQueue.submit([&](handler& cgh) {
      // Enqueue a parallel kernel iterating on a N 1D iteration space
      cgh.parallel_for(range<1>{N-2}, [=](id<1> index) {
        A_new[index+1] = A_old[index] / 2 + A_old[index+2] / 2;
      });
    });

    myGraph.end_recording();

    khr::command_graph<khr::graph_state::executable> myExecGraph =
        myGraph.finalize();

    // Replay linear graph which holds two iterations
    for(size_t i = 0; i < Iter; i+=2)
      myQueue.khr_graph(myExecGraph);
      
    myQueue.wait();
  }

  std::cout << "Success!" << std::endl;

  free(A_old, myQueue);
  free(A_new, myQueue);
  return 0;
}

