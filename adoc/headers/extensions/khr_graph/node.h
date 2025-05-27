// Copyright (c) 2011-2025 The Khronos Group, Inc.
// SPDX-License-Identifier: Apache-2.0

namespace sycl::khr {
namespace property {
namespace node {
class depends_on {
 public:
  template <typename... NodeTN>
  depends_on(NodeTN... nodes);
};

class depends_on_all_leaves {
 public:
  depends_on_all_leaves() = default;
};
}  // namespace node
}  // namespace property
enum class node_type : /* unspecified */ {
  empty,
  subgraph,
  kernel,
  memcpy,
  memset,
  memfill,
  prefetch,
  memadvise,
  host_task,
};

class node {
 public:
  node() = delete;

  /* -- common interface members -- */

  node_type get_type() const noexcept;

  std::vector<node> get_predecessors() const noexcept;

  std::vector<node> get_successors() const noexcept;

  static node get_node_from_event(event nodeEvent);
};

}  // namespace sycl::khr
