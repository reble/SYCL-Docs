// Copyright (c) 2011-2025 The Khronos Group, Inc.
// SPDX-License-Identifier: Apache-2.0

namespace sycl::khr {

enum class graph_state : /*unspecified*/ { modifiable, executable };

template <graph_state State = graph_state::modifiable>
class command_graph {};

template <>
class command_graph<graph_state::modifiable> {
 public:
  command_graph(const context& syclContext, const device& syclDevice,
                const property_list& propList = {});

  command_graph(const queue& syclQueue, const property_list& propList = {});

  /* -- common interface members -- */

  command_graph<graph_state::executable> finalize(
      const property_list& propList = {}) const;

  void begin_recording(queue& recordingQueue,
                       const property_list& propList = {});
  void begin_recording(const std::vector<queue>& recordingQueues,
                       const property_list& propList = {});

  void end_recording();
  void end_recording(queue& recordingQueue);
  void end_recording(const std::vector<queue>& recordingQueues);

  node add(const property_list& propList = {});

  template <typename T>
  node add(T cgf, const property_list& propList = {});

  void make_edge(node& src, node& dest);

  void print_graph(std::string path, bool verbose = false) const;

  std::vector<node> get_nodes() const noexcept;
  std::vector<node> get_root_nodes() const noexcept;
};

template <>
class command_graph<graph_state::executable> {
 public:
  command_graph() = delete;
};

}  // namespace sycl::khr
