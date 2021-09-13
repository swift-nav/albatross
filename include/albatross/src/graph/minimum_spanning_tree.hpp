/*
 * Copyright (C) 2021 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_GRAPH_MINIMUM_SPANNING_TREE_HPP_
#define ALBATROSS_GRAPH_MINIMUM_SPANNING_TREE_HPP_

namespace albatross {

template <typename VertexType> struct Edge {

  Edge(VertexType a_, VertexType b_, double cost_, std::size_t id_ = 0)
      : a(a_), b(b_), cost(cost_), id(id_){};

  bool operator<(const Edge &other) const {
    return std::tie(this->cost, this->a, this->b) <
           std::tie(other.cost, other.a, other.b);
  }

  bool operator==(const Edge &other) const {
    return std::tie(this->cost, this->a, this->b) ==
           std::tie(other.cost, other.a, other.b);
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const Edge<VertexType> &edge) {
    os << edge.a << " <--> " << edge.b << " [" << edge.cost << "]";
    return os;
  }

  VertexType a;
  VertexType b;
  double cost;
  std::size_t id;
};

template <typename VertexType>
std::set<VertexType>
compute_vertices(const std::vector<Edge<VertexType>> &edges) {
  std::set<VertexType> vertices;
  for (const auto &edge : edges) {
    vertices.insert(edge.a);
    vertices.insert(edge.b);
  }
  return vertices;
}

template <typename VertexType> struct Graph {
  std::vector<Edge<VertexType>> edges;
  std::set<VertexType> vertices;

  friend std::ostream &operator<<(std::ostream &os,
                                  const Graph<VertexType> &graph) {
    for (const auto &edge : graph.edges) {
      std::cout << edge << std::endl;
    }
    return os;
  }
};

template <typename VertexType>
Graph<VertexType> create_graph(const std::vector<Edge<VertexType>> &edges) {
  Graph<VertexType> graph;
  graph.edges = edges;
  graph.vertices = compute_vertices(edges);
  return graph;
}

template <typename VertexType>
void add_edge(const Edge<VertexType> &edge, Graph<VertexType> *graph) {
  graph->edges.push_back(edge);
  graph->vertices.insert(edge.a);
  graph->vertices.insert(edge.b);
}

template <typename VertexType>
bool contains_vertex(const Graph<VertexType> &graph, const VertexType &vertex) {
  return graph.vertices.find(vertex) != graph.vertices.end();
}

template <typename VertexType>
std::map<VertexType, std::vector<Edge<VertexType>>>
adjacency_map(const Graph<VertexType> &graph) {
  std::map<VertexType, std::vector<Edge<VertexType>>> output;

  auto construct_or_push = [&output](const VertexType &key,
                                     const Edge<VertexType> &new_edge) {
    auto value = output.find(key);
    if (value == output.end()) {
      output[key] = {new_edge};
    } else {
      value->second.push_back(new_edge);
    }
  };

  for (const auto &edge : graph.edges) {
    construct_or_push(edge.a, edge);
    Edge<VertexType> flipped(edge);
    std::swap(flipped.a, flipped.b);
    construct_or_push(edge.b, flipped);
  }
  return output;
}

template <typename VertexType>
Graph<VertexType> maximum_spanning_tree(const Graph<VertexType> &graph) {
  // https://en.wikipedia.org/wiki/Prim%27s_algorithm
  // https://algs4.cs.princeton.edu/43mst/
  // https://www.algotree.org/algorithms/minimum_spanning_tree/prims_c++/
  Graph<VertexType> output;

  const auto adjacency = adjacency_map(graph);

  std::priority_queue<Edge<VertexType>> queue;

  if (graph.edges.size() == 0) {
    return output;
  }

  const Edge<VertexType> first_edge =
      *std::max_element(graph.edges.begin(), graph.edges.end());

  for (const auto &edge : adjacency.at(first_edge.a)) {
    if (!contains_vertex(output, edge.b)) {
      queue.push(edge);
    }
  }

  double total_cost = 0;
  while (!queue.empty()) {
    const Edge<VertexType> next_edge = queue.top();
    queue.pop();

    const VertexType &next_vertex = next_edge.b;

    if (!contains_vertex(output, next_vertex)) {
      total_cost += next_edge.cost;
      add_edge(next_edge, &output);

      for (const auto &edge : adjacency.at(next_vertex)) {
        if (!contains_vertex(output, edge.b)) {
          queue.push(edge);
        }
      }
    }
  }
  return output;
}

template <typename VertexType>
Graph<VertexType> minimum_spanning_tree(const Graph<VertexType> &graph) {
  Graph<VertexType> inverse_cost_graph(graph);
  for (auto &edge : inverse_cost_graph.edges) {
    edge.cost *= -1;
  }

  auto output = maximum_spanning_tree(inverse_cost_graph);
  for (auto &edge : output.edges) {
    edge.cost *= -1;
  }
  return output;
}

} // namespace albatross

#endif
