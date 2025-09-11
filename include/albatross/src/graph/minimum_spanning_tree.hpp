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
      : a(a_), b(b_), cost(cost_), id(id_) {}

  bool operator<(const Edge &other) const {
    return std::tie(this->cost, this->a, this->b) <
           std::tie(other.cost, other.a, other.b);
  }

  bool operator==(const Edge &other) const {
    return std::tie(this->cost, this->a, this->b) ==
           std::tie(other.cost, other.a, other.b);
  }

  VertexType a;
  VertexType b;
  double cost;
  // This can be used to keep track of a mapping between a real world problem
  // and the equivalent graph theory problem
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

/*
 * This implementation of Prim's algorithm was based largely off the following
 * references:
 *
 *   https://en.wikipedia.org/wiki/Prim%27s_algorithm
 *   https://algs4.cs.princeton.edu/43mst/
 *   https://www.algotree.org/algorithms/minimum_spanning_tree/prims_c++/
 *
 * The general idea is that we start with an arbitrary vertex.  Here we've
 * picked a vertex from the edge with maximum cost which we set as the current
 * vertex.  The algorithm then proceeds by:
 *
 * 1) Finding all adjacent vertices which are not part of the output
 * 2) Placing all the corresponding edges into the priority queue.
 * 3) By popping from the priority queue we end up with the highest remaining
 * edge. 4) If the highest edge leads to a vertex which isn't in the output:
 *     - Move to the new vertex, add all it's adjacent unused edges to the
 * queue. Else:
 *     - Repeat starting at 3.
 */
template <typename VertexType>
Graph<VertexType> maximum_spanning_tree(const Graph<VertexType> &graph) {
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

  const auto maximum_required_edges = graph.vertices.size() - 1;
  while (!queue.empty() && output.edges.size() < maximum_required_edges) {
    const Edge<VertexType> next_edge = queue.top();
    queue.pop();

    const VertexType &next_vertex = next_edge.b;

    if (!contains_vertex(output, next_vertex)) {
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

/*
 * This implementation of Kruskal's algorithm was based largely off the
 * following reference:
 *
 *   https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
 *
 * The general idea is that we add edges with the minimum weight until each edge
 * is included, skipping addition of edges that would form a closed loop. To
 * track closed loops and disjoint trees, the class associates each node with a
 * tree, initialized to each node in a unique tree. When a new edge is added,
 * the trees associated with the edge's nodes are merged into one tree. If a
 * proposed edge already has both nodes in the same tree, a closed loop would be
 * formed, and the edge can be rejected.
 */
template <typename VertexType> class KruskalAlgoRunner {
public:
  KruskalAlgoRunner(const Graph<VertexType> &input_graph)
      : sorted_graph_(input_graph), vertices_() {
    std::sort(sorted_graph_.edges.begin(), sorted_graph_.edges.end());
    size_t tree_id = 0;
    for (const auto &v : sorted_graph_.vertices) {
      vertices_.emplace_back(v, tree_id);
      tree_id++;
    }
  };

  Graph<VertexType> run() {
    Graph<VertexType> output;
    for (const auto &e : sorted_graph_.edges) {
      const auto vertex_and_tree_a = find_vertex_or_assert(e.a);
      const auto vertex_and_tree_b = find_vertex_or_assert(e.b);
      if (vertex_and_tree_a.tree != vertex_and_tree_b.tree) {
        merge_trees(vertex_and_tree_a.tree, vertex_and_tree_b.tree);
        add_edge(e, &output);
      }
    }
    return output;
  }

private:
  struct VertexWithTreeID {
    VertexWithTreeID(const VertexType &v_, const size_t &tree_)
        : v(v_), tree(tree_) {}
    VertexType v;
    size_t tree;
  };

  VertexWithTreeID &find_vertex_or_assert(const VertexType &x) {
    auto is_x = [&x](const auto &p) { return p.v == x; };
    const auto iter = std::find_if(vertices_.begin(), vertices_.end(), is_x);
    ALBATROSS_ASSERT(iter != vertices_.end());
    return *iter;
  }

  void merge_trees(const size_t old_tree, const size_t new_tree) {
    for (size_t i = 0; i < vertices_.size(); ++i) {
      if (vertices_[i].tree == old_tree) {
        vertices_[i].tree = new_tree;
      }
    }
  }

  Graph<VertexType> sorted_graph_;
  std::vector<VertexWithTreeID> vertices_;
};

template <typename VertexType>
Graph<VertexType> minimum_spanning_forest(const Graph<VertexType> &graph) {
  if (graph.edges.size() == 0) {
    return Graph<VertexType>();
  }

  KruskalAlgoRunner<VertexType> algo_runner(graph);
  return algo_runner.run();
}

/*
 * The maximum_spanning_tree with negative costs is equivalent to the
 * minimum_spanning_tree.
 */
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

template <typename VertexType>
Graph<VertexType> maximum_spanning_forest(const Graph<VertexType> &graph) {
  Graph<VertexType> inverse_cost_graph(graph);
  for (auto &edge : inverse_cost_graph.edges) {
    edge.cost *= -1;
  }

  auto output = minimum_spanning_forest(inverse_cost_graph);
  for (auto &edge : output.edges) {
    edge.cost *= -1;
  }
  return output;
}

template <typename VertexType>
inline std::ostream &operator<<(std::ostream &os,
                                const Edge<VertexType> &edge) {
  os << edge.a << " <--> " << edge.b << " [" << edge.cost << "]";
  return os;
}

template <typename VertexType>
inline std::ostream &operator<<(std::ostream &os,
                                const Graph<VertexType> &graph) {
  for (const auto &edge : graph.edges) {
    std::cout << edge << std::endl;
  }
  return os;
}

} // namespace albatross

#endif
