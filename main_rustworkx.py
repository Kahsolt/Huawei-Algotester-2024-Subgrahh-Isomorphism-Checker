#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/03 

# https://www.rustworkx.org/apiref/rustworkx.graph_vf2_mapping.html#rustworkx.graph_vf2_mapping

from sys import stdin
from typing import List, Tuple

from rustworkx import PyGraph
from rustworkx import graph_vf2_mapping


def read_graph() -> PyGraph:
  n, m = [int(x) for x in stdin.readline().split()]
  a: List[int] = [int(x) for x in stdin.readline().split()]
  e: List[Tuple[int, int]] = [tuple(int(x) - 1 for x in stdin.readline().split()) for _ in range(m)]

  g = PyGraph()
  node_ids = []
  for it in a:
    node_ids.append(g.add_node(it))
  for u, v in e:
    g.add_edge(node_ids[u], node_ids[v], 1.0)
  return g


def find_isomorphism(g:PyGraph, s:PyGraph):
  # 从大图 g 中枚举出小图 s 的同构子图，找到一个就算成功；按 label 一致判定节点等价性
  vf2 = graph_vf2_mapping(g, s, node_matcher=(lambda x, y: x == y), subgraph=True)
  for it in vf2:
    return [e[0] for e in sorted(it.items(), key=lambda e: e[-1])]


def run_from_stdin():
  g = read_graph()
  k = int(stdin.readline())
  res: List[Tuple[int]] = []
  for i in range(k):
    s = read_graph()
    f = find_isomorphism(g, s)
    if f: res.append((i, f))

  print(len(res))
  for i, f in res:
    print(i + 1, end='')
    for x in f:
      print(f' {x + 1}', end='')
    print()


def run_from_random():
  from data import get_query_pair
  def graph_to_pygraph(graph) -> PyGraph:
    a, e = graph
    g = PyGraph()
    node_ids = []
    for it in a:
      node_ids.append(g.add_node(it))
    for u, v in e:
      g.add_edge(node_ids[u], node_ids[v], 1.0)
    return g

  G, S_list = get_query_pair()
  for i, S in enumerate(S_list):
    g = graph_to_pygraph(G)
    s = graph_to_pygraph(S)

    f = find_isomorphism(g, s)
    if f:
      print(f'[{i}] ' + ' '.join(str(x + 1) for x in f))


if __name__ == '__main__':
  #run_from_stdin()
  run_from_random()
