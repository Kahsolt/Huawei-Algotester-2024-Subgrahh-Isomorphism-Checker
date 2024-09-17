#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/03 

# https://networkx.org/documentation/stable/reference/algorithms/isomorphism.html

from sys import stdin
from typing import List, Tuple

from networkx import Graph
from networkx.algorithms.isomorphism.vf2pp import vf2pp_isomorphism


def read_graph() -> Graph:
  n, m = [int(x) for x in stdin.readline().split()]
  a: List[int] = [int(x) for x in stdin.readline().split()]
  e: List[Tuple[int, int]] = [tuple(int(x) - 1 for x in stdin.readline().split()) for _ in range(m)]

  g = Graph()
  for i, it in enumerate(a):
    g.add_node(i, label=it)
  for u, v in e:
    g.add_edge(u, v)
  return g


def find_isomorphism(g:Graph, s:Graph):
  # TODO: 从大图 g 中扣一个特征和 s 一致的图
  g_subs = [g]
  for g_sub in g_subs:
    # 尝试寻找两个同构图的映射关系；按 label 一致判定节点等价性
    vf2pp = vf2pp_isomorphism(g_sub, s, node_label='label')
    for it in vf2pp:
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
  def graph_to_nxgraph(graph) -> Graph:
    a, e = graph
    g = Graph()
    for i, it in enumerate(a):
      g.add_node(i, label=it)
    for u, v in e:
      g.add_edge(u, v)
    return g

  G, S_list = get_query_pair()
  for i, S in enumerate(S_list):
    g = graph_to_nxgraph(G)
    s = graph_to_nxgraph(S)

    f = find_isomorphism(g, s)
    if f:
      print(f'[{i}] ' + ' '.join(str(x + 1) for x in f))


if __name__ == '__main__':
  #run_from_stdin()
  run_from_random()
