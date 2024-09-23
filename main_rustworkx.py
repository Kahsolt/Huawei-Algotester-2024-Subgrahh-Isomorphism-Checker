#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/03 

# https://www.rustworkx.org/apiref/rustworkx.graph_vf2_mapping.html#rustworkx.graph_vf2_mapping

from time import time_ns
ts_start = time_ns()
TIME_LIMIT = 59   # s
TTL = ts_start + int(TIME_LIMIT * 10**9)  # ns

from sys import stdin, stdout, platform
from typing import List, Tuple

from rustworkx import PyGraph
from rustworkx import graph_vf2_mapping


def read_graph() -> PyGraph:
  n, m = [int(x) for x in stdin.readline().split()]
  a = [int(x) for x in stdin.readline().split()]
  e = [tuple(int(x) - 1 for x in stdin.readline().split()) for _ in range(m)]

  g = PyGraph()
  node_ids = []
  for it in a:
    node_ids.append(g.add_node(it))
  for u, v in e:
    g.add_edge(node_ids[u], node_ids[v], 1.0)
  return g


def find_isomorphism(g:PyGraph, s:PyGraph):
  # 从大图 g 中枚举出小图 s 的同构子图，找到一个就算成功；按 label 一致判定节点等价性
  vf2 = graph_vf2_mapping(g, s, node_matcher=(lambda x, y: x == y), subgraph=True, induced=False)
  for it in vf2:
    return [e[0] for e in sorted(it.items(), key=lambda e: e[-1])]


if __name__ == '__main__':
  g = read_graph()
  k = int(stdin.readline())
  res: List[Tuple[int]] = []
  for i in range(1, 1+k):
    if time_ns() > TTL: break

    s = read_graph()
    f = find_isomorphism(g, s)
    if f: res.append((i, f))

  stdout.write(str(len(res)))
  stdout.write('\n')
  for i, f in res:
    stdout.write(str(i))
    for x in f:
      stdout.write(' ')
      stdout.write(str(x + 1))
    stdout.write('\n')
  stdout.flush()

  if platform == 'win32':
    print()
    print(f'RX TIME: {(time_ns() - ts_start) / 10**9:.2f}')
