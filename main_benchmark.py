#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/17 

# benchmark different implmentations

from time import time
from typing import List
from tqdm import tqdm
from data import QUERY_GRAPH_EDGES, TARGET_GRAPHS_LEN, get_query_pair

# ↓↓ rustworkx baseline
from rustworkx import PyGraph, graph_vf2_mapping
# ↓↓ networkx impl. migrated
from main import Graph, find_isomorphism, Labels, Edges, Result
# ↓↓ rustworkx impl. migrated
from main_rustworkx_impl import StableGraph, Vf2Algorithm


def make_graph(labels:Labels, edges:Edges, offset_by_one:bool=False) -> Graph:
  if offset_by_one:
    edges = [(u-1, v-1) for u, v in edges]
  return Graph(labels, edges)

def to_rx(graph:Graph) -> PyGraph:
  g = PyGraph()
  node_ids: List[int] = []
  for it in graph.labels:
    node_ids.append(g.add_node(it))
  for u, v in graph.edges:
    g.add_edge(node_ids[u], node_ids[v], 1.0)
  return g

def find_isomorphism_rx(g:PyGraph, s:PyGraph) -> Result:
  # 从大图 g 中枚举出小图 s 的同构子图，找到一个就算成功；按 label 一致判定节点等价性
  vf2 = graph_vf2_mapping(g, s, node_matcher=(lambda x, y: x == y), subgraph=True, induced=False)
  for it in vf2:
    return [e[0] for e in sorted(it.items(), key=lambda e: e[-1])]

def to_rx2py(graph:Graph) -> StableGraph:
  g = StableGraph(graph.n, graph.m)
  node_ids: List[int] = []
  for it in graph.labels:
    node_ids.append(g.add_node(it))
  for u, v in graph.edges:
    g.add_edge(node_ids[u], node_ids[v])
  return g

def find_isomorphism_rx2py(g:StableGraph, s:StableGraph) -> Result:
  mapping = Vf2Algorithm(g, s).next_vf2()
  if mapping is not None:
    return [e[0] for e in sorted(mapping.items(), key=lambda e: e[-1])]


def run_demo():
  G  = make_graph(
    [4, 1, 3, 2, 1],
    [
      (1, 2),
      (2, 3),
      (3, 4),
      (1, 3),
      (2, 4),
      (4, 5),
    ],
    offset_by_one=True,
  )
  S1 = make_graph(
    [1, 2, 4],
    [
      (1, 2),
      (2, 3),
    ],
    offset_by_one=True,
  )
  S2 = make_graph(
    [1, 2, 3, 4],
    [
      (1, 2),
      (2, 3),
      (3, 4),
      (1, 4),
      (1, 3),
    ],
    offset_by_one=True,
  )

  res = []
  for i, S in enumerate([S1, S2], start=1):
    f = find_isomorphism(G, S)
    if f: res.append((i, f))
  print(len(res))
  for i, f in res:
    print(i, end='')
    for x in f:
      print(f' {x + 1}', end='')
    print()

def run_random(gid:int=12, n_edges:int=8, log:bool=False) -> bool:
  if log: print(f'[Run] graph_id={gid}, subgraph_edges={n_edges}')

  G, S_list = get_query_pair(gid, n_edges)
  g = make_graph(*G)
  g_rx = to_rx(g)
  g_rx2py = to_rx2py(g)

  ts_sum = 0.0
  results_nx = []
  found_nx = []
  for i, S in enumerate(S_list, start=1):
    s = make_graph(*S)
    ts_start = time()
    f = find_isomorphism(g, s)
    ts_sum += time() - ts_start
    results_nx.append(f)
    if f: found_nx.append(i) 
  if log: print(f'>> [NetworkxImpl] time cost: {ts_sum:.3f}')

  ts_sum = 0.0
  results_rx = []
  found_rx = []
  for i, S in enumerate(S_list, start=1):
    s_rx = to_rx(make_graph(*S))
    ts_start = time()
    f = find_isomorphism_rx(g_rx, s_rx)
    ts_sum += time() - ts_start
    results_rx.append(f)
    if f: found_rx.append(i)
  if log: print(f'>> [Rustworkx] time cost: {ts_sum:.3f}')

  ts_sum = 0.0
  results_rx2py = []
  found_rx2py = []
  for i, S in enumerate(S_list, start=1):
    s_rx2py = to_rx2py(make_graph(*S))
    ts_start = time()
    f = find_isomorphism_rx2py(g_rx2py, s_rx2py)
    ts_sum += time() - ts_start
    results_rx2py.append(f)
    if f: found_rx2py.append(i) 
  if log: print(f'>> [RustworkxImpl] time cost: {ts_sum:.3f}')

  chk = found_nx == found_rx == found_rx2py

  if log and not chk:
    print('>> check FAILED! :/')
    print(f'   found_nx ({len(found_nx)}):', found_nx)
    print(f'   found_rx ({len(found_rx)}):', found_rx)
    print(f'   found_rx2py ({len(found_rx2py)}):', found_rx2py)
    print()

  return chk


if __name__ == '__main__':
  n_lim = 10    # TARGET_GRAPHS_LEN

  pbar = tqdm(total=n_lim * len(QUERY_GRAPH_EDGES))
  ok = 0
  # 选 1 张主图 (共10000张)
  for gid in range(n_lim):
    # 跑各种边数的模式子图，各1000个模式
    for n_edges in QUERY_GRAPH_EDGES:
      ok += run_random(gid, n_edges, log=False)
      pbar.update()
      pbar.set_postfix({'ok': ok, 'sr': ok / pbar.n})
