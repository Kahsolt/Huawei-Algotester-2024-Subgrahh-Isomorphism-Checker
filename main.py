#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/17 

import os
DEBUG = os.getenv('LOCAL_DEBUG')

from time import time_ns
ts_start = time_ns()
TIME_LIMIT = 59.9   # s
TTL = ts_start + int(TIME_LIMIT * 10**9)  # ns
TS_query: int = None
TTL_query: int = None

from sys import stdin, stdout, platform
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Union

Labels = List[int]
Nodes = List[int]
Edges = List[Tuple[int, int]]
Degrees = List[int]
Groups = Dict[int, Set[int]]
Mapping = Dict[int, int]
Result = Tuple[int]

class Graph:

  def __init__(self, labels:Labels, edges:Edges):
    self.n = len(labels)
    self.m = len(edges)
    self.labels = labels
    self.edges = edges

    self.degree = [0] * self.n
    self.adj = [set() for _ in range(self.n)]
    for u, v in edges:
      self.degree[u] += 1
      self.degree[v] += 1
      self.adj[u].add(v)
      self.adj[v].add(u)

  def __getitem__(self, i:int):
    return self.adj[i]

# Info
G1: Graph
G2: Graph
nodes_of_G1Labels: Groups
nodes_of_G2Labels: Groups
G1_nodes_cover_degree: Groups
G2_nodes_cover_degree: Groups
label_rarity_G2: Dict[int, int]
# State
mapping: Dict[int, int]           # subgraph (u) -> graph (v)
reverse_mapping: Dict[int, int]   # graph (v) -> subgraph (u)
T1: Set[int]          # Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are neighbors of nodes that are (the frontiers)
T2: Set[int]

def vf2pp_find_isomorphism() -> Mapping:
  # 剪枝检查: 大图覆盖子图标签 (存在性)
  if not set(nodes_of_G1Labels).issubset(nodes_of_G2Labels): return
  # 剪枝检查: 大图覆盖子图标签 (计数)，不涨分
  #for lbl, G1_nodes in nodes_of_G1Labels.items():
  #  if len(G1_nodes) > len(nodes_of_G2Labels[lbl]):
  #    return
  # 剪枝检查: 大图覆盖子图度数 (存在性)
  if not set(G1_nodes_cover_degree).issubset(G2_nodes_cover_degree): return
  # 剪枝检查: 大图覆盖子图度数 (计数)，不涨分
  #for deg, G1_nodes in G1_nodes_cover_degree.items():
  #  if len(G1_nodes) > len(G2_nodes_cover_degree[deg]):
  #    return

  # 确定最优的子图顶点匹配顺序
  node_order = _matching_order()
  if DEBUG: print('node_order:', node_order)
  # 初始化DFS栈
  stack: List[int, Nodes] = []
  candidates = iter(_find_candidates(node_order[0]))
  stack.append((node_order[0], candidates))
  matching_node = 1
  # 开始DFS!!
  while stack:
    ts = time_ns()
    if ts > TTL or ts > TTL_query: return
    current_node, candidate_nodes = stack[-1]

    try:
      candidate = next(candidate_nodes)
    except StopIteration:
      stack.pop()
      matching_node -= 1
      if stack:
        popped_node1, _ = stack[-1]
        popped_node2 = mapping[popped_node1]
        mapping.pop(popped_node1)
        reverse_mapping.pop(popped_node2)
        _restore_Tinout(popped_node1, popped_node2)
      continue

    cut = _cut_PT(current_node, candidate)
    if DEBUG: print(f"u->v: {current_node} -> {candidate} cut={cut}")

    if not cut:
      if len(mapping) == G1.n - 1:
        mapping[current_node] = candidate
        return mapping

      mapping[current_node] = candidate
      reverse_mapping[candidate] = current_node
      _update_Tinout(current_node, candidate)
      candidates = iter(_find_candidates(node_order[matching_node]))
      stack.append((node_order[matching_node], candidates))
      matching_node += 1

def groups(many_to_one:Union[dict, list]) -> Groups:
  one_to_many = defaultdict(set)
  for v, k in (many_to_one.items() if isinstance(many_to_one, dict) else enumerate(many_to_one)):
    one_to_many[k].add(v)
  return dict(one_to_many)

def groups_to_accumulated_groups(group:dict) -> Groups:
  group_acc = defaultdict(set)
  for deg in sorted(group):
    nodes = group[deg]
    for d in range(deg, 0, -1):
      group_acc[d].update(nodes)
    group_acc[deg] = nodes
  return group_acc

def _matching_order():
  # 大图各label计数
  label_rarity = label_rarity_G2.copy()
  # 子图未排序节点 & 各节点已征用度数 (拟连通度)    # TODO: 改为百分比(?)
  V1_unordered = set(range(G1.n))
  used_degrees = {node: 0 for node in V1_unordered}
  # 子图已排序节点
  node_order: Nodes = []
  # 排序！
  while V1_unordered:
    # 未排序节点中label最罕见的节点
    max_rarity = min(label_rarity[G1.labels[x]] for x in V1_unordered)
    if DEBUG: print('max_rarity:', max_rarity)
    rarest_nodes = [n for n in V1_unordered if label_rarity[G1.labels[n]] == max_rarity]
    # 其中度最大的一个
    max_node = max(rarest_nodes, key=lambda e: G1.degree[e])
    if DEBUG: print('max_node:', max_node)
    # 宽搜处理整个连通域
    current_layer = [max_node]
    visited = {max_node}
    while current_layer:
      nodes_to_add = current_layer.copy()
      while nodes_to_add:
        # 近邻中拟连通度数最大的节点
        max_used_degree = max(used_degrees[n] for n in nodes_to_add)
        if DEBUG: print('max_used_degree:', max_used_degree)
        max_used_degree_nodes = [n for n in nodes_to_add if used_degrees[n] == max_used_degree]
        # 其中度最大的的节点
        max_degree = max(G1.degree[n] for n in max_used_degree_nodes)
        if DEBUG: print('max_degree:', max_degree)
        max_degree_nodes = [n for n in max_used_degree_nodes if G1.degree[n] == max_degree]
        # 其中最label最罕见一个
        next_node = min(max_degree_nodes, key=lambda x: label_rarity[G1.labels[x]])
        if DEBUG: print('next_node:', next_node)
        # 选定，加入排序！
        nodes_to_add.remove(next_node)
        V1_unordered.discard(next_node)
        node_order.append(next_node)
        # 更新辅助统计信息
        label_rarity[G1.labels[next_node]] -= 1
        for node in G1[next_node]:
          used_degrees[node] += 1
      next_layer: Nodes = []
      for node in current_layer:
        for child in G1[node]:
          if child not in visited:
            visited.add(child)
            next_layer.append(child)
      current_layer = next_layer
  return node_order

def _find_candidates(u:int):
  # 节点 u 的一些近邻已在映射中？
  covered_neighbors = [nbr for nbr in G1[u] if nbr in mapping]
  if DEBUG: print('covered_neighbors:', covered_neighbors)
  # 匹配子图节点 u 标签的大图节点 v
  valid_label_nodes = nodes_of_G2Labels[G1.labels[u]]
  if DEBUG: print('valid_label_nodes:', valid_label_nodes)
  # 覆盖子图节点 u 度数的大图节点 v
  valid_degree_nodes = G2_nodes_cover_degree[G1.degree[u]]
  if DEBUG: print('valid_degree_nodes:', valid_degree_nodes)
  # 初始情况，从 G2 全图选匹配点
  if not covered_neighbors:
    if DEBUG: print("case-init")
    candidates = set(valid_label_nodes)   # 与子图节点 u 标签一致的大图节点 v
    if DEBUG: print('nxt:', candidates)
    candidates.intersection_update(valid_degree_nodes)  # 节点 v 需覆盖节点 u 的度
    if DEBUG: print('nxt:', candidates)
    candidates.difference_update(reverse_mapping)       # 节点 v 未被映射
    if DEBUG: print('nxt:', candidates)
    return candidates
  # 后续情况，在 G2 已映射支撑集的近邻中选匹配点
  if DEBUG: print("case-succ")
  nbr = covered_neighbors[0]
  common_nodes = set(G2[mapping[nbr]])
  if DEBUG:
    print('nbr:', nbr)
    print('mapping[nbr]:', mapping[nbr])
    print('G2[mapping[nbr]]:', G2[mapping[nbr]])
    print('nxt-i:', common_nodes)
  for nbr in covered_neighbors[1:]:
    common_nodes.intersection_update(G2[mapping[nbr]])  # 所有已映射支撑集的近邻节点 v
    if DEBUG: print('nxt-i:', common_nodes)
  if DEBUG: print('nxt:', common_nodes)
  common_nodes.difference_update(reverse_mapping)       # 节点 v 未被映射
  if DEBUG: print('nxt:', common_nodes)
  common_nodes.intersection_update(valid_degree_nodes)  # 节点 v 需覆盖节点 u 的度
  if DEBUG: print('nxt:', common_nodes)
  common_nodes.intersection_update(valid_label_nodes)   # 节点 v 需与节点 u 标签一致
  if DEBUG: print('nxt:', common_nodes)
  return common_nodes

def _cut_PT(u:int, v:int):
  # 节点 u 和 v 的近邻的标签计数
  u_labels_successors = groups({n1: G1.labels[n1] for n1 in G1[u]})
  v_labels_successors = groups({n2: G2.labels[n2] for n2 in G2[v]})
  # 小图节点 u 的邻居标签必须被所配大图节点 v 的邻居标签覆盖 (是否存在此标签)
  if not set(u_labels_successors).issubset(v_labels_successors):
    if DEBUG: print("cut by size()")
    return True
  # 小图节点 u 的邻居数量必须被所配大图节点 v 的邻居数量覆盖
  for label, G1_nbh in u_labels_successors.items():
    # 对于同一个标签label，两个图上的近邻集 G1_nbh 和 G2_nbh
    G2_nbh = v_labels_successors[label]
    if len(T1.intersection(G1_nbh)) > len(T2.intersection(G2_nbh)):
      if DEBUG: print("cut by count_exist()")
      return True
  return False

def _update_Tinout(new_node1:int, new_node2:int):
  for succ in G1[new_node1]:
    if succ not in mapping:
      T1.add(succ)
  for succ in G2[new_node2]:
    if succ not in reverse_mapping:
      T2.add(succ)
  T1.discard(new_node1)
  T2.discard(new_node2)

def _restore_Tinout(popped_node1:int, popped_node2:int):
  for neighbor in G1[popped_node1]:
    if neighbor in mapping:
      T1.add(popped_node1)
    else:
      if any(nbr in mapping for nbr in G1[neighbor]):
        continue
      T1.discard(neighbor)
  for neighbor in G2[popped_node2]:
    if neighbor in reverse_mapping:
      T2.add(popped_node2)
    else:
      if any(nbr in reverse_mapping for nbr in G2[neighbor]):
        continue
      T2.discard(neighbor)


def read_graph() -> Graph:
  n, m = [int(x) for x in stdin.readline().split()]
  a = [int(x) for x in stdin.readline().split()]
  e = [tuple(int(x) - 1 for x in stdin.readline().split()) for _ in range(m)]
  return Graph(a, e)


if __name__ == '__main__':
  G2 = read_graph()
  if DEBUG: print('G2.degree:', G2.degree)
  nodes_of_G2Labels = groups(G2.labels)
  G2_nodes_cover_degree = groups_to_accumulated_groups(groups(G2.degree))
  if DEBUG: print('G2_nodes_cover_degree:', G2_nodes_cover_degree)
  label_rarity_G2 = {label: len(nodes) for label, nodes in nodes_of_G2Labels.items()}
  if DEBUG: print('label_rarity_G2:', label_rarity_G2)

  k = int(stdin.readline())
  TS_query = (TTL - ts_start) // k
  res: List[Result] = []
  for i in range(1, 1+k):
    if time_ns() > TTL: break

    TTL_query = ts_start + TS_query * i
    G1 = read_graph()
    if DEBUG: print('G1.degree:', G1.degree)
    nodes_of_G1Labels = groups(G1.labels)
    G1_nodes_cover_degree = groups_to_accumulated_groups(groups(G1.degree))
    if DEBUG: print('G1_nodes_cover_degree:', G1_nodes_cover_degree)
    mapping = {}
    reverse_mapping = {}
    T1 = set()
    T2 = set()

    fmap = vf2pp_find_isomorphism()
    if fmap is not None:
      f = tuple(fmap[i] for i in range(len(fmap)))
      res.append((i, f))

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
    print(f'PY TIME: {(time_ns() - ts_start) / 10**9:.2f}')
