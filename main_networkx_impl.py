#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/17 

# Copy & modify the impl. of networkx
# https://networkx.org/documentation/stable/reference/algorithms/isomorphism.html

from sys import stdin
from collections import defaultdict
from typing import List, Tuple, NamedTuple, Set, Dict

# 用于对拍测试
from rustworkx import PyGraph
from rustworkx import graph_vf2_mapping


''' ↓↓↓ networkx stuff  '''

from networkx import Graph

def vf2pp_find_isomorphism(graph:Graph, subgraph:Graph) -> Dict[int, int]:
  # 初始化图和状态信息 (注意图的编号顺序与论文相反!!)
  G1, G2 = subgraph, graph
  G1_degree = dict(G1.degree)
  G2_degree = dict(G2.degree)
  graph_params, state_params = _initialize_parameters(G1, G2, G1_degree, G2_degree)

  # 剪枝检查: 大图覆盖子图标签
  if not set(graph_params.nodes_of_G1Labels).issubset(graph_params.nodes_of_G2Labels): return
  # 剪枝检查: 大图覆盖子图度数计数
  if not set(graph_params.G1_nodes_cover_degree).issubset(graph_params.G2_nodes_cover_degree): return

  # just make short
  mapping = state_params.mapping    
  reverse_mapping = state_params.reverse_mapping

  # 确定最优的子图顶点匹配顺序
  node_order = _matching_order(graph_params)

  # 初始化DFS栈
  stack: List[int, List[int]] = []
  candidates = iter(_find_candidates(node_order[0], graph_params, state_params, G1_degree))
  stack.append((node_order[0], candidates))
  matching_node = 1
  # 开始DFS!!
  while stack:
    current_node, candidate_nodes = stack[-1]

    # 匹配失败，回溯
    try:
      candidate = next(candidate_nodes)
    except StopIteration:
      # If no remaining candidates, return to a previous state, and follow another branch
      stack.pop()
      matching_node -= 1
      if stack:
        # Pop the previously added u-v pair, and look for a different candidate _v for u
        popped_node1, _ = stack[-1]
        popped_node2 = mapping[popped_node1]
        mapping.pop(popped_node1)
        reverse_mapping.pop(popped_node2)
        _restore_Tinout(popped_node1, popped_node2, graph_params, state_params)
      continue

    # 匹配成功
    if not _cut_PT(current_node, candidate, graph_params, state_params):
      # 找到一个解
      if len(mapping) == G1.number_of_nodes() - 1:
        cp_mapping = mapping.copy()
        cp_mapping[current_node] = candidate
        return cp_mapping   # just need one!

      # Feasibility rules pass, so extend the mapping and update the parameters
      mapping[current_node] = candidate
      reverse_mapping[candidate] = current_node
      _update_Tinout(current_node, candidate, graph_params, state_params)
      # Append the next node and its candidates to the stack
      candidates = iter(_find_candidates(node_order[matching_node], graph_params, state_params, G1_degree))
      stack.append((node_order[matching_node], candidates))
      matching_node += 1

class GraphParameters(NamedTuple):
  G1: Graph
  G2: Graph
  G1_labels: Dict[int, int]
  G2_labels: Dict[int, int]
  nodes_of_G1Labels: Dict[int, Set[int]]
  nodes_of_G2Labels: Dict[int, Set[int]]
  G1_nodes_cover_degree: Dict[int, Set[int]]
  G2_nodes_cover_degree: Dict[int, Set[int]]

class StateParameters(NamedTuple):
  mapping: Dict[int, int]           # subgraph (u) -> graph (v)
  reverse_mapping: Dict[int, int]   # graph (v) -> subgraph (u)
  T1: Set[int]
  T1_in: Set[int]
  T1_tilde: Set[int]
  T1_tilde_in: Set[int]
  T2: Set[int]
  T2_in: Set[int]
  T2_tilde: Set[int]
  T2_tilde_in: Set[int]

def groups(many_to_one:dict) -> dict:
  one_to_many = defaultdict(set)
  for v, k in many_to_one.items():
    one_to_many[k].add(v)
  return dict(one_to_many)

def groups_to_accumulated_groups(group:dict) -> dict:
  group_acc: Dict[int, Set[int]] = {}
  for deg in sorted(group):
    nodes = group[deg]
    for v in group_acc.values():
      v.update(nodes)
    group_acc[deg] = nodes
  return group_acc

def bfs_layers(G:Graph, sources:List[int]):
  if sources in G: sources = [sources]

  current_layer = list(sources)
  visited = set(sources)
  while current_layer:
    cur_layer = current_layer.copy()
    yield cur_layer
    next_layer: List[int] = []
    for node in current_layer:
      for child in G[node]:
        if child not in visited:
          visited.add(child)
          next_layer.append(child)
    current_layer = next_layer

def _initialize_parameters(G1:Graph, G2:Graph, G1_degree:Dict[int, int], G2_degree:Dict[int, int]):
  G1_labels = dict(G1.nodes(data='label'))  # node_id => label
  G2_labels = dict(G2.nodes(data='label'))  # node_id => label

  G1_nodes_of_degree = groups(G1_degree)
  G1_nodes_cover_degree = groups_to_accumulated_groups(G1_nodes_of_degree)
  G2_nodes_of_degree = groups(G2_degree)
  G2_nodes_cover_degree = groups_to_accumulated_groups(G2_nodes_of_degree)

  graph_params = GraphParameters(
    G1,
    G2,
    G1_labels,
    G2_labels,
    groups(G1_labels),
    groups(G2_labels),
    G1_nodes_cover_degree,
    G2_nodes_cover_degree,
  )

  state_params = StateParameters(
    {},
    {},
    set(),
    set(),
    set(G1.nodes()),
    set(),
    set(),
    set(),
    set(G2.nodes()),
    set(),
  )

  return graph_params, state_params

def _matching_order(graph_params:GraphParameters) -> List[int]:
  G1, _, G1_labels, _, _, nodes_of_G2Labels, _, _ = graph_params

  # 大图各label计数
  label_rarity = {label: len(nodes) for label, nodes in nodes_of_G2Labels.items()}
  # 子图未排序节点 & 各节点已征用度数 (拟连通度)    # TODO: 改为百分比(?)
  V1_unordered = set(G1.nodes())
  used_degrees = {node: 0 for node in V1_unordered}
  # 子图已排序节点
  node_order: List[int] = []

  while V1_unordered:
    # 未排序节点中label最罕见的节点
    max_rarity = min(label_rarity[G1_labels[x]] for x in V1_unordered)
    rarest_nodes = [n for n in V1_unordered if label_rarity[G1_labels[n]] == max_rarity]
    # 其中度最大的一个
    max_node = max(rarest_nodes, key=G1.degree)
    # 宽搜处理整个连通域
    for nodes_to_add in bfs_layers(G1, max_node):
      while nodes_to_add:
        # 近邻中拟连通度数最大的节点
        max_used_degree = max(used_degrees[n] for n in nodes_to_add)
        max_used_degree_nodes = [n for n in nodes_to_add if used_degrees[n] == max_used_degree]
        # 其中度最大的的节点
        max_degree = max(G1.degree[n] for n in max_used_degree_nodes)
        max_degree_nodes = [n for n in max_used_degree_nodes if G1.degree[n] == max_degree]
        # 其中最label最罕见一个
        next_node = min(max_degree_nodes, key=lambda x: label_rarity[G1_labels[x]])
        # 选定，加入排序！
        nodes_to_add.remove(next_node)
        V1_unordered.discard(next_node)
        node_order.append(next_node)
        # 更新辅助统计信息
        label_rarity[G1_labels[next_node]] -= 1
        for node in G1.neighbors(next_node):
          used_degrees[node] += 1

  return node_order

def _find_candidates(u:int, graph_params:GraphParameters, state_params:StateParameters, G1_degree:Dict[int, int]):
  G1, G2, G1_labels, _, _, nodes_of_G2Labels, _, G2_nodes_cover_degree = graph_params
  mapping, reverse_mapping, _, _, _, _, _, _, T2_tilde, _ = state_params

  # 节点 u 的一些近邻已在映射中？
  covered_neighbors = [nbr for nbr in G1[u] if nbr in mapping]

  # 匹配子图节点 u 标签的大图节点 v
  valid_label_nodes = nodes_of_G2Labels[G1_labels[u]]
  # 覆盖子图节点 u 度数的大图节点 v
  valid_degree_nodes = G2_nodes_cover_degree[G1_degree[u]]

  # 初始情况，从 G2 全图选匹配点
  if not covered_neighbors:
    candidates = set(valid_label_nodes)   # 与子图节点 u 标签一致的大图节点 v
    candidates.intersection_update(valid_degree_nodes)  # 节点 v 需覆盖节点 u 的度
    candidates.intersection_update(T2_tilde)            # 节点 v 在 G2 图中 (??)
    candidates.difference_update(reverse_mapping)       # 节点 v 未被映射
    return candidates

  # 后续情况，在 G2 已映射支撑集的近邻中选匹配点
  nbr = covered_neighbors[0]
  common_nodes = set(G2[mapping[nbr]])
  for nbr in covered_neighbors[1:]:
    common_nodes.intersection_update(G2[mapping[nbr]])  # 所有已映射支撑集的近邻节点 v
  common_nodes.difference_update(reverse_mapping)       # 节点 v 未被映射
  common_nodes.intersection_update(valid_degree_nodes)  # 节点 v 需覆盖节点 u 的度
  common_nodes.intersection_update(valid_label_nodes)   # 节点 v 需与节点 u 标签一致
  return common_nodes

def _cut_PT(u:int, v:int, graph_params:GraphParameters, state_params:StateParameters):
  G1, G2, G1_labels, G2_labels, _, _, _, _ = graph_params
  _, _, T1, _, T1_tilde, _, T2, _, T2_tilde, _ = state_params

  u_labels_successors = groups({n1: G1_labels[n1] for n1 in G1[u]})
  v_labels_successors = groups({n2: G2_labels[n2] for n2 in G2[v]})

  # 小图节点 u 的邻居标签必须被所配大图节点 v 的邻居标签覆盖
  # if the neighbors of u, do not have the same labels as those of v, NOT feasible.
  if not set(u_labels_successors).issubset(v_labels_successors):
    return True

  # 小图节点 u 的邻居数量必须被所配大图节点 v 的邻居数量覆盖
  for label, G1_nbh in u_labels_successors.items():
    G2_nbh = v_labels_successors[label]
    if len(T1.intersection(G1_nbh)) > len(T2.intersection(G2_nbh)):
      return True
    if len(T1_tilde.intersection(G1_nbh)) > len(T2_tilde.intersection(G2_nbh)):
      return True

  return False

def _update_Tinout(new_node1:int, new_node2:int, graph_params:GraphParameters, state_params:StateParameters):
  G1, G2, _, _, _, _, _, _ = graph_params
  mapping, reverse_mapping, T1, _, T1_tilde, _, T2, _, T2_tilde, _ = state_params

  uncovered_successors_G1 = {succ for succ in G1[new_node1] if succ not in mapping}
  uncovered_successors_G2 = {succ for succ in G2[new_node2] if succ not in reverse_mapping}

  # Add the uncovered neighbors of node1 and node2 in T1 and T2 respectively
  T1.update(uncovered_successors_G1)
  T2.update(uncovered_successors_G2)
  T1.discard(new_node1)
  T2.discard(new_node2)

  T1_tilde.difference_update(uncovered_successors_G1)
  T2_tilde.difference_update(uncovered_successors_G2)
  T1_tilde.discard(new_node1)
  T2_tilde.discard(new_node2)

def _restore_Tinout(popped_node1:int, popped_node2:int, graph_params:GraphParameters, state_params:StateParameters):
  # If the node we want to remove from the mapping, has at least one covered neighbor, add it to T1.
  G1, G2, _, _, _, _, _, _ = graph_params
  mapping, reverse_mapping, T1, _, T1_tilde, _, T2, _, T2_tilde, _ = state_params

  is_added = False
  for neighbor in G1[popped_node1]:
    if neighbor in mapping:
      # if a neighbor of the excluded node1 is in the mapping, keep node1 in T1
      is_added = True
      T1.add(popped_node1)
    else:
      # check if its neighbor has another connection with a covered node. If not, only then exclude it from T1
      if any(nbr in mapping for nbr in G1[neighbor]):
        continue
      T1.discard(neighbor)
      T1_tilde.add(neighbor)

  # Case where the node is not present in neither the mapping nor T1. By definition, it should belong to T1_tilde
  if not is_added:
    T1_tilde.add(popped_node1)

  is_added = False
  for neighbor in G2[popped_node2]:
    if neighbor in reverse_mapping:
      is_added = True
      T2.add(popped_node2)
    else:
      if any(nbr in reverse_mapping for nbr in G2[neighbor]):
        continue
      T2.discard(neighbor)
      T2_tilde.add(neighbor)

  if not is_added:
    T2_tilde.add(popped_node2)

''' ↑↑↑ networkx stuff '''


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

def make_graph(lables:List[int], edges:List[Tuple[int, int]], offset_by_one:bool=True):
  offset = int(offset_by_one)
  g = Graph()
  for i, label in enumerate(lables):
    g.add_node(i, label=label)
  for u, v in edges:
    g.add_edge(u - offset, v - offset)
  return g

def nx_to_rx(g:Graph) -> PyGraph:
  a = g.nodes
  e = g.edges

  g = PyGraph()
  node_ids: List[int] = []
  for it in a:
    node_ids.append(g.add_node(it))
  for u, v in e:
    g.add_edge(node_ids[u], node_ids[v], 1.0)
  return g

def find_isomorphism(g:Graph, s:Graph) -> Tuple[int]:
  # 尝试子图同构匹配；按 label 一致判定节点等价性
  mapping = vf2pp_find_isomorphism(g, s)
  if mapping is None: return None
  return tuple(mapping[i] for i in range(len(mapping)))


def run_from_demo():
  G = make_graph(
    [4, 1, 3, 2, 1],
    [
      (1, 2),
      (2, 3),
      (3, 4),
      (1, 3),
      (2, 4),
      (4, 5),
    ]
  )
  S1 = make_graph(
    [1, 2, 4],
    [
      (1, 2),
      (2, 3),
    ]
  )
  S2 = make_graph(
    [1, 2, 3, 4],
    [
      (1, 2),
      (2, 3),
      (3, 4),
      (1, 4),
      (1, 3),
    ]
  )
  S_list = [S1, S2]

  res = []
  for i, S in enumerate(S_list, start=1):
    f = find_isomorphism(G, S)
    if f: res.append((i, f))

  print(len(res))
  for i, f in res:
    print(i, end='')
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
      print(f'[{i + 1}] ' + ' '.join(str(x + 1) for x in f))


if __name__ == '__main__':
  #run_from_demo()
  run_from_random()
