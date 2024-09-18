#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/18 

''' ↓↓↓ Copy & modified from https://github.com/6god-rail-flower-water/Subgraph-Isomorphic '''

from enum import Enum
from typing import Iterator, List, Tuple
from dataclasses import dataclass
import numpy as np

INDEX_MAX = 5000

class OpenList(Enum):
    Out = 0
    In = 1
    Other = 2

class FrameType(Enum):
    Outer = "Outer"
    Inner = "Inner"
    Unwind = "Unwind"

@dataclass
class FrameData:
    frame_type: FrameType
    nodes: List[int] = None
    open_list: OpenList = None

class StableGraph:
    def __init__(self, num_nodes:int, num_edges:int):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.nodes = {}
        self.edges = {}
        self.free_node_indices = set()  # 空闲节点索引
        self.free_edge_indices = set()  # 空闲边索引
        self.next_node_index = 0        # 下一个可用的节点索引
        self.next_edge_index = 0        # 下一个可用的边索引

    def add_node(self, value:int) -> int:
        if self.free_node_indices:
            index = self.free_node_indices.pop()
        else:
            index = self.next_node_index
            self.next_node_index += 1
        self.nodes[index] = value
        return index

    def node_bound(self) -> int:
        return self.next_node_index

    def add_edge(self, source:int, target:int) -> int:
        if self.free_edge_indices:
            index = self.free_edge_indices.pop() 
        else:
            index = self.next_edge_index
            self.next_edge_index += 1
        self.edges[index] = (source, target)
        return index

    def node_weight(self, index:int) -> int:
        return self.nodes.get(index)

    def edge_references(self) -> Iterator[List[int]]: 
        for edge_index, (source, target) in self.edges.items():
            yield source, target

    def get_degrees(self) -> List[int]:
        degree = [0] * self.num_nodes
        for edge in self.edges.values():
            degree[edge[0]] += 1
            degree[edge[1]] += 1
        return degree

    def get_neighbours(self, index:int) -> Iterator[int]:
        for source, target in self.edge_references():
            if   source == index: yield target
            elif target == index: yield source

def find_max_element(vd:List[int], i:int, conn_in:List[int], dout:List[int], conn_out:List[int], din:List[int]) -> List[int]:
    def sorting_key(node):
        return (conn_in[node], dout[node], conn_out[node], din[node], -node)
    sub_vd = vd[i:]
    max_index, max_item = max(enumerate(sub_vd), key=lambda x: sorting_key(x[1]))
    return i + max_index, max_item

def adj_mat(G:StableGraph) -> set:
    adj = set()
    for edge in G.edge_references():
        source, target = min(edge[0], edge[1]), max(edge[0], edge[1])
        adj.add((source, target))
    return adj

def has_edge(adj:set, a:int, b:int) -> int:
    return int((min(a, b), max(a, b)) in adj)

class VF2ppSorter:
    def __init__(self, G:StableGraph):
        self.G = G

    def sort(self) -> List[int]:
        n = self.G.node_bound()
        dout = self.G.get_degrees()
        din = [0] * n
        conn_in = [0] * n
        conn_out = [0] * n
        order: List[int] = []

        def process(vd:list) -> list:
            for i in range(len(vd)):
                index, item = find_max_element(vd, i, conn_in, dout, conn_out, din)
                vd[i], vd[i + index] = vd[i + index], vd[i]
                order.append(item)
                for neigh in self.G.get_neighbours(item):
                    conn_in[neigh] += 1
            return vd

        seen = np.full(n, False, dtype=bool)    # BUG: here must use ndarray rather than List[bool], why???
        def bfs_tree(root:int):
            if seen[root] is True: return

            next_level = []
            seen[root] = True
            next_level.append(root)

            while len(next_level):
                this_level = process(next_level)
                next_level = []
                for bfs_node in this_level:
                    for neighbor in self.G.get_neighbours(bfs_node):
                        if seen[neighbor] is False:
                            seen[neighbor] = True
                            next_level.append(neighbor)

        sorted_list = list(self.G.nodes)
        sorted_list.sort(key=lambda x: (dout[x], din[x], -x), reverse=True)
        for node in sorted_list:
            bfs_tree(node)
        return order

    def reorder(self) -> Tuple[StableGraph, dict]:
        order_nodes = self.sort()
        new_graph = StableGraph(self.G.num_nodes, self.G.num_edges)
        id_map = {}
        for node_index in order_nodes:
            node_data = self.G.node_weight(node_index)
            new_index = new_graph.add_node(node_data)
            id_map[node_index] = new_index
        for source, target in self.G.edge_references():
            new_graph.add_edge(id_map[source], id_map[target])
        return new_graph, {v: k for k, v in id_map.items()}

class Vf2State:
    def __init__(self, G:StableGraph):
        self.G = G
        self.c0 = self.G.num_nodes
        self.mapping = [INDEX_MAX] * self.c0
        self.out = [0] * self.c0
        self.ins = []       # wft, empty?
        self.out_size = 0
        self.ins_size = 0
        self.adjacency_matrix = adj_mat(G)
        self.generation = 0

    def is_complete(self) -> bool:
        return self.generation == len(self.mapping)

    def push_mapping(self, idx_from:int, idx_to:int):
        self.generation += 1
        s = self.generation
        self.mapping[idx_from] = idx_to

        for ix in self.G.get_neighbours(idx_from):
            if self.out[ix] == 0:
                self.out[ix] = s
                self.out_size += 1

    def pop_mapping(self, idx_from:int):
        s = self.generation
        self.generation -= 1
        self.mapping[idx_from] = INDEX_MAX

        for ix in self.G.get_neighbours(idx_from):
            if self.out[ix] == s:
                self.out[ix] = 0
                self.out_size -= 1

    def next_out_index(self, idx_from:int) -> int:
        for ix, elt in enumerate(self.out[idx_from:]):
            if elt > 0 and self.mapping[idx_from + ix] == INDEX_MAX:
                return ix

    def next_in_index(self, idx_from:int) -> int:
        for ix, elt in enumerate(self.ins[idx_from:]):
            if elt > 0 and self.mapping[idx_from + ix] == INDEX_MAX:
                return ix
    
    def next_rest_index(self, idx_from:int) -> int:
        for ix, elt in enumerate(self.mapping[idx_from:]):
            if elt == INDEX_MAX:
                return ix

class Vf2Algorithm:
    def __init__(self, g0:StableGraph, g1:StableGraph):
        self.g0 = g0
        self.g1 = g1
        self.node_map_g0 = {}
        self.node_map_g1 = {}
        self.g0, self.node_map_g0 = VF2ppSorter(g0).reorder()
        self.g1, self.node_map_g1 = VF2ppSorter(g1).reorder()

        self.st = [Vf2State(g0), Vf2State(g1)]
        self.stack: List[FrameData] = [FrameData(FrameType.Outer)]

    def mapping(self) -> dict:
        mapping = {}
        for index, val in enumerate(self.st[1].mapping):
            g0_node = self.node_map_g0[val]
            g1_node = self.node_map_g1[index]
            mapping[g0_node] = g1_node
        return mapping

    def next_candidate(self) -> tuple[int, int, OpenList]:
        to_idx = self.st[1].next_out_index(idx_from=0)
        from_idx = None
        open_list = OpenList.Out
        
        if to_idx != None:
            from_idx = self.st[0].next_out_index(idx_from=0)
            open_list = OpenList.Out
        
        if to_idx is None or from_idx is None:
            to_idx = self.st[1].next_in_index(idx_from=0)
            if to_idx != None:
                from_idx = self.st[0].next_in_index(idx_from=0)
                open_list = OpenList.In
                
        if to_idx is None or from_idx is None:
            to_idx = self.st[1].next_rest_index(idx_from=0)
            if to_idx != None:
                from_idx = self.st[0].next_rest_index(idx_from=0)
                open_list = OpenList.Other
                
        if to_idx != None and from_idx != None:
            return from_idx, to_idx, open_list
        return [None, None, None]

    def next_from_ix(self, nx:int, open_list:OpenList):
        start = nx + 1
        if open_list == OpenList.Out:
            cand0 = self.st[0].next_out_index(idx_from=start)
        elif open_list == OpenList.In:
            cand0 = self.st[0].next_in_index(idx_from=start)
        else:
            cand0 = self.st[0].next_rest_index(idx_from=start)
        if cand0 is None: return

        cand0 += start
        assert(cand0 >= start)
        return cand0

    def pop_state(self, nodes:List[int]):
        self.st[0].pop_mapping(nodes[0])
        self.st[1].pop_mapping(nodes[1])

    def push_state(self, nodes:List[int]):
        self.st[0].push_mapping(nodes[0], nodes[1])
        self.st[1].push_mapping(nodes[1], nodes[0])

    def is_feasible(self, nodes:List[int]) -> bool:
        succ_count = [0, 0]
        for j in range(2):
            for n_neigh in self.st[j].G.get_neighbours(nodes[j]):
                succ_count[j] += 1
                if j == 0: continue

                if nodes[j] != n_neigh:
                    m_neigh = self.st[j].mapping[n_neigh]
                else:
                    m_neigh = nodes[1 - j]
                if m_neigh == INDEX_MAX: continue

                # A strange expression in source code 
                if has_edge(self.st[1-j].adjacency_matrix, m_neigh, nodes[1-j]) < has_edge(self.st[j].adjacency_matrix, nodes[j], n_neigh):
                    return False

        if succ_count[0] < succ_count[1]:
            return False

        def rule(arr:str, st_idx:int):
            count = 0
            for n_neigh in self.st[st_idx].G.get_neighbours(nodes[st_idx]):
                if getattr(self.st[st_idx], arr)[n_neigh] > 0 and self.st[st_idx].mapping[n_neigh] == INDEX_MAX:
                    count += 1
            return count

        if rule("out", 0) < rule("out", 1):
            return False
        if self.st[0].G.node_weight(nodes[0]) != self.st[1].G.node_weight(nodes[1]):
            return False

        return True

    def next_vf2(self):
        if self.st[0].G.num_nodes < self.st[1].G.num_nodes or self.st[0].G.num_edges < self.st[1].G.num_edges:
            return

        while len(self.stack) > 0:
            frame = self.stack.pop()
            if frame.frame_type == FrameType.Unwind:
                nodes = frame.nodes
                ol = frame.open_list
                self.pop_state(nodes)

                if self.next_from_ix(nodes[0], ol) is None:
                    continue
                else:
                    nx = self.next_from_ix(nodes[0], ol)
                    f = FrameData(FrameType.Inner, [nx, nodes[1]], ol)
                    self.stack.append(f)
                    
            if frame.frame_type == FrameType.Outer:
                if self.next_candidate() == [None, None, None]:
                    if self.st[1].is_complete():
                        return self.mapping()
                    continue
                else:
                    nx, mx, ol = self.next_candidate()
                    f = FrameData(FrameType.Inner, [nx, mx], ol)
                    self.stack.append(f)
                    
            if frame.frame_type == FrameType.Inner:
                nodes = frame.nodes
                ol = frame.open_list
                
                if self.is_feasible(nodes):
                    self.push_state(nodes)
                    if self.st[0].out_size >= self.st[1].out_size and self.st[0].ins_size >= self.st[1].ins_size:
                        f0 = FrameData(FrameType.Unwind, nodes, ol)
                        self.stack.append(f0)
                        self.stack.append(FrameData(FrameType.Outer))
                        continue

                    self.pop_state(nodes)

                if self.next_from_ix(nodes[0], ol) is None:
                    continue
                else:
                    nx = self.next_from_ix(nodes[0], ol)
                    f = FrameData(FrameType.Inner, [nx, nodes[1]], ol)
                    self.stack.append(f)

''' ↑↑↑ Copy & modified from https://github.com/6god-rail-flower-water/Subgraph-Isomorphic '''
