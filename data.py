#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/03 

# get & cvt graphDB data

import sys
import random
from pathlib import Path
from typing import Tuple, List, Dict

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'
DATA_GRAPHDB_PATH = DATA_PATH / 'graphDB'
DATA_GRAPHDB_CVT_PATH = DATA_PATH / 'graphDB_cvt'

Vertex = int
Label = int
Edge = Tuple[int, int]
Graph = Tuple[List[Label], List[Edge]]

TARGET_GRAPHS: List[Graph] = []
QUERY_GRAPHS: Dict[int, List[Graph]] = {}
TARGET_GRAPHS_LEN = 10000
QUERY_GRAPHS_LEN  = 1000
QUERY_GRAPH_EDGES = [4, 8, 12, 16, 20, 24]


def _load_graphDB_file(fp:Path) -> List[Graph]:
  graphs: List[Graph] = []
  tmp: List[str] = []

  def _cvt():
    nonlocal tmp, graphs
    lables: List[Label] = []
    edges: List[Edge] = []
    for line in tmp:
      if line.startswith('v'):
        lables.append(int(line.split(' ')[2]))
      elif line.startswith('e'):
        edges.append(tuple(int(e) + 1 for e in line.split(' ')[1:3]))   # offset by one for storage
      else:
        raise ValueError(f'>> Error parse line: {line}')
    graphs.append((lables, edges))

  with open(fp, encoding='utf-8') as fh:
    lines = [e.strip() for e in fh.read().strip().split('\n') if e.strip()]
    for line in lines[1:]:
      if line == 't # -1': break
      if line.startswith('t'):
        _cvt()
        tmp.clear()
      else:
        tmp.append(line)
    if tmp: _cvt()
  return graphs

def _cache_graphDB_data(n_edges=None):
  global TARGET_GRAPHS, QUERY_GRAPHS
  assert n_edges in [None] + QUERY_GRAPH_EDGES
  if n_edges is None:
    if not TARGET_GRAPHS:
      TARGET_GRAPHS = _load_graphDB_file(DATA_GRAPHDB_PATH / 'dataset.txt')
      assert len(TARGET_GRAPHS) == TARGET_GRAPHS_LEN
  else:
    if n_edges not in QUERY_GRAPHS:
      QUERY_GRAPHS[n_edges] = _load_graphDB_file(DATA_GRAPHDB_PATH / f'Q{n_edges}.txt')
      assert len(QUERY_GRAPHS[n_edges]) == QUERY_GRAPHS_LEN

def get_query_pair(target:int=None, n_edges:int=None) -> Tuple[Graph, List[Graph]]:
  target = random.randrange(TARGET_GRAPHS_LEN) if target is None else target
  assert 0 <= target < TARGET_GRAPHS_LEN
  _cache_graphDB_data()
  g = TARGET_GRAPHS[target]
  n_edges = n_edges or random.choice(QUERY_GRAPH_EDGES)
  _cache_graphDB_data(n_edges)
  s_list = QUERY_GRAPHS[n_edges]
  return g, s_list


# 从 graphDB 转出测试样例: 选 1 张主图搭配所有的 6000 张模式子图 
def cvt_graphDB(gids):
  DATA_GRAPHDB_CVT_PATH.mkdir(exist_ok=True)
  for gid in gids:
    fp = DATA_GRAPHDB_CVT_PATH / f'{gid}.txt'
    if fp.exists():
      print(f'>> file {fp.relative_to(BASE_PATH)} exists, skip data_cvt :)')
      continue

    print(f'>> file saved to {fp}')
    with open(fp, 'w', encoding='ascii') as fh:
      # graph
      G, S_list = get_query_pair(gid, None)

      a, e = G
      fh.write(f'{len(a)} {len(e)}\n')
      for l in a:
        fh.write(f'{min(max(l, 1), 10)} ')  # label refix
      fh.write('\n')
      for u, v in e:
        fh.write(f'{u+1} {v+1}\n')          # node_id refix

      # subgraph
      fh.write(f'{len(QUERY_GRAPH_EDGES) * len(S_list)}\n')
      for n_edge in QUERY_GRAPH_EDGES:
        _, S_list = get_query_pair(None, n_edge)

        for a, e in S_list:
          fh.write(f'{len(a)} {len(e)}\n')
          for l in a:
            fh.write(f'{min(max(l, 1), 10)} ')  # label refix
          fh.write('\n')
          for u, v in e:
            fh.write(f'{u+1} {v+1}\n')          # node_id refix


if __name__ == '__main__':
  # 若不指定主图 id，则默认生成前 10 张
  gids = [int(sys.argv[1])] if len(sys.argv) >= 2 else range(10)

  cvt_graphDB(gids)
