#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/03 

import random
from pathlib import Path
from typing import Tuple, List, Dict

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'
DATA_GRAPHDB_PATH = DATA_PATH / 'graphDB'

Vertex = int
Label = int
Edge = Tuple[int, int]
Graph = Tuple[List[Label], List[Edge]]

TARGET_GRAPHS: List[Graph] = []
QUERY_GRAPHS: Dict[int, List[Graph]] = {}
TARGET_GRAPHS_LEN = 10000
QUERY_GRAPHS_LEN  = 1000


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
        edges.append(tuple(int(e) for e in line.split(' ')[1:3]))
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
  assert n_edges in [None, 4, 8, 12, 16, 20]
  if n_edges is None:
    if not TARGET_GRAPHS:
      TARGET_GRAPHS = _load_graphDB_file(DATA_GRAPHDB_PATH / 'dataset.txt')
      assert len(TARGET_GRAPHS) == TARGET_GRAPHS_LEN
  else:
    if n_edges not in QUERY_GRAPHS:
      QUERY_GRAPHS[n_edges] = _load_graphDB_file(DATA_GRAPHDB_PATH / f'Q{n_edges}.txt')
      assert len(QUERY_GRAPHS[n_edges]) == QUERY_GRAPHS_LEN

def get_query_pair(target:int=None, n_edges:int=None) -> Tuple[Graph, List[Graph]]:
  target = target or random.randrange(TARGET_GRAPHS_LEN)
  assert 0 <= target < TARGET_GRAPHS_LEN
  _cache_graphDB_data()
  g = TARGET_GRAPHS[target]
  n_edges = n_edges or random.choice([4, 8, 12, 16, 20])
  _cache_graphDB_data(n_edges)
  s_list = QUERY_GRAPHS[n_edges]
  return g, s_list


if __name__ == '__main__':
  g, s_list = get_query_pair()
  print('g:', g)
  print(f's_list({len(s_list)}):', s_list[0])
