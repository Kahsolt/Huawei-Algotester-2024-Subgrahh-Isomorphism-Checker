#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/19 

# 生成超~级~大的完全随机测试样例，每个模式子图都有匹配！

import random
from collections import defaultdict
from argparse import ArgumentParser
from typing import List, Tuple

from tqdm import tqdm
from rustworkx import PyGraph, graph_is_subgraph_isomorphic

from data import DATA_PATH

OUT_PATH = DATA_PATH / 'random'
OUT_PATH.mkdir(exist_ok=True)

MAX_M  = 50000    # test cases
MAX_N  = 2000     # graph V (1~2000)
MAX_K  = 12000    # graph E
MAX_Ni = 50       # subgraph V (1~50)
MAX_Ki = 250      # subgraph E
MAX_L  = 10       # label id (1~10)


# 产生一个随机主图
def rand_graph():
  n = random.randint(2*MAX_Ni, MAX_N)
  a = [random.randint(1, MAX_L) for _ in range(n)]
  e = []
  for u in range(1, n):
    for v in range(u+1, n+1):
      if random.random() < 0.01:
        e.append((u, v))  # 1 ~ n
  return a, e

# 从主图里抠一个随机子图
def rand_subgraph(a:List[int], e:List[Tuple[int, int]]):
  n_i = random.randint(5, MAX_Ni)
  adj = defaultdict(set)
  for u, v in e:
    adj[u].add(v) # 1 ~ n
    adj[v].add(u)
  # 抠图！
  V = {random.randint(1, len(a))}     # 随机起点
  while len(V) < n_i:                 # 泛洪邻域 n_i 个点
    newV = V.copy()
    for u in V:
      for v in adj[u]:
        newV.add(v)
        if len(newV) >= n_i:
          break
      if len(newV) >= n_i:
        break
    V = newV
  V = list(V)   # rewire ids to 1 ~ n
  E = []                              # 收集该邻域所有边
  for u in V:
    for v in adj[u]:
      if v not in V: continue
      if u >= v: continue
      E.append((u, v))
  # id转换！
  a_i = [a[i-1] for i in V] # 0 ~ n-1
  e_i = [(V.index(u)+1, V.index(v)+1) for u, v in E]
  if len(e_i) > MAX_Ki:
    random.shuffle(e_i)
    e_i = sorted(e_i[:MAX_Ki])
  return a_i, e_i


def VE_to_PyGraph(a, e) -> PyGraph:
  g = PyGraph()
  node_ids = []
  for it in a:
    node_ids.append(g.add_node(it))
  for u, v in e:
    g.add_edge(node_ids[u-1], node_ids[v-1], 1.0)
  return g


def run(args):
  fp = OUT_PATH / f'{args.seed}.txt'
  if fp.exists():
    print('>> file exists, skip data_gen :)')
    return

  with open(fp, 'w', encoding='ascii') as fh:
    a, e = G = rand_graph()
    fh.write(f'{len(a)} {len(e)}\n')
    for l in a:
      fh.write(f'{l} ')
    fh.write('\n')
    for u, v in e:
      fh.write(f'{u} {v}\n')

    fh.write(f'{MAX_M}\n')
    for _ in tqdm(range(MAX_M)):
      a, e = S = rand_subgraph(*G)

      Gx = VE_to_PyGraph(*G)
      Sx = VE_to_PyGraph(*S)
      assert graph_is_subgraph_isomorphic(Gx, Sx, node_matcher=(lambda x, y: x==y), induced=False)

      fh.write(f'{len(a)} {len(e)}\n')
      for l in a:
        fh.write(f'{l} ')
      fh.write('\n')
      for u, v in e:
        fh.write(f'{u} {v}\n')

  print(f'>> file saved to {fp}')


if __name__ =='__main__':
  parser = ArgumentParser()
  parser.add_argument('--seed', default=114514, type=int)
  args = parser.parse_args()

  random.seed(args.seed)
  print(f'>> use seed: {args.seed}')

  run(args)
