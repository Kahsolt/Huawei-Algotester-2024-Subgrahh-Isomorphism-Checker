from __future__ import annotations

from sys import stdin
from typing import List, Tuple


class Graph:

  def __init__(self):
    n, m = [int(x) for x in stdin.readline().split()]
    self.n: int = n
    self.a: List[int] = [int(x) for x in stdin.readline().split()]
    self.e: List[Tuple[int, int]] = [tuple(int(x) for x in stdin.readline().split()) for _ in range(m)]

  def find_isomorphism(self, g:Graph, f:List[int]) -> bool:
    # check if subgraph isomorphism exists
    # if found return true and fill f with mapping according to the statement
    # if not found return false
    return False


if __name__ == '__main__':
  g = Graph()
  k = int(stdin.readline())
  res: List[Tuple[int]] = []
  for i in range(k):
    s = Graph()
    f: List[int] = []
    if s.find_isomorphism(g, f):
      res.append((i, f))

  print(len(res))
  for i, f in res:
    print(i + 1, end='')
    for x in f:
      print(f' {x}', end='')
    print()
