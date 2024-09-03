import sys


class Graph:

  def read(self):
    n, m = [int(x) for x in sys.stdin.readline().split()]
    self.n = n
    self.a = [int(x) - 1 for x in sys.stdin.readline().split()]
    self.e = []
    for _ in range(m):
      x, y = [int(x) for x in sys.stdin.readline().split()]
      self.e += [(x - 1, y - 1)]
    return self

  def find_isomorphism(self, g, f):
    # check if subgraph isomorphism exists
    # if found return true and fill f with mapping according to the statement
    # if not found return false
    return False


if __name__ == '__main__':
  g = Graph().read()
  k = int(sys.stdin.readline())
  res = []
  for i in range(k):
    s = Graph().read()
    f = []
    if s.find_isomorphism(g, f):
      res += [(i, f)]

  print(len(res))
  for i, f in res:
    print(i + 1, end='')
    for x in f:
      print(f' {x + 1}', end='')
    print()
