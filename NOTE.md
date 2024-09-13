### Solution

目前 VF 系列就是世界上最好的算法，操作流程可见 [VF3 PPT](https://mivia.unisa.it/wp-content/uploads/2016/05/VF3_inBrief.pdf)，因此有两条捷径：

- 翻译移植 rustworkx 中的 `graph_vf2_mapping`
- 移植 networkx 中的 `vf2pp_isomorphism`，再手动改造以支持子图匹配

⚪ VF2 算法伪码

```python
Mapping = List[Tuple[int, int]]   # 映射关系

def VF2(m:Mapping):
  if m covers V1:   # 若当前映射集已经覆盖了待匹配子图
    return m
  P_m = ...         # 否则枚举候选映射集
  for p in P_m:
    if cons(p, m) and not cut(p, m):  # 一致且不剪枝
      m.append(p)   # 扩张当前映射集
      VF2(m)        # 递归
```

其升级版的核心思想都是更好的剪枝:

- [VF2++](https://egres.elte.hu/tr/egres-18-03.pdf)
  - 更好的匹配顺序
    - 使用 BFS 而非 DFS (注意内存开销!!)
    - 在 BFS 的每一层，将子图 `V1` 中剩余未匹配的顶点按 `(most connected with current partial mapping, most degree (uncovered neigbours) in G1, rarest uncovered labels in G2)` 排序
      - 第一位序规则要求用 BFS, DFS 语境下无意义
  - 更好的 `cut()` 规则
    - 剩余 label 还够用吗
  - 多用哈希表实现
- [VF3](https://www.iris.unisa.it/bitstream/11386/4688387/8/vf3.pdf)
  - 节点分类函数: 确保不同类节点不会被匹配 （一般取 label 匹配函数，所以这是废话）
  - 节点排序: `(most connections to already matched nodes, less P_f(u) value, more node degree)`
    - 概率定义为 `P_f(u) = 节点u所带标签出现的概率 * 节点u的度占总度数的概率`，须为每个 `V1` 节点提前计算，均假定为均匀分布
