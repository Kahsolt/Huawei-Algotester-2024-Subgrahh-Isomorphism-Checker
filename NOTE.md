### Problem

ℹ 本题考察**无向图上的子图匹配**：给定大图 $ G $ 和一系列子图 $ S_i $，检查 $ S_i $ 是否为 $ G $ 的子图。

```
[测试样例说明]
数据范围
  - 输入样例数 `T = 10`
  - 查询样例数 `1 ≤ m ≤ 50000`
  - 大图
    - 顶点数: `2 ≤ n ≤ 2000`
    - 边数: `1 ≤ k ≤ 12000`
    - 节点编号 `1 ≤ x,y ≤ n`
    - 标签 `1 ≤ L ≤ 10`
  - 模式图
    - 顶点数: `2 ≤ n_i ≤ 50`
    - 边数: `1 ≤ k_i ≤ 250`
    - 节点编号 `1 ≤ x_i,y_i ≤ n_i`
    - 标签 `1 ≤ L_i ≤ 10`
分数: `p/m * 10^7`; `p` 为成功匹配数, `m` 为子图测试数
时间: 10s
空间: 1024MB
```


### Ranking

ℹ 排名刷新时间：2024-09-19 10:20:01

| 排名 | 团队名 | 得分 | 作品提交时间 |
| :-:| :-:| :-:| :-: |
|  1 | koooooooooooooooo   | 57866242 | 2024/09/13 |
|  2 | fhysmile            | 55749128 | 2024/09/17 |
|  3 | hw038263481         | 46117411 | 2024/09/19 |
|  4 | boom154             | 38789941 | 2024/09/18 |
|  5 | flytobug            | 23297515 | 2024/09/17 |
|  6 | hid_qdt7el8ik_gk-m4 | 11082265 | 2024/09/16 |
|  7 | kahsolt             | 6842356  | 2024/09/18 |
|  8 | hid_bfv8oy72pkarrom | 6534354  | 2024/09/13 |
|  9 | codercyh1910        | 363642   | 2024/09/02 |
| 10 | jsonzb              | 2330     | 2024/09/04 |


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
