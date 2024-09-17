# Huawei-Algotester-2024-Subgraph-Isomorphism-Checker

    Contest solution for 华为算法精英实战营第十二期-子图召回

----

Contest page: https://competition.huaweicloud.com/information/1000042127/circumstance  

ℹ 本题考察**无向图上的子图匹配**：给定大图 $ G $ 和一系列子图 $ S_i $，检查 $ S_i $ 是否为 $ G $ 的子图。


### Results

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

⚠ Python 严重超时，不得不上 C++ (?)

| lang | score | comment |
| :-: | :-: | :-: |
| python |  11886 | rt = 0.1s |
| python | 135261 | rt = 1s |
| python | 326390/391708/381910 | rt = 4s*90% |
| python | 377640/398963/403736 | rt = 4s*95% |
| python | 359803/415042/415474 | rt = 4s*98% |
| python | 423046/414288/418056 | rt = 4s*99% |
| python | 422084 | rt = 4s*101% |
| python | 379147 | rt = 4s*105% |
| python | 455902 | rt = 4s*110% |
| python | 546308 | rt = 4s*130% |
| python | 614138 | rt = 6s |
| python | 1290512 | rt = 12s |
| python | 4235055 | rt = 40s |
| python | 5587740 | rt = 59s |
| python | 5723904 | rt = 59.5s |
| python | 5741489 | rt = 60s |


### Quickstart

⚪ run

- `python main.py`

⚪ run benchmark

- `pip install -r requirements_dev.txt`
- `python main_benchmark.py`


#### refenrence

- thesis
  - networkx-VF2算法: https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html
  - rustworkx-is_subgraph_isomorphic: https://www.rustworkx.org/apiref/rustworkx.is_subgraph_isomorphic.html#rustworkx.is_subgraph_isomorphic
  - LIGHT: https://ieeexplore.ieee.org/abstract/document/8731613
  - modified VF3 & BoostISO: https://arxiv.org/abs/2012.06802
- libs
  - rustworkx: https://github.com/Qiskit/rustworkx
  - networkx: https://github.com/networkx/networkx
- repos
  - (C) https://github.com/MiviaLab/vf2lib
  - (C) https://github.com/MiviaLab/vf3lib
    - https://mivia.unisa.it/datasets/graph-database/vf3-library/vf3-in-action/
    - https://mivia.unisa.it/wp-content/uploads/2016/05/VF3_inBrief.pdf
  - (C++) https://github.com/xysmlx/VF2
    - report: https://github.com/xysmlx/VF2/blob/master/Report/VF2Report.pdf
  - (C++) https://github.com/bookug/VF2
  - (Python) https://github.com/yaolili/VF2
  - (Java) https://github.com/pfllo/VF2
  - (Java) https://github.com/InnoFang/subgraph-isomorphism
  - (Python) https://github.com/kpetridis24/vf2-pp
  - (Python) https://github.com/mjyoussef/VF2SAGE
  - (Rust2Python) https://github.com/6god-rail-flower-water/Subgraph-Isomorphic
- C++ 快读快写 & 火车头优化
  - https://www.acwing.com/blog/content/36914/
  - https://www.cnblogs.com/fusiwei/p/11457143.html
  - https://www.cnblogs.com/wjnclln/p/11582220.html
  - https://blog.csdn.net/A_zjzj/article/details/105496376
  - https://blog.csdn.net/yudui666/article/details/132144670
  - https://blog.csdn.net/qssssss79/article/details/126017455
  - https://blog.csdn.net/m0_54615144/article/details/126141161

----
by Armit
2024/09/03
