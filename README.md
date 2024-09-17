# Huawei-Algotester-2024-Subgraph-Isomorphism-Checker

    Contest solution for 华为算法精英实战营第十二期-子图召回

----

Contest page: https://competition.huaweicloud.com/information/1000042127/circumstance  

ℹ 本题考察**无向图上的子图匹配**：给定大图 $ G $ 和一系列子图 $ S_i $，检查 $ S_i $ 是否为 $ G $ 的子图。


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
