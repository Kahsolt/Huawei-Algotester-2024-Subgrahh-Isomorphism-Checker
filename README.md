# Huawei-Algotester-2024-Subgraph-Isomorphism-Checker

    Contest solution for 华为算法精英实战营第十二期-子图召回

----

Contest page: https://competition.huaweicloud.com/information/1000042127/circumstance  

ℹ 本题考察**无向图上的子图匹配**：给定大图 $ G $ 和一系列子图 $ S_i $，检查 $ S_i $ 是否为 $ G $ 的子图。  
ℹ 解决方案: 基于 `networkx.vf2pp` 改造以支持子图匹配 🎉  


### Results

⚠ Python 严重超时，不得不上 C++ 😈

| lang | score | comment |
| :-: | :-: | :-: |
| python |  7270649 | rt = 59s |
| python |  7288572 | rt = 59.5s |
| python |  7288572 | rt = 59.9s |
| python | 15913500 | rt = 59.9s (`ts_query=TTL//k`), order=naive |
| python | 13280564 | rt = 59.9s (`ts_query=TTL//k`), order=degree |
| python | 13336066 | rt = 59.9s (`ts_query=TTL//k`), order=1st_only |
| python | 20681433 | rt = 59.9s (`ts_query=TTL//k`), order=vf2pp_no_conn |
| python | 19914172 | rt = 59.9s (`ts_query=TTL//k*0.95`) |
| python | 20497856 | rt = 59.9s (`ts_query=TTL//k`) |
| python | 17373889 | rt = 59.9s (`ts_query=TTL//k*2`) |
| cpp    |    36629 | rt = 45s |
| cpp    |    27305 | rt = 30s |


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
