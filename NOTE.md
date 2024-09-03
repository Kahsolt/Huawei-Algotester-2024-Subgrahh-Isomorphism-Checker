### Solution

目前 VF2 就是世界上最好的算法，因此有两条捷径：

- 翻译移植 rustworkx 中的 `graph_vf2_mapping`
- 移植 networkx 中的 `vf2pp_isomorphism`，加上子图筛选前处理
