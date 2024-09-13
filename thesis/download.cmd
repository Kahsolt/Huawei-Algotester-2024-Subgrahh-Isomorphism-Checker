@ECHO OFF

PUSHD %~dp0

REM A (Sub)Graph Isomorphism Algorithm for Matching Large Graphs
wget -nc https://www.researchgate.net/profile/Mario-Vento/publication/3193784_A_SubGraph_Isomorphism_Algorithm_for_Matching_Large_Graphs/links/0c960516872cd61d3f000000/A-SubGraph-Isomorphism-Algorithm-for-Matching-Large-Graphs.pdf

REM HALLENGING MEMORY AND TIME COMPLEXITY OF SUBGRAPH ISOMORPHISM PROBLEM WITH VF3
wget -nc https://www.iris.unisa.it/bitstream/11386/4688387/8/vf3.pdf
REM (PPT) HALLENGING MEMORY AND TIME COMPLEXITY OF SUBGRAPH ISOMORPHISM PROBLEM WITH VF3
wget -nc https://mivia.unisa.it/wp-content/uploads/2016/05/VF3_inBrief.pdf

REM VF2++ — An Improved Subgraph Isomorphism Algorithm
wget -nc https://egres.elte.hu/tr/egres-18-03.pdf

REM Deep Analysis on Subgraph Isomorphism
wget -nc https://arxiv.org/pdf/2012.06802.pdf

REM Efficient Parallel Subgraph Enumeration on a Single Machine
wget -nc https://shixuansun.github.io/files/ICDE19-LIGHT.pdf

POPD

ECHO Done!
ECHO.

PAUSE
