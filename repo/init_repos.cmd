@ECHO OFF

PUSHD %~dp0

git clone https://github.com/RapidsAtHKUST/LIGHT

git clone https://github.com/MiviaLab/vf2lib
git clone https://github.com/MiviaLab/vf3lib
git clone https://github.com/xysmlx/VF2 VF2.xysmlx
git clone https://github.com/bookug/VF2 VF2.bookug
git clone https://github.com/yaolili/VF2 VF2.yaolili
git clone https://github.com/pfllo/VF2 VF2.pfllo
git clone https://github.com/InnoFang/subgraph-isomorphism
git clone https://github.com/kpetridis24/vf2-pp
git clone https://github.com/mjyoussef/VF2SAGE

git clone https://github.com/6god-rail-flower-water/Subgraph-Isomorphic

POPD

ECHO Done!
ECHO.

PAUSE
