@REM run compare using data/graphDB_cvt
@ECHO OFF

IF "%1"=="" EXIT /B

IF NOT EXIST main.exe CALL make.cmd
IF NOT EXIST tmp MKDIR tmp

python data.py %1

python main_rustworkx.py < "data\graphDB_cvt\%1.txt" > tmp\out_rx.txt
head -n 1 tmp\out_rx.txt
tail -n 1 tmp\out_rx.txt

python main.py < "data\graphDB_cvt\%1.txt" > tmp\out_py.txt
head -n 1 tmp\out_py.txt
tail -n 1 tmp\out_py.txt

main.exe < "data\graphDB_cvt\%1.txt" > tmp\out_cpp.txt
head -n 1 tmp\out_cpp.txt
tail -n 1 tmp\out_cpp.txt
