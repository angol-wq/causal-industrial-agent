@echo off
chcp 65001 >nul
title 因果增强工业智能体 — 溯因智工

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║     因果增强工业智能体 — 溯因智工                      ║
echo ║     Causal Enhanced Industrial Agent                 ║
echo ║     创智青山AI智能体创新大赛 · 技术挑战赛道            ║
echo ╚══════════════════════════════════════════════════════╝
echo.
echo  请选择运行模式:
echo.
echo    [1] 命令行演示 (合成数据)
echo    [2] 命令行演示 (真实TEP工业数据)
echo    [3] 三种场景交互式演示
echo    [4] Web可视化界面 (Streamlit)
echo    [5] 生成PPT
echo    [6] 一键运行全部演示
echo    [0] 退出
echo.
set /p choice="  请输入选项 [1-6]: "

if "%choice%"=="1" (
    echo.
    echo 🚀 运行合成数据演示...
    python run_demo.py
    pause
)
if "%choice%"=="2" (
    echo.
    echo 🚀 运行真实TEP数据演示...
    python run_tep_demo.py
    pause
)
if "%choice%"=="3" (
    echo.
    echo 🚀 运行三种场景演示...
    python usage_demo.py
    pause
)
if "%choice%"=="4" (
    echo.
    echo 🚀 启动Streamlit Web界面...
    echo 打开浏览器访问: http://localhost:8501
    echo 按 Ctrl+C 停止
    streamlit run app.py --server.port 8501
)
if "%choice%"=="5" (
    echo.
    echo 🚀 生成答辩PPT...
    python create_ppt.py
    pause
)
if "%choice%"=="6" (
    echo.
    echo 🚀 一键运行全部演示...
    echo ================================================================
    echo [1/4] 合成数据演示
    echo ================================================================
    python run_demo.py
    echo.
    echo ================================================================
    echo [2/4] 真实TEP数据演示
    echo ================================================================
    python run_tep_demo.py
    echo.
    echo ================================================================
    echo [3/4] 三种场景演示
    echo ================================================================
    python usage_demo.py
    echo.
    echo ================================================================
    echo [4/4] 全部完成！
    echo ================================================================
    pause
)
if "%choice%"=="0" exit

echo.
echo 按任意键重新选择...
pause >nul
cls
%0
