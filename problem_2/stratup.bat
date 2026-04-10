@echo off
REM 进入后端目录并启动服务器
cd /d "backend"
start "Backend Server" python server.py
REM 返回上级目录，再进入前端目录并启动HTTP服务
cd /d "..\frontend"
start "Frontend Server" python -m http.server 8080
REM 自动打开浏览器
start http://localhost:8080
echo 程序启动完成！请确保Python已安装且依赖已安装(pip install -r requirements.txt)。