# 使用轻量级的 Python 3.10 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统基础依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目所有代码到容器内
COPY . .

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "保单优化系统.py", "--server.port=8501", "--server.address=0.0.0.0"]