#!/usr/bin/env bash
# SkillsBench 测评启动 wrapper —— 依赖走国内源，解决容器内 verifier 联网超时
set -euo pipefail

# PyPI 换源（test.sh 里 uvx 装 pytest / pytest-json-ctrf / python-docx 用），已实测生效
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# uv 二进制换源（test.sh 里 install.sh 下载 uv-x86_64-unknown-linux-gnu.tar.gz 用）
# 未设置时仍走 github.com/astral-sh/uv/releases —— 在国内网络会极慢（当前唯一剩余阻塞）
# UV_DOWNLOAD_URL 需指向一个可直连的镜像，或走宿主机本地 HTTP 服务（见下）
# export UV_DOWNLOAD_URL="http://172.17.0.1:PORT/uv/0.9.7"

cd /home/guo/project/mindmemos-skill
exec .venv/bin/mindmemos-eval skill test \
  --config config/mindmemos_eval/env_test/dev.yaml \
  "$@"