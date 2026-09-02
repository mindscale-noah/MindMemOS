"""最简单的 Brave 搜索请求（yibuapi 中转）。

key 只从环境变量或 .env 文件读取，绝不写进代码或命令行。
用法：python brave_search.py "查询词"
"""

import os
import sys

import requests

BRAVE_URL = "https://yibuapi.com/brave/v1/web/search"
ENV_FILE = r"C:\working_projects\Memory\WildClawBench\.env"


def _load_key() -> str:
    key = os.environ.get("BRAVE_API_KEY", "")
    if key:
        return key
    with open(ENV_FILE, encoding="utf-8") as f:
        for line in f:
            if line.startswith("BRAVE_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("BRAVE_API_KEY not found in env or .env")


def brave_search(query: str, count: int = 5) -> dict:
    """GET https://yibuapi.com/brave/v1/web/search，Bearer 认证，返回原始 JSON。"""
    resp = requests.get(
        BRAVE_URL,
        params={"q": query, "count": count},
        headers={"Accept": "application/json", "Authorization": f"Bearer {_load_key()}"},
        timeout=30,
    )
    if resp.status_code != 200:
        # 打出中转层的错误信息（如 No available channel for model brave-web-search）
        raise SystemExit(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "MindMemOS memory system"
    data = brave_search(query)
    results = data.get("web", {}).get("results", [])
    print(f"{len(results)} results for: {query}")
    for r in results:
        print("-", r.get("title"), "|", r.get("url"))
