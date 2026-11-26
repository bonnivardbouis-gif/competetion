"""命令行入口：跑一次拉曼光谱甄别查询。

用法：
    python main.py "请识别这条拉曼光谱属于哪种已知物质"
    python main.py --example 3 --top-k 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sciagent.app import build_raman_agent

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def load_example_queries():
    data = np.load(EXAMPLES_DIR / "query_spectra.npz", allow_pickle=False)
    return data["spectra"], [str(x) for x in data["labels"]], data["wavenumbers"]


def main() -> None:
    parser = argparse.ArgumentParser(description="科研数据甄别 Agent（拉曼示例）")
    parser.add_argument(
        "query",
        nargs="?",
        default="请识别这条拉曼光谱属于哪种已知物质，给出候选、证据与置信度；如果证据不足请明确拒识。",
        help="要问的问题",
    )
    parser.add_argument("--example", type=int, default=0, help="使用哪条示例光谱（0 起）")
    parser.add_argument("--top-k", type=int, default=3, help="识别候选数量")
    args = parser.parse_args()

    spectra, labels, wavenumbers = load_example_queries()
    idx = max(0, min(args.example, len(spectra) - 1))

    agent, _raman = build_raman_agent(spectrum=spectra[idx], wavenumbers=wavenumbers)

    print(f"[系统] 已加载示例光谱 #{idx}")
    print(f"[系统] 真实标签：{labels[idx]}（仅作对照，不会告诉模型）")
    print("[系统] 正在调用 LLM 分析 …\n")

    answer = agent.run(args.query)

    print("=" * 50)
    print("Agent 回答：")
    print("=" * 50)
    print(answer)
    print("\n" + "=" * 50)
    print(f"真实标签：{labels[idx]}")
    print("=" * 50)


if __name__ == "__main__":
    main()
