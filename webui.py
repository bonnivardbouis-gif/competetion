# -*- coding: utf-8 -*-
"""拉曼光谱识别助手 —— 简洁 WebUI。

流程（全自动、以 Agent 为核心）：
  ① 侧边栏输入 API Key（必填，匹配由 Agent 完成）
  ② 上传光谱 CSV（完整谱或部分区间），或选内置示例
  ③ 点击「开始识别」→ Agent（大模型 function-calling）亲自调用匹配工具，
     返回 Agent 的分析结果：
     - 本地匹配置信度为「高」→ 输出 Agent 的本地匹配结论；
     - 置信度达不到「高」→ 不输出本地结论，自动联网多源检索并输出结果。

启动：.venv\\Scripts\\python.exe -m streamlit run webui.py
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sciagent.config import PROJECT_ROOT
from sciagent.knowledge import KnowledgeBase
from sciagent.preprocess import savgol_smooth, snv
from sciagent.tools_raman import RamanTools

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
EXAMPLES_NPZ = PROJECT_ROOT / "examples" / "query_spectra.npz"

st.set_page_config(page_title="拉曼光谱识别助手", page_icon="🧪", layout="wide")


# --------------------------------------------------------------------------- #
# 资源加载（缓存，只加载一次）
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_engine():
    kb = KnowledgeBase(KNOWLEDGE_DIR / "raman_reference.json")
    raman = RamanTools(KNOWLEDGE_DIR / "raman_centroids.npz", kb)
    return raman, kb


@st.cache_data(show_spinner=False)
def load_example_data():
    d = np.load(EXAMPLES_NPZ, allow_pickle=False)
    return d["wavenumbers"], d["spectra"], [str(x) for x in d["labels"]]


# --------------------------------------------------------------------------- #
# 数据解析
# --------------------------------------------------------------------------- #
def _row_is_numeric(line: str) -> bool:
    toks = [t for t in line.replace(";", ",").replace("\t", ",").split(",") if t.strip()]
    if not toks:
        return False
    try:
        for t in toks[:2]:
            float(t)
        return True
    except ValueError:
        return False


def parse_spectrum_csv(file) -> tuple[np.ndarray | None, np.ndarray]:
    """读取 CSV/文本，返回 (波数或 None, 强度)。两列=波数+强度，单列=仅强度。"""
    raw = file.getvalue()
    text = raw.decode("utf-8", errors="ignore")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    has_header = not _row_is_numeric(first)
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", header=0 if has_header else None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    arr = df.values.astype(float)
    if arr.size == 0:
        raise ValueError("文件里没有可用的数值列")
    if arr.ndim == 1 or arr.shape[1] == 1:
        return None, arr.ravel()
    return arr[:, 0], arr[:, 1]


# --------------------------------------------------------------------------- #
# 识别与绘图
# --------------------------------------------------------------------------- #
def find_peaks_local(wav, y, min_distance: int = 5):
    """在部分区间谱上找峰（用 scipy 局部极大值）。"""
    from scipy.signal import find_peaks

    y = np.asarray(y, dtype=float)
    thr = float(np.percentile(np.abs(y), 90))
    idx, _ = find_peaks(y, height=thr, distance=min_distance)
    peaks = [{"position_cm1": float(wav[i]), "intensity_snv": float(y[i])} for i in idx]
    return {"n_peaks": len(peaks), "peaks": peaks}


def run_identify(raman, spec, wav, top_k: int = 5):
    """本地数值匹配：自动判断完整谱 / 部分区间，返回绘图数据与结果（兜底/证据用）。"""
    spec = np.asarray(spec, dtype=float)
    wav = np.asarray(wav, dtype=float) if wav is not None else None
    full_len = len(raman.wavenumbers)

    is_partial = wav is not None and spec.size < full_len

    if is_partial:
        result = raman.identify_partial(wav, spec, top_k)
        if result.get("peak_based"):
            # 少量峰值数据：绘图用散点（不连线），峰就是这些点本身
            peaks = {
                "n_peaks": int(spec.size),
                "peaks": [
                    {"position_cm1": float(wav[i]), "intensity_snv": float(spec[i])}
                    for i in range(spec.size)
                ],
            }
            xp, xo, yo = wav, wav, spec
            yp = spec
        else:
            yp = snv(savgol_smooth(spec))
            peaks = find_peaks_local(wav, yp)
            xp, xo, yo = wav, wav, spec
    else:
        raman.set_query(spec, wav)
        result = raman.identify(top_k=top_k)
        peaks = raman.find_peaks()
        xp = raman.wavenumbers
        yp = raman._preprocess_query()
        if wav is not None:
            xo, yo = wav, spec
        else:
            xo = raman.wavenumbers if spec.size == full_len else np.arange(spec.size)
            yo = spec
    return result, peaks, xp, yp, xo, yo, is_partial


def _agent_query(is_partial: bool) -> str:
    if is_partial:
        return (
            "请对这条部分区间拉曼光谱做识别：先调用 raman_identify_partial 得到候选，"
            "再对 Top 候选调用 raman_retrieve 查询分子式/类别/用途等详情，"
            "最后给出结论、证据与不确定性提示。"
        )
    return (
        "请对这条拉曼光谱做识别：先调用 raman_identify 得到候选，"
        "再对 Top 候选调用 raman_retrieve 查询分子式/类别/用途等详情，"
        "必要时调用 raman_find_peaks 看主要峰位，最后给出结论、证据与不确定性提示。"
    )


def _extract_identify_result(agent, is_partial: bool):
    """从 Agent 的工具调用轨迹里取回匹配工具的数值结果（Agent 亲自匹配的证据）。"""
    names = ("raman_identify_partial",) if is_partial else ("raman_identify",)
    for t in getattr(agent, "last_trace", []):
        r = t.get("result")
        if t.get("tool") in names and isinstance(r, dict) and "top_k_candidates" in r:
            return r
    return None


def run_agent_identify(api_key, base_url, model, raman, spec, wav, is_partial, top_k):
    """用 Agent（大模型 function-calling）完成匹配，返回 (answer, identify_result)。"""
    from sciagent.app import build_raman_agent

    agent, _ = build_raman_agent(
        spec, wav,
        llm_overrides={"api_key": api_key, "base_url": base_url, "model": model},
    )
    answer = agent.run(_agent_query(is_partial))

    identify_result = _extract_identify_result(agent, is_partial)
    if identify_result is None:
        # 兜底：LLM 未调用匹配工具时，用本地数值匹配，保证界面仍有数值证据
        if is_partial:
            identify_result = raman.identify_partial(wav, spec, top_k)
        else:
            raman.set_query(spec, wav)
            identify_result = raman.identify(top_k=top_k)
    return answer, identify_result


def make_figure(xo, yo, xp, processed, peaks, scatter: bool = False) -> go.Figure:
    fig = go.Figure()
    if scatter:
        # 少量峰值数据：直接画散点，不连线（避免假连接离散峰）
        fig.add_trace(
            go.Scatter(
                x=xo, y=yo, mode="markers", name="峰值",
                marker=dict(color="#1f77b4", size=10, symbol="diamond"), yaxis="y",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(x=xo, y=yo, name="原始谱", line=dict(color="#b0b0b0", width=1), yaxis="y")
        )
        fig.add_trace(
            go.Scatter(x=xp, y=processed, name="预处理后", line=dict(color="#1f77b4", width=1.6), yaxis="y2")
        )
        if peaks.get("peaks"):
            px = [p["position_cm1"] for p in peaks["peaks"]]
            py = [p["intensity_snv"] for p in peaks["peaks"]]
            fig.add_trace(
                go.Scatter(
                    x=px, y=py, mode="markers", name="峰位",
                    marker=dict(color="#d62728", size=7, symbol="triangle-down"), yaxis="y2",
                )
            )
    fig.update_layout(
        height=430,
        margin=dict(l=50, r=50, t=40, b=40),
        xaxis_title="波数 (cm⁻¹)",
        yaxis=dict(title="原始强度", side="left"),
        yaxis2=dict(title="处理后 (SNV)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def confidence_label(result):
    if result.get("peak_based"):
        top1 = result["top_k_candidates"][0] if result["top_k_candidates"] else {}
        score = float(result.get("best_peak_score", 0))
        cov = float(top1.get("peak_coverage", 0))
        if score >= 0.7 and cov >= 0.5:
            return "高", "#2e7d32", "峰位高度吻合，且与第二名区分明显。"
        if score >= 0.4:
            return "中", "#f57c00", "峰位部分吻合，建议结合更多峰复核。"
        return "较低", "#f57c00", "峰位匹配度偏低，请谨慎解读。"
    if result.get("partial"):
        cos = float(result.get("best_cosine", 0))
        margin = float(result.get("margin_top1_top2", 0))
        if cos >= 0.90 and margin >= 0.02:
            return "高", "#2e7d32", "与库内物质在重叠区间高度吻合，且领先第二名明显。"
        if cos >= 0.75:
            return "中", "#f57c00", "有一定把握，但部分区间信息有限，建议复核。"
        return "较低", "#f57c00", "匹配度偏低，部分区间不足以唯一判定，请谨慎解读。"
    if not result.get("is_known"):
        return "未知 / 证据不足", "#6c757d", "未能在库内找到高置信度匹配，可能是库外未知样品。"
    top = result["top_k_candidates"][0]
    cos = float(top["cosine_similarity"])
    if cos >= 0.95:
        return "高", "#2e7d32", "与库内物质高度吻合。"
    if cos >= 0.85:
        return "中", "#f57c00", "有一定把握，建议人工复核。"
    return "较低", "#f57c00", "匹配度偏低，请谨慎解读。"


def candidates_df(kb, result) -> pd.DataFrame:
    rows = []
    peak_based = bool(result.get("peak_based"))
    for i, c in enumerate(result["top_k_candidates"], 1):
        e = kb.get(c["compound"]) or {}
        role = e.get("role") or ""
        zh = e.get("zh") or ""
        if peak_based:
            rows.append(
                {
                    "排名": i,
                    "化合物": c["compound"],
                    "分子式": e.get("formula", ""),
                    "类别/用途": f"{role} / {zh}".strip(" /"),
                    "峰位得分": c.get("peak_match_score", ""),
                    "峰位覆盖率": c.get("peak_coverage", ""),
                }
            )
        else:
            rows.append(
                {
                    "排名": i,
                    "化合物": c["compound"],
                    "分子式": e.get("formula", ""),
                    "类别/用途": f"{role} / {zh}".strip(" /"),
                    "余弦相似度": c["cosine_similarity"],
                    "z 分数": c.get("z_score", ""),
                }
            )
    return pd.DataFrame(rows)


def _open_search(api_key, base_url, model, raman, result, peaks):
    """执行联网检索，返回结果 dict。"""
    from sciagent.open_search import open_identify

    wrange = result.get("wavenumber_range_cm1")
    if wrange is None:
        wrange = [float(raman.wavenumbers[0]), float(raman.wavenumbers[-1])]
    return open_identify(
        api_key, base_url, model,
        peaks["peaks"], wrange, result.get("top_k_candidates", []),
    )


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
def main():
    raman, kb = load_engine()

    st.title("拉曼光谱识别助手")
    st.caption("上传一条拉曼光谱，Agent 自动识别；本地置信度不足时，自动联网检索可能的物质。")

    c1, c2, c3 = st.columns(3)
    c1.markdown("**① 输入 Key**  \n侧边栏填 API Key（必填）")
    c2.markdown("**② 导入数据**  \n上传 CSV 或选内置示例")
    c3.markdown("**③ 识别并导出**  \nAgent 分析，下载结果")
    st.markdown("---")

    # ------------------------- 侧边栏：Key ------------------------- #
    with st.sidebar:
        st.header("① 输入 API Key")
        api_key = st.text_input(
            "API Key", type="password",
            help="必填：每次匹配都由 Agent（大模型）完成，需要 Key。",
        )
        base_url = st.text_input("接口地址", value="https://api.deepseek.com")
        model = st.text_input("模型名", value="deepseek-flash")
        st.caption("DeepSeek 官方：base_url=https://api.deepseek.com，model=deepseek-flash 或 deepseek-v4-pro。")

    # ------------------------- 数据导入 ------------------------- #
    st.subheader("② 导入光谱数据")
    if "spectrum" not in st.session_state:
        st.session_state.spectrum = None

    tab_file, tab_example = st.tabs(["上传 CSV", "选内置示例"])

    with tab_file:
        up = st.file_uploader(
            "上传光谱文件（CSV / 文本；两列=波数,强度，或单列=强度）",
            type=["csv", "txt", "dat", "text"],
        )
        if up is not None:
            try:
                wav, spec = parse_spectrum_csv(up)
                st.session_state.spectrum = (wav, spec, up.name)
                st.session_state.pop("result", None)
                if wav is not None:
                    st.success(f"已读取：{len(spec)} 个数据点，波数 {wav.min():.0f} ~ {wav.max():.0f} cm⁻¹")
                else:
                    st.success(f"已读取：{len(spec)} 个数据点（单列强度）")
            except Exception as e:
                st.error(f"读取失败：{e}")

    with tab_example:
        ex_wav, ex_spec, ex_labels = load_example_data()
        names = [f"{i} - {lab}" for i, lab in enumerate(ex_labels)]
        choice = st.selectbox(
            "选择示例光谱", names,
            help="0~5 是真实化合物，6 是合成噪声（用于演示低置信度联网检索）。",
        )
        if st.button("使用这条示例"):
            idx = names.index(choice)
            st.session_state.spectrum = (ex_wav, ex_spec[idx], choice)
            st.session_state.pop("result", None)
            st.success(f"已载入示例：{choice}")

    # ------------------------- 识别 ------------------------- #
    st.subheader("③ 识别与导出")
    data = st.session_state.spectrum
    if data is None:
        st.info("请先在上方导入光谱数据（上传 CSV 或选内置示例）。")
        return

    wav, spec, label = data
    st.caption(f"当前数据：{label} · {len(spec)} 个数据点")

    top_k = st.slider("候选数量", 1, 10, 5)

    if st.button("开始识别", type="primary"):
        if not api_key:
            st.error("Agent 匹配需要 API Key，请先在左侧填写。")
            st.session_state.result = None
        else:
            with st.spinner("Agent 正在分析…"):
                try:
                    # 本地数值匹配：用于绘图数据 + 兜底证据
                    result_local, peaks, xp, yp, xo, yo, is_partial = run_identify(raman, spec, wav, top_k)
                    # Agent（大模型 function-calling）亲自匹配 + 分析
                    answer, identify_result = run_agent_identify(
                        api_key, base_url, model, raman, spec, wav, is_partial, top_k
                    )
                    st.session_state.answer = answer
                    st.session_state.result = identify_result
                    st.session_state.peaks = peaks
                    st.session_state.fig = make_figure(
                        xo, yo, xp, yp, peaks,
                        scatter=bool(identify_result.get("peak_based")),
                    )
                    st.session_state.is_partial = is_partial
                    st.session_state.open_result = None  # 重置联网结果
                except Exception as e:
                    st.session_state.result = None
                    st.error(f"识别失败：{e}")

    result = st.session_state.get("result")
    if not result:
        return

    answer = st.session_state.get("answer", "")
    peaks = st.session_state.peaks
    fig = st.session_state.fig
    is_partial = st.session_state.get("is_partial", False)
    level, color, hint = confidence_label(result)

    # ---- ① Agent 分析结果（核心，总是展示） ---- #
    st.markdown("**Agent 分析结果**")
    if answer.strip():
        st.markdown(answer)
    else:
        st.warning("Agent 未返回文本结论（可能只做了工具调用）。数值证据见下方。")

    st.markdown("---")

    # ---- ② 数值证据（置信度 + 图 + 候选） ---- #
    top1 = result["top_k_candidates"][0]
    if result.get("peak_based"):
        n_peaks = peaks.get("n_peaks", len(peaks.get("peaks", [])))
        st.info(f"检测到少量峰值数据：已用 {n_peaks} 个峰位做峰位匹配（不依赖连续谱插值）。")
    elif is_partial:
        wmin, wmax = result["wavenumber_range_cm1"]
        st.info(f"检测到部分光谱：波数 {wmin:.0f} ~ {wmax:.0f} cm⁻¹，已按该区间匹配。")

    if result.get("peak_based"):
        metric = f"（峰位得分 {top1.get('peak_match_score')}，覆盖率 {top1.get('peak_coverage')}）"
    else:
        metric = f"（余弦相似度 {top1['cosine_similarity']}）"

    st.markdown(
        f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
        f"background:{color};margin-right:8px;'></span>"
        f"置信度：<b>{level}</b>　·　最匹配：<b>{top1['compound']}</b>"
        f"{metric}",
        unsafe_allow_html=True,
    )
    st.write(hint)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**候选列表**")
    st.dataframe(candidates_df(kb, result), use_container_width=True, hide_index=True)

    with st.expander("查看检测到的谱峰"):
        if peaks.get("peaks"):
            st.dataframe(pd.DataFrame(peaks["peaks"]), use_container_width=True, hide_index=True)
        else:
            st.write("未检测到明显谱峰。")

    # ---- ③ 高置信度：直接给本地结论；低置信度：自动联网检索 ---- #
    if level == "高":
        st.success(f"判定：库内已知物质，Top-1 候选为 **{top1['compound']}**。")
    else:
        st.warning("本地参考库未达到高置信度，不输出本地匹配结论。")
        if not api_key:
            st.info("如需联网检索，请填写 API Key 后重新点击「开始识别」。")
        elif st.session_state.get("open_result") is None:
            with st.spinner("正在联网检索适配物质…"):
                try:
                    st.session_state.open_result = _open_search(api_key, base_url, model, raman, result, peaks)
                except Exception as e:
                    st.session_state.open_result = {"error": str(e)}

        res = st.session_state.get("open_result")
        if res:
            if res.get("error"):
                st.error(f"联网检索失败：{res['error']}")
            else:
                st.markdown("**联网检索结果**")
                if res.get("candidates"):
                    st.markdown("**候选化合物名**")
                    st.write("、".join(res["candidates"]))
                if res.get("pubchem"):
                    st.markdown("**PubChem 查证**")
                    st.dataframe(pd.DataFrame(res["pubchem"]), use_container_width=True, hide_index=True)
                if res.get("nist"):
                    st.markdown("**NIST WebBook 查证**")
                    st.dataframe(pd.DataFrame(res["nist"]), use_container_width=True, hide_index=True)
                if res.get("sdbs"):
                    st.markdown("**SDBS 谱库**")
                    for s in res["sdbs"]:
                        if s.get("sdbs_url"):
                            st.markdown(f"- [{s['name']} → SDBS 检索]({s['sdbs_url']})")
                        if s.get("note"):
                            st.caption(s["note"])
                if res.get("web"):
                    st.markdown("**峰位搜索佐证**")
                    for w in res["web"][:8]:
                        if w.get("title"):
                            st.markdown(f"- {w['title']}")
                if res.get("summary"):
                    st.markdown(res["summary"])

    # ---- ④ 导出 ---- #
    st.markdown("**导出结果**")
    export = {
        "query": label,
        "partial": is_partial,
        "peak_based": bool(result.get("peak_based")),
        "confidence": level,
        "agent_analysis": answer,
        "wavenumber_range_cm1": result.get("wavenumber_range_cm1"),
        "best_peak_score": result.get("best_peak_score"),
        "is_known": result.get("is_known"),
        "best_z_score": result.get("best_z_score"),
        "threshold_z": result.get("threshold_z"),
        "top_k_candidates": result["top_k_candidates"],
        "peaks": peaks["peaks"],
        "open_search": st.session_state.get("open_result"),
    }
    json_str = json.dumps(export, ensure_ascii=False, indent=2)
    csv_buf = candidates_df(kb, result).to_csv(index=False).encode("utf-8-sig")
    html_buf = fig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8")

    d1, d2, d3 = st.columns(3)
    d1.download_button("下载结果 JSON", json_str, "identify_result.json", "application/json")
    d2.download_button("下载候选 CSV", csv_buf, "candidates.csv", "text/csv")
    d3.download_button("下载光谱图 HTML", html_buf, "spectrum.html", "text/html")


if __name__ == "__main__":
    main()
