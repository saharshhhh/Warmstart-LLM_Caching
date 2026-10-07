import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="LLM Cache Dashboard", layout="wide")

def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.load(open(p))

replay = load_json("reports/replay_results.json")
curve = load_json("reports/threshold_curve.json", [])
judge = load_json("reports/judge_results.json", {})

st.title("Support AI — Cache Performance")
st.caption("How much caching is saving us, and whether it's still giving correct answers.")

if replay is None:
    st.error("No results found. Run `python run_replay.py` first.")
    st.stop()

summary = replay["summary"]
precision = replay.get("precision", {})

# ---- Top-line numbers, the first thing a support lead should see ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cost savings", f"{summary['cost_savings_pct']:.1f}%",
            help="vs. calling the AI model for every single question")
col2.metric("Overall hit rate", f"{summary['overall_hit_rate']*100:.1f}%",
            help="Fraction of questions answered instantly from cache")
col3.metric("p95 latency", f"{summary['p95_latency_ms']:.0f} ms",
            delta=f"-{summary['baseline_p95_latency_ms'] - summary['p95_latency_ms']:.0f} ms vs no cache",
            delta_color="normal")
col4.metric("Monthly cost (est.)", f"${summary['total_cost_usd']*400:.0f}",
            delta=f"-${(summary['baseline_cost_usd'] - summary['total_cost_usd'])*400:.0f} vs no cache",
            help="Scaled from this sample to ~400,000 questions/month")

st.divider()

# ---- Where the savings come from ----
st.subheader("Where answers come from")
hit_data = pd.DataFrame({
    "Source": ["Exact match\n(same question, reused instantly)",
               "Semantic match\n(similar question, reused)",
               "New question\n(sent to AI model)"],
    "Share": [summary["exact_hit_rate"], summary["semantic_hit_rate"], summary["miss_rate"]],
})
fig = go.Figure(go.Bar(
    x=hit_data["Share"] * 100, y=hit_data["Source"], orientation="h",
    marker_color=["#2563eb", "#60a5fa", "#d1d5db"],
    text=[f"{v*100:.1f}%" for v in hit_data["Share"]], textposition="outside",
))
fig.update_layout(xaxis_title="Share of all questions (%)", height=280, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---- Is the semantic cache actually correct? ----
st.subheader("Is the semantic cache giving the right answers?")

c1, c2 = st.columns(2)
with c1:
    fhr = precision.get("false_hit_rate", None)
    if fhr is not None:
        st.metric("Automated check: wrong-topic rate", f"{fhr*100:.1f}%",
                   help="Checks whether the cached answer matches the question's general topic")
        status = "✅ Within safe range (<1%)" if fhr < 0.01 else "⚠️ Above safe threshold"
        st.write(status)

with c2:
    if judge:
        j80 = judge.get("threshold_0.80", {})
        jfhr = j80.get("judge_false_hit_rate")
        if jfhr is not None:
            st.metric("AI-judge check: incomplete-answer rate", f"{jfhr*100:.1f}%",
                       help="A separate AI reviewed a sample and checked if the answer FULLY addressed the specific question")
            st.write("⚠️ Some cached answers are on-topic but incomplete for the specific phrasing — see Known Limitations below.")

st.divider()

# ---- Threshold tuning curve ----
if curve:
    st.subheader("Similarity threshold: strictness vs. savings")
    st.caption("How strict the match needs to be before we trust it enough to reuse an answer.")
    df = pd.DataFrame(curve)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["threshold"], y=df["hits"], name="Questions caught",
                                yaxis="y1", mode="lines+markers", line=dict(color="#2563eb")))
    fig2.add_trace(go.Scatter(x=df["threshold"], y=df["false_hit_rate"]*100, name="Wrong-topic rate (%)",
                                yaxis="y2", mode="lines+markers", line=dict(color="#dc2626")))
    fig2.update_layout(
        xaxis_title="Strictness threshold",
        yaxis=dict(title="Questions caught", side="left"),
        yaxis2=dict(title="Wrong-topic rate (%)", side="right", overlaying="y"),
        height=350, legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Current production setting: **0.80** — comfortably past the point where errors start appearing.")

st.divider()

# ---- Known limitations, in plain language ----
with st.expander("Known limitations (read before trusting these numbers blindly)"):
    st.markdown("""
- **Synthetic traffic**: based on simulated questions, not real customer logs. Real traffic may have more variety.
- **Answer completeness**: a separate AI review found some cached answers are on the right topic but don't
  fully answer narrower versions of the question (e.g. "is it free?" vs "how do returns work?").
  This needs either more specific answer templates or finer question grouping before wider rollout.
- **Small sample at the edges**: threshold behavior below 0.70 is based on a small number of examples
  and should be re-checked with more data before changing the production threshold.
    """)

st.caption("Last updated from `reports/replay_results.json`, `threshold_curve.json`, `judge_results.json`")