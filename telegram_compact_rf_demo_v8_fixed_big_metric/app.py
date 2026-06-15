
from pathlib import Path
import json
import re
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Telegram Channel Cluster RF Demo",
    page_icon="📡",
    layout="wide",
)


st.markdown(
    """
    <style>
    .build-card {
        background: rgba(125, 125, 125, 0.10);
        border: 1px solid rgba(125, 125, 125, 0.18);
        border-radius: 18px;
        padding: 24px 22px;
        min-height: 130px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    }
    .build-card-label {
        font-size: 1.02rem;
        font-weight: 650;
        opacity: 0.86;
        margin-bottom: 12px;
    }
    .build-card-value {
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.02em;
    }
    .build-card-note {
        font-size: 0.86rem;
        opacity: 0.68;
        margin-top: 9px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE = Path(__file__).parent
DATA = BASE / "data"

@st.cache_data
def load_data():
    scores = pd.read_csv(DATA / "channel_cluster_scores.csv")
    fi = pd.read_csv(DATA / "rf_feature_importance.csv") if (DATA / "rf_feature_importance.csv").exists() else pd.DataFrame()
    cluster_dist = pd.read_csv(DATA / "cluster_distribution.csv") if (DATA / "cluster_distribution.csv").exists() else pd.DataFrame()
    figs = pd.read_csv(DATA / "figures.csv") if (DATA / "figures.csv").exists() else pd.DataFrame()
    log_summary = pd.read_csv(DATA / "network_log_summary.csv") if (DATA / "network_log_summary.csv").exists() else pd.DataFrame()
    log_rows = pd.read_csv(DATA / "network_cell_logs.csv") if (DATA / "network_cell_logs.csv").exists() else pd.DataFrame()
    failed = pd.read_csv(DATA / "network_failed_parquet_files.csv") if (DATA / "network_failed_parquet_files.csv").exists() else pd.DataFrame()
    progress = pd.read_csv(DATA / "network_progress_events.csv") if (DATA / "network_progress_events.csv").exists() else pd.DataFrame()
    top_pr = pd.read_csv(DATA / "network_top_pagerank.csv") if (DATA / "network_top_pagerank.csv").exists() else pd.DataFrame()
    with open(DATA / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    return scores, fi, cluster_dist, figs, log_summary, log_rows, failed, progress, top_pr, summary

scores, fi, cluster_dist, figs, log_summary, log_rows, failed, progress, top_pr, summary = load_data()

if "cluster_key" in scores.columns:
    scores["channel_cluster"] = scores["cluster_key"].astype(str)
else:
    scores["channel_cluster"] = scores.index.astype(str)

if "risk_level" not in scores.columns and "amplification_score" in scores.columns:
    scores["risk_level"] = scores["amplification_score"].apply(lambda x: "High" if x >= 0.66 else ("Medium" if x >= 0.33 else "Low"))


def channel_tokens(value, limit=12):
    if pd.isna(value):
        return []
    ids = re.findall(r"channel_\d+", str(value))
    if not ids and "," in str(value):
        ids = [x.strip() for x in str(value).split(",") if x.strip()]
    return ids[:limit]


def section_note(text):
    st.markdown(
        f"""
        <div style="background:rgba(125,125,125,0.08); border:1px solid rgba(125,125,125,0.15);
        border-radius:12px; padding:14px 16px; margin:8px 0 18px 0;">
        {text}
        </div>
        """,
        unsafe_allow_html=True,
    )



def big_build_metric(label, value, note=""):
    st.markdown(
        f"""
        <div class="build-card">
            <div class="build-card-label">{label}</div>
            <div class="build-card-value">{value}</div>
            <div class="build-card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_big_number(x):
    try:
        return f"{int(float(x)):,}"
    except Exception:
        return str(x)


def show_top_bar(df, score_col, top_n):
    top = df.sort_values(score_col, ascending=False).head(top_n)
    fig = px.bar(
        top.sort_values(score_col),
        x=score_col,
        y="channel_cluster",
        orientation="h",
        color="risk_level" if "risk_level" in top.columns else None,
        hover_data=[c for c in ["channel_count", "message_count", "forwards_per_channel", "toxicity_mean", "threat_rate"] if c in top.columns],
        title=f"Top {top_n} channel clusters by {score_col}",
        height=420,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)


def show_cluster_table(df, score_col):
    cols = [
        "channel_cluster", "cluster_id", "channel_count", "message_count",
        "forwards_per_channel", "forwards_mean", "forwards_sum",
        "toxicity_mean", "threat_mean", "toxic_rate", "threat_rate",
        "amplification_score", "observed_amplification_score",
        "risk_level", "observed_risk_level",
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df.sort_values(score_col, ascending=False)[cols], use_container_width=True, hide_index=True)


def recompute_dynamic_risk(df, low, high):
    df = df.copy()
    df["demo_risk_level"] = df["amplification_score"].apply(
        lambda x: "High" if x >= high else ("Medium" if x >= low else "Low")
    )
    return df


def show_small_images(section_name):
    sub = figs[figs["section"] == section_name]
    if sub.empty:
        st.info(f"Không có hình cho phần {section_name}.")
        return

    # Network figures are wider and need more room, especially the feature distribution chart.
    if section_name == "Network":
        for _, row in sub.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['title']}**")
                if "Feature distributions" in str(row["title"]):
                    st.image(str(BASE / row["file"]), use_container_width=True)
                else:
                    st.image(str(BASE / row["file"]), width=850)
        return

    # SVD figures stay compact in 2 columns.
    image_width = 500
    cols = st.columns(2)
    for idx, (_, row) in enumerate(sub.iterrows()):
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{row['title']}**")
                st.image(str(BASE / row["file"]), width=image_width)


def read_text(rel_path):
    return (BASE / rel_path).read_text(encoding="utf-8", errors="ignore")




# Sidebar
st.sidebar.title("📡 Demo")
page = st.sidebar.radio(
    "Chọn phần",
    [
        "Overview",
        "Channel Cluster Ranking",
        "RandomForest Demo",
        "SVD & Network Evidence",
        "Network Cell Logs",
        "Raw Results",
    ],
)

st.sidebar.caption("Dữ liệu đã đóng gói sẵn. Dữ liệu đã đóng gói sẵn.")


# Header
st.title("Telegram Channel Cluster Amplification Demo")
st.caption(
    "Dashboard rút gọn: kết quả từ các file hiện tại + log cell network. "
    "Ở đây cluster được hiểu là cụm channel Telegram thật, không phải topic cluster."
)


if page == "Overview":
    st.header("1. Tóm tắt kết quả")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Channel clusters", f'{summary.get("n_clusters", len(scores)):,}')
    c2.metric("Real channels", f'{summary.get("n_channels", 0):,}')
    c3.metric("Filtered messages", f'{summary.get("n_messages", 0):,}')
    c4.metric("High-risk clusters", f'{summary.get("high_risk", 0):,}')

    st.subheader("Build-network output statistics")
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        big_build_metric("Network edges", f'{summary.get("network_edges", 0):,}', "Graph relationships")
    with n2:
        big_build_metric("Channel profiles", f'{summary.get("network_profiles", 0):,}', "Channel-level records")
    with n3:
        big_build_metric("Communities", f'{summary.get("network_communities", 0):,}', "Detected groups")
    with n4:
        big_build_metric("Parquet errors", f'{summary.get("network_failed_parquet", 0):,}', "Logged skipped files")

    section_note(
        "<b>Ý tưởng demo:</b> hệ thống gom các channel Telegram thật thành channel clusters, "
        "sau đó đo mức độ khuếch đại bằng forwards_per_channel và RandomForest amplification_score. "
        "Tab Network Cell Logs bổ sung log xử lý từ notebook network để chứng minh pipeline đã chạy trên dữ liệu lớn."
    )

    col1, col2 = st.columns(2)

    with col1:
        if "amplification_score" in scores.columns:
            show_top_bar(scores, "amplification_score", min(10, len(scores)))

    with col2:
        if "forwards_per_channel" in scores.columns and "amplification_score" in scores.columns:
            fig = px.scatter(
                scores,
                x="forwards_per_channel",
                y="amplification_score",
                color="risk_level",
                size="channel_count" if "channel_count" in scores.columns else None,
                hover_name="channel_cluster",
                title="Observed amplification vs RF score",
                height=420,
            )
            fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Kết luận nhanh")
    st.markdown(
        f"""
        - **Top RF risk cluster:** `{summary.get("top_cluster")}`.
        - **Top observed amplification cluster:** `{summary.get("top_observed_cluster")}` theo `forwards_per_channel`.
        - Network notebook tạo được **{summary.get("network_edges", 0):,} edges**, **{summary.get("network_profiles", 0):,} profiles** và **{summary.get("network_communities", 0):,} communities**.
        """
    )


elif page == "Channel Cluster Ranking":
    st.header("2. Ranking theo channel cluster")

    section_note(
        "Phần này là kết quả chính để demo: xếp hạng các channel cluster theo model score hoặc theo chỉ số quan sát thật."
    )

    c1, c2, c3 = st.columns(3)
    score_cols = [c for c in ["amplification_score", "observed_amplification_score", "forwards_per_channel", "forwards_mean", "forwards_sum"] if c in scores.columns]
    score_col = c1.selectbox("Ranking score", score_cols)
    top_n = c2.slider("Top N", 5, min(30, len(scores)), min(14, len(scores)))
    risks = sorted(scores["risk_level"].dropna().unique().tolist()) if "risk_level" in scores.columns else []
    selected_risks = c3.multiselect("Risk filter", risks, default=risks)

    view = scores.copy()
    if selected_risks and "risk_level" in view.columns:
        view = view[view["risk_level"].isin(selected_risks)]

    show_top_bar(view, score_col, top_n)

    st.subheader("Chi tiết một channel cluster")
    selected = st.selectbox("Chọn cluster", view.sort_values("channel_cluster")["channel_cluster"].tolist())
    row = view[view["channel_cluster"] == selected].iloc[0]

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("RF score", f'{row.get("amplification_score", 0):.3f}')
    d2.metric("Observed score", f'{row.get("observed_amplification_score", 0):.3f}')
    d3.metric("Forwards/channel", f'{row.get("forwards_per_channel", 0):,.0f}')
    d4.metric("Channels", f'{int(row.get("channel_count", 0)):,}')
    d5.metric("Messages", f'{int(row.get("message_count", 0)):,}')

    tokens = channel_tokens(row.get("channel_keys_sample", row.get("channel_keys", "")))
    if tokens:
        st.markdown("**Một số channel thật trong cluster này:**")
        st.code(", ".join(tokens))

    show_cluster_table(view, score_col)


elif page == "RandomForest Demo":
    st.header("3. RandomForest model explanation")

    section_note(
        "<b>Phần này giải thích mô hình RandomForest đã train:</b> feature importance, "
        "quan hệ giữa observed amplification và model score, threshold risk level, "
        "và so sánh hai channel clusters."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Feature importance", "Observed vs model", "Threshold demo", "Compare clusters"]
    )

    with tab1:
        st.subheader("Feature importance")
        if not fi.empty and {"feature", "importance"}.issubset(fi.columns):
            fig = px.bar(
                fi.sort_values("importance").tail(20),
                x="importance",
                y="feature",
                orientation="h",
                title="RandomForest feature importance",
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(fi, use_container_width=True, hide_index=True)
        else:
            st.info("Không tìm thấy bảng feature importance.")

    with tab2:
        st.subheader("Observed amplification vs RandomForest score")
        fig = px.scatter(
            scores,
            x="forwards_per_channel" if "forwards_per_channel" in scores.columns else "observed_amplification_score",
            y="amplification_score",
            color="risk_level",
            size="channel_count" if "channel_count" in scores.columns else None,
            hover_name="channel_cluster",
            hover_data=[c for c in ["message_count", "toxicity_mean", "threat_rate", "observed_amplification_score"] if c in scores.columns],
            title="Model score compared with observed amplification",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Risk threshold demo")
        low = st.slider("Low → Medium threshold", 0.0, 1.0, 0.33, 0.01)
        high = st.slider("Medium → High threshold", 0.0, 1.0, 0.66, 0.01)
        if low >= high:
            st.warning("Low threshold phải nhỏ hơn High threshold.")
        else:
            dynamic = recompute_dynamic_risk(scores, low, high)
            risk = dynamic["demo_risk_level"].value_counts().rename_axis("risk").reset_index(name="count")
            fig = px.pie(risk, names="risk", values="count", title="Risk distribution under selected thresholds", height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                dynamic[["channel_cluster", "amplification_score", "demo_risk_level"]].sort_values("amplification_score", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    with tab4:
        st.subheader("Compare two channel clusters")
        options = scores.sort_values("amplification_score", ascending=False)["channel_cluster"].tolist()
        c1, c2 = st.columns(2)
        a = c1.selectbox("Cluster A", options, index=0)
        b = c2.selectbox("Cluster B", options, index=min(1, len(options)-1))

        comp = scores[scores["channel_cluster"].isin([a, b])].copy()
        compare_features = [
            c for c in [
                "amplification_score", "observed_amplification_score",
                "forwards_per_channel", "messages_per_channel",
                "toxicity_mean", "toxicity_std", "threat_mean",
                "insult_mean", "identity_attack_mean",
                "toxic_rate", "threat_rate", "severe_rate",
            ]
            if c in comp.columns
        ]

        melted = comp[["channel_cluster"] + compare_features].melt(
            id_vars="channel_cluster",
            var_name="feature",
            value_name="value",
        )

        fig = px.bar(
            melted,
            x="feature",
            y="value",
            color="channel_cluster",
            barmode="group",
            title="Side-by-side cluster comparison",
            height=520,
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comp[["channel_cluster"] + compare_features], use_container_width=True, hide_index=True)



elif page == "SVD & Network Evidence":
    st.header("4. Output biểu đồ từ các notebook")

    section_note(
        "Phần này giữ lại các biểu đồ quan trọng nhất từ notebook SVD và Network. Ảnh được hiển thị nhỏ hơn dạng 2 cột để dễ đọc."
    )

    st.subheader("SVD evidence")
    show_small_images("SVD")

    st.subheader("Network evidence")
    st.caption("Các biểu đồ Network được hiển thị rộng hơn để dễ đọc boxplot và PCA.")
    show_small_images("Network")


elif page == "Network Cell Logs":
    st.header("5. Network notebook cell logs")

    section_note(
        "Phần này bổ sung kết quả log cell từ build-networking notebook: progress xử lý ZIP/parquet, số edges/profiles, top PageRank, số communities và các parquet đọc lỗi."
    )

    if not log_summary.empty:
        st.subheader("Build-network statistics")
        metric_map = {
            str(row["metric"]): row["value"]
            for _, row in log_summary.iterrows()
        }

        card_cols = st.columns(4)
        with card_cols[0]:
            big_build_metric(
                "Network edges",
                format_big_number(metric_map.get("Number of edges", summary.get("network_edges", 0))),
                "Forwarding / relationship edges"
            )
        with card_cols[1]:
            big_build_metric(
                "Channel profiles",
                format_big_number(metric_map.get("Number of profiles", summary.get("network_profiles", 0))),
                "Unique channel-level profiles"
            )
        with card_cols[2]:
            big_build_metric(
                "Communities",
                format_big_number(metric_map.get("Detected communities", summary.get("network_communities", 0))),
                "Detected network communities"
            )
        with card_cols[3]:
            big_build_metric(
                "Parquet errors",
                format_big_number(metric_map.get("Failed parquet files", summary.get("network_failed_parquet", 0))),
                "Files skipped but logged"
            )

    tab_a, tab_b, tab_c, tab_d = st.tabs(["Summary logs", "Progress events", "Failed parquet files", "Top PageRank"])

    with tab_a:
        st.subheader("Cell log previews")
        if log_rows.empty:
            st.info("Không có log rows.")
        else:
            st.caption("Chỉ hiển thị preview để dashboard gọn hơn.")
            st.dataframe(
                log_rows[["cell", "output", "length", "preview"]],
                use_container_width=True,
                hide_index=True,
            )

    with tab_b:
        st.subheader("ZIP/parquet processing events")
        if progress.empty:
            st.info("Không tìm thấy progress events.")
        else:
            st.dataframe(progress, use_container_width=True, hide_index=True)

    with tab_c:
        st.subheader("Parquet read errors")
        st.caption("Các lỗi này chủ yếu là lỗi đọc footer/parquet trong Spark. Pipeline vẫn tiếp tục chạy và bỏ qua file lỗi.")
        if failed.empty:
            st.success("Không thấy failed parquet files.")
        else:
            st.metric("Unique failed parquet files", len(failed))
            st.dataframe(failed, use_container_width=True, hide_index=True)
            if "zip_group" in failed.columns and failed["zip_group"].notna().any():
                count = failed["zip_group"].fillna("unknown").value_counts().rename_axis("zip_group").reset_index(name="count")
                fig = px.bar(count, x="zip_group", y="count", title="Failed parquet files by ZIP group", height=420)
                st.plotly_chart(fig, use_container_width=True)

    with tab_d:
        st.subheader("Top PageRank channels from log")
        if top_pr.empty:
            st.info("Không parse được top PageRank.")
        else:
            st.dataframe(top_pr, use_container_width=True, hide_index=True)
            fig = px.bar(top_pr.sort_values("pagerank"), x="pagerank", y="from_id", orientation="h", title="Top PageRank channels from notebook log", height=420)
            st.plotly_chart(fig, use_container_width=True)


elif page == "Raw Results":
    st.header("6. Bảng kết quả gốc")

    st.subheader("Channel cluster scores")
    st.dataframe(scores, use_container_width=True, hide_index=True)

    if not fi.empty:
        st.subheader("RandomForest feature importance")
        st.dataframe(fi, use_container_width=True, hide_index=True)

    if not cluster_dist.empty:
        st.subheader("Cluster distribution")
        st.dataframe(cluster_dist, use_container_width=True, hide_index=True)

    if not log_summary.empty:
        st.subheader("Network log summary")
        st.dataframe(log_summary, use_container_width=True, hide_index=True)
