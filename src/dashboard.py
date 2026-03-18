import json
import os
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from praat_analysis import analyze_formant_track

RESULT_DIR = "results"
WORD_HISTORY_FILE = os.path.join(RESULT_DIR, "history.csv")
PHONEME_HISTORY_FILE = os.path.join(RESULT_DIR, "phoneme_history.csv")

MAX_COMPARE_FILES = 3
CHART_COLORS = px.colors.qualitative.Plotly

st.set_page_config(
    page_title="Pronunciation Analyzer",
    page_icon="🎙️",
    layout="wide",
)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "accurecy" in df.columns and "accuracy" not in df.columns:
        df = df.rename(columns={"accurecy": "accuracy"})
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


@st.cache_data
def load_data():
    return load_csv(WORD_HISTORY_FILE), load_csv(PHONEME_HISTORY_FILE)


def normalize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def init_state():
    if "selected_wavs" not in st.session_state:
        st.session_state["selected_wavs"] = []
    if "custom_date_range" not in st.session_state:
        st.session_state["custom_date_range"] = None
    if "quick_range" not in st.session_state:
        st.session_state["quick_range"] = "30日"


def apply_date_filter(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df.empty or "time" not in df.columns:
        return df.copy()
    start_ts = pd.Timestamp(datetime.combine(start_date, datetime.min.time()))
    end_ts = pd.Timestamp(datetime.combine(end_date, datetime.max.time()))
    return df[(df["time"] >= start_ts) & (df["time"] <= end_ts)].copy()


def parse_samples_json(samples_json):
    if not isinstance(samples_json, str) or not samples_json.strip():
        return []
    try:
        data = json.loads(samples_json)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def parse_candidates_json(candidates_json):
    if not isinstance(candidates_json, str) or not candidates_json.strip():
        return []
    try:
        data = json.loads(candidates_json)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def format_score(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        value = float(value)
    except Exception:
        return str(value)
    return f"{value:.0f}" if value.is_integer() else f"{value:.1f}"


def pick_misrecognition_candidates(candidates: list, target_phoneme: str | None, limit: int = 3) -> list[dict]:
    picked = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        phoneme = candidate.get("phoneme")
        if not phoneme or phoneme == target_phoneme:
            continue
        picked.append({
            "phoneme": str(phoneme),
            "score": pd.to_numeric(candidate.get("score"), errors="coerce"),
        })
        if len(picked) >= limit:
            break
    return picked


def format_misrecognition_summary(candidates: list[dict], target_phoneme: str | None = None, limit: int = 3, show_source: bool = False) -> str:
    picked = pick_misrecognition_candidates(candidates, target_phoneme, limit=limit)
    if not picked:
        return ""

    parts = []
    for candidate in picked:
        label = candidate["phoneme"]
        if show_source and target_phoneme:
            label = f"{target_phoneme}->{label}"
        score = candidate.get("score")
        if score is not None and not pd.isna(score):
            label = f"{label}({format_score(score)})"
        parts.append(label)
    return " | ".join(parts)


def enrich_phoneme_misrecognitions(phoneme_df: pd.DataFrame) -> pd.DataFrame:
    if phoneme_df.empty:
        return phoneme_df.copy()

    enriched = phoneme_df.copy()
    summaries = []
    primary_confusions = []
    primary_confusion_scores = []

    json_col = "misrecognition_candidates_json"
    for _, row in enriched.iterrows():
        candidates = parse_candidates_json(row.get(json_col))
        target_phoneme = row.get("phoneme")
        picked = pick_misrecognition_candidates(candidates, target_phoneme, limit=3)
        summaries.append(format_misrecognition_summary(candidates, target_phoneme, limit=3))
        if picked:
            primary_confusions.append(picked[0]["phoneme"])
            primary_confusion_scores.append(picked[0]["score"])
        else:
            primary_confusions.append(None)
            primary_confusion_scores.append(None)

    enriched["misrecognition_top3"] = summaries
    enriched["primary_confusion"] = primary_confusions
    enriched["primary_confusion_score"] = primary_confusion_scores
    return enriched


def build_recording_misrecognition_summary(phoneme_df: pd.DataFrame) -> pd.DataFrame:
    if phoneme_df.empty or "wav_path" not in phoneme_df.columns:
        return pd.DataFrame(columns=["wav_path", "phoneme_top1_confusions"])

    rows = []
    for _, row in phoneme_df.iterrows():
        target_phoneme = row.get("phoneme")
        primary_confusion = row.get("primary_confusion")
        if not target_phoneme or not primary_confusion:
            continue

        rows.append({
            "wav_path": row.get("wav_path"),
            "word_index": pd.to_numeric(row.get("word_index"), errors="coerce"),
            "phoneme_index": pd.to_numeric(row.get("phoneme_index"), errors="coerce"),
            "target_phoneme": str(target_phoneme),
            "primary_confusion": str(primary_confusion),
            "primary_confusion_score": pd.to_numeric(row.get("primary_confusion_score"), errors="coerce"),
        })

    if not rows:
        return pd.DataFrame(columns=["wav_path", "phoneme_top1_confusions"])

    summary_df = pd.DataFrame(rows).sort_values(
        ["wav_path", "word_index", "phoneme_index"],
        ascending=[True, True, True],
        na_position="last",
    )

    agg_rows = []
    for wav_path, grp in summary_df.groupby("wav_path", dropna=False):
        labels = []
        for _, item in grp.iterrows():
            label = f"{item['target_phoneme']}->{item['primary_confusion']}"
            if pd.notna(item["primary_confusion_score"]):
                label += f"({format_score(item['primary_confusion_score'])})"
            labels.append(label)
        agg_rows.append({
            "wav_path": wav_path,
            "phoneme_top1_confusions": " | ".join(labels),
        })
    agg = pd.DataFrame(agg_rows)
    return agg


def attach_recording_misrecognitions(word_df: pd.DataFrame, phoneme_df: pd.DataFrame) -> pd.DataFrame:
    if word_df.empty:
        return word_df.copy()

    summary_df = build_recording_misrecognition_summary(phoneme_df)
    if summary_df.empty:
        enriched = word_df.copy()
        enriched["phoneme_top1_confusions"] = ""
        return enriched

    return word_df.merge(summary_df, on="wav_path", how="left")


def make_confusion_stats(phoneme_df: pd.DataFrame) -> pd.DataFrame:
    if phoneme_df.empty:
        return pd.DataFrame()

    rows = []
    json_col = "misrecognition_candidates_json"
    for _, row in phoneme_df.iterrows():
        target_phoneme = row.get("phoneme")
        candidates = parse_candidates_json(row.get(json_col))
        picked = pick_misrecognition_candidates(candidates, target_phoneme, limit=3)
        for rank, candidate in enumerate(picked, start=1):
            rows.append({
                "phoneme": target_phoneme,
                "candidate_phoneme": candidate.get("phoneme"),
                "candidate_score": candidate.get("score"),
                "rank": rank,
                "accuracy": pd.to_numeric(row.get("accuracy"), errors="coerce"),
                "target_word": row.get("target_word"),
                "wav_path": row.get("wav_path"),
            })

    if not rows:
        return pd.DataFrame()

    confusion_df = pd.DataFrame(rows)
    source_totals = (
        phoneme_df.groupby("phoneme", dropna=False)
        .agg(source_samples=("phoneme", "count"))
        .reset_index()
    )
    agg = (
        confusion_df.groupby(["phoneme", "candidate_phoneme"], dropna=False)
        .agg(
            pair_samples=("candidate_phoneme", "count"),
            avg_candidate_score=("candidate_score", "mean"),
            avg_accuracy=("accuracy", "mean"),
            rank1_hits=("rank", lambda s: int((s == 1).sum())),
            unique_recordings=("wav_path", pd.Series.nunique),
        )
        .reset_index()
    )
    agg = agg.merge(source_totals, on="phoneme", how="left")
    agg["pair_rate"] = agg["pair_samples"] / agg["source_samples"]
    agg["rank1_rate"] = agg["rank1_hits"] / agg["source_samples"]
    agg["pair_rate_pct"] = agg["pair_rate"] * 100.0
    agg["rank1_rate_pct"] = agg["rank1_rate"] * 100.0
    agg = agg.sort_values(
        ["rank1_rate", "pair_rate", "avg_candidate_score"],
        ascending=[False, False, False],
        na_position="last",
    )
    return agg


def make_phoneme_confusion_overview(confusion_stats: pd.DataFrame) -> pd.DataFrame:
    if confusion_stats.empty:
        return pd.DataFrame()

    rows = []
    for phoneme, grp in confusion_stats.groupby("phoneme", dropna=False):
        grp = grp.sort_values(
            ["rank1_rate", "pair_rate", "avg_candidate_score"],
            ascending=[False, False, False],
            na_position="last",
        )
        top = grp.iloc[0]
        rows.append({
            "phoneme": phoneme,
            "source_samples": int(top["source_samples"]),
            "distinct_confusions": int(grp["candidate_phoneme"].nunique()),
            "most_confused_with": top["candidate_phoneme"],
            "top_pair_rate_pct": top["pair_rate_pct"],
            "top1_confusion_rate_pct": top["rank1_rate_pct"],
            "top_pair_score": top["avg_candidate_score"],
        })

    return pd.DataFrame(rows).sort_values(
        ["top1_confusion_rate_pct", "top_pair_rate_pct", "top_pair_score"],
        ascending=[False, False, False],
        na_position="last",
    )


@st.cache_data(show_spinner=False)
def load_formant_track(wav_path: str, time_step_sec: float):
    if not isinstance(wav_path, str) or not os.path.exists(wav_path):
        return pd.DataFrame()

    track = analyze_formant_track(wav_path, time_step_sec=time_step_sec)
    samples = track.get("samples", [])
    if not samples:
        return pd.DataFrame()

    df = pd.DataFrame(samples)
    for c in ["time", "f1", "f2", "f3", "f3_f2_gap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def checkbox_selector(label: str, items: list[str], key_prefix: str) -> list[str]:
    st.markdown(f"**{label}**")
    if not items:
        st.caption("選択可能な項目がありません。")
        return []

    c1, c2 = st.columns(2)
    if c1.button("すべて選択", key=f"{key_prefix}_all", use_container_width=True):
        for item in items:
            st.session_state[f"{key_prefix}_{item}"] = True
    if c2.button("すべて解除", key=f"{key_prefix}_none", use_container_width=True):
        for item in items:
            st.session_state[f"{key_prefix}_{item}"] = False

    selected = []
    cols = st.columns(3)
    for i, item in enumerate(items):
        k = f"{key_prefix}_{item}"
        if k not in st.session_state:
            st.session_state[k] = True
        with cols[i % 3]:
            checked = st.checkbox(item, key=k)
        if checked:
            selected.append(item)
    return selected


def render_global_filters(word_df: pd.DataFrame, phoneme_df: pd.DataFrame):
    st.title("🎙️ Pronunciation Analyzer")

    all_times = []
    if not word_df.empty and "time" in word_df.columns:
        all_times.extend(word_df["time"].dropna().tolist())
    if not phoneme_df.empty and "time" in phoneme_df.columns:
        all_times.extend(phoneme_df["time"].dropna().tolist())

    if all_times:
        min_date = min(all_times).date()
        max_date = max(all_times).date()
    else:
        min_date = max_date = date.today()

    st.markdown("### 期間")

    quick_options = {"7日": 7, "14日": 14, "30日": 30, "全期間": None}
    selected_quick = st.radio(
        "クイック選択",
        list(quick_options.keys()),
        index=list(quick_options.keys()).index(st.session_state["quick_range"]),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["quick_range"] = selected_quick

    if quick_options[selected_quick] is None:
        default_start = min_date
        default_end = max_date
    else:
        days = quick_options[selected_quick]
        default_end = max_date
        default_start = max(min_date, max_date - timedelta(days=days - 1))

    if st.session_state["custom_date_range"] is None:
        st.session_state["custom_date_range"] = (default_start, default_end)

    date_range = st.date_input(
        "詳細期間",
        value=st.session_state["custom_date_range"],
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        st.session_state["custom_date_range"] = (start_date, end_date)
    else:
        start_date, end_date = default_start, default_end

    return start_date, end_date


def _prepare_chart_df(df: pd.DataFrame, x: str, y: str, color: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    target = df.copy()
    if y == "accuracy" and "accuracy" not in target.columns and "accurecy" in target.columns:
        target = target.rename(columns={"accurecy": "accuracy"})

    if x not in target.columns or y not in target.columns or color not in target.columns:
        return pd.DataFrame()

    target[y] = pd.to_numeric(target[y], errors="coerce")
    target = target.dropna(subset=[x, y, color]).sort_values(x).copy()
    return target


def build_interactive_line_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str, hover_data: list[str], key: str):
    target = _prepare_chart_df(df, x, y, color)
    if target.empty:
        st.info(f"{title} に表示できるデータがありません。")
        return

    fig = px.line(
        target,
        x=x,
        y=y,
        color=color,
        markers=True,
        hover_data=[c for c in hover_data if c in target.columns],
        custom_data=[c for c in ["wav_path", "session_id", "time"] if c in target.columns],
        title=title,
        color_discrete_sequence=CHART_COLORS,
    )

    fig.update_layout(height=420, legend_title_text=color, hovermode="closest")

    st.plotly_chart(fig, use_container_width=True, key=key)


def build_interactive_scatter_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str, hover_data: list[str], symbol: str | None, key: str):
    target = _prepare_chart_df(df, x, y, color)
    if target.empty:
        st.info(f"{title} に表示できるデータがありません。")
        return

    fig = px.scatter(
        target,
        x=x,
        y=y,
        color=color,
        symbol=symbol if symbol and symbol in target.columns else None,
        hover_data=[c for c in hover_data if c in target.columns],
        custom_data=[c for c in ["wav_path", "session_id", "time"] if c in target.columns],
        title=title,
        color_discrete_sequence=CHART_COLORS,
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(autorange="reversed")
    fig.update_layout(height=420, legend_title_text=color, hovermode="closest")

    st.plotly_chart(fig, use_container_width=True, key=key)


def plot_phoneme_formants(df: pd.DataFrame, formant_mode: str, key_prefix: str):
    if formant_mode == "F1-F2":
        build_interactive_scatter_chart(
            df,
            x="f2_mean",
            y="f1_mean",
            color="phoneme",
            symbol="target_word",
            hover_data=["time", "target_word", "wav_path"],
            title="音素ごとの F1-F2 分布",
            key=f"{key_prefix}_f1f2",
        )
        return

    mode_map = {
        "F1": "f1_mean",
        "F2": "f2_mean",
        "F3": "f3_mean",
        "F3(min)": "f3_min",
        "F3-F2 gap(min)": "f3_f2_gap_min",
    }
    col = mode_map[formant_mode]
    build_interactive_line_chart(
        df,
        x="time",
        y=col,
        color="phoneme",
        title=f"音素ごとの {formant_mode} 推移",
        hover_data=["target_word", "wav_path"],
        key=f"{key_prefix}_{formant_mode}",
    )


def add_phoneme_bands(fig: go.Figure, phoneme_rows: pd.DataFrame):
    if phoneme_rows.empty:
        return

    rows = phoneme_rows.sort_values(["offset_sec", "phoneme_index"], na_position="last").copy()
    for _, row in rows.iterrows():
        start = row.get("offset_sec")
        duration = row.get("duration_sec")
        label = row.get("phoneme")
        if pd.isna(start) or pd.isna(duration):
            continue
        end = float(start) + float(duration)
        fig.add_vrect(
            x0=float(start),
            x1=float(end),
            fillcolor="lightgray",
            opacity=0.16,
            line_width=0,
            annotation_text=str(label) if pd.notna(label) else "",
            annotation_position="top left",
        )


def plot_selected_formant_track_with_bands(wav_path: str, phoneme_rows: pd.DataFrame, key_prefix: str, panel_index: int):
    if not isinstance(wav_path, str) or not os.path.exists(wav_path):
        st.info("フォルマント時系列を表示できる音声ファイルがありません。")
        return

    cols = st.columns([3, 1.4, 1, 1, 1, 1])
    cols[0].markdown(f"#### 比較 {panel_index + 1}")
    cols[1].caption(os.path.basename(wav_path))

    time_step_ms = cols[2].selectbox(
        "刻み",
        [5, 10, 20, 30],
        index=1,
        key=f"{key_prefix}_step_{panel_index}",
    )
    show_f1 = cols[3].checkbox("F1", value=True, key=f"{key_prefix}_f1_{panel_index}")
    show_f2 = cols[4].checkbox("F2", value=True, key=f"{key_prefix}_f2_{panel_index}")
    show_f3 = cols[5].checkbox("F3", value=True, key=f"{key_prefix}_f3_{panel_index}")

    track_df = load_formant_track(wav_path, time_step_sec=time_step_ms / 1000.0)
    if track_df.empty:
        st.info("フォルマント時系列データが取得できませんでした。")
        return

    fig = go.Figure()

    if show_f1 and "f1" in track_df.columns:
        d = track_df.dropna(subset=["f1"])
        fig.add_trace(go.Scatter(x=d["time"], y=d["f1"], mode="lines", name="F1"))
    if show_f2 and "f2" in track_df.columns:
        d = track_df.dropna(subset=["f2"])
        fig.add_trace(go.Scatter(x=d["time"], y=d["f2"], mode="lines", name="F2"))
    if show_f3 and "f3" in track_df.columns:
        d = track_df.dropna(subset=["f3"])
        fig.add_trace(go.Scatter(x=d["time"], y=d["f3"], mode="lines", name="F3"))

    add_phoneme_bands(fig, phoneme_rows)

    fig.update_layout(
        title="フォルマント時間変化 + Azure phoneme 範囲",
        xaxis_title="time (sec)",
        yaxis_title="Hz",
        height=360,
        dragmode="pan",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if os.path.exists(wav_path):
        st.audio(wav_path)

    with st.expander("時系列データ", expanded=False):
        st.dataframe(track_df, use_container_width=True, hide_index=True)


def render_selected_files_comparison(phoneme_df: pd.DataFrame, key_prefix: str):
    selected_wavs = st.session_state.get("selected_wavs", [])
    if not selected_wavs:
        return None
    st.markdown("### 選択中ファイル比較")
    for i, wav_path in enumerate(selected_wavs[:MAX_COMPARE_FILES]):
        file_phonemes = (
            phoneme_df[phoneme_df["wav_path"] == wav_path].copy()
            if "wav_path" in phoneme_df.columns
            else pd.DataFrame()
        )
        plot_selected_formant_track_with_bands(wav_path, file_phonemes, key_prefix, i)
    return None


def build_audio_panel(df: pd.DataFrame, key_prefix: str):
    st.markdown("### 録音一覧")

    if df.empty:
        st.info("表示対象がありません。")
        return None

    sorted_df = df.sort_values("time", ascending=False).reset_index(drop=True)

    display_cols = [
        c for c in [
            "time", "word", "target_word", "phoneme", "accuracy", "pron",
            "fluency", "completeness", "f1_mean", "f2_mean", "f3_mean",
            "f3_min", "f3_f2_gap_min", "phoneme_top1_confusions", "misrecognition_top3", "wav_path"
        ] if c in sorted_df.columns
    ]

    editor_df = sorted_df[display_cols].copy()
    editor_df.insert(0, "selected", editor_df["wav_path"].isin(st.session_state.get("selected_wavs", [])))

    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        disabled=[c for c in editor_df.columns if c != "selected"],
        column_config={
            "selected": st.column_config.CheckboxColumn("selected"),
            "phoneme_top1_confusions": st.column_config.TextColumn("音素別誤認候補 Top1", width="large"),
            "misrecognition_top3": st.column_config.TextColumn("音素別誤認候補 Top3", width="large"),
        },
        key=f"{key_prefix}_editor",
    )

    selected_wavs = (
        edited.loc[edited["selected"], "wav_path"].dropna().astype(str).drop_duplicates().tolist()
        if "wav_path" in edited.columns
        else []
    )
    selected_wavs = selected_wavs[:MAX_COMPARE_FILES]
    previous_selected = st.session_state.get("selected_wavs", [])
    st.session_state["selected_wavs"] = selected_wavs
    if selected_wavs != previous_selected:
        st.rerun()
    return st.session_state["selected_wavs"]


def filter_word_view(word_df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 単語フィルタ")
        words = sorted(word_df["word"].dropna().unique().tolist()) if "word" in word_df.columns else []
        selected_words = checkbox_selector("表示する単語", words, "word_filter")

        show_optional = st.toggle("追加スコアを表示", value=False, key="word_optional_toggle")
        optional_scores = []
        if show_optional:
            optional_candidates = [c for c in ["fluency", "completeness"] if c in word_df.columns]
            optional_scores = checkbox_selector("追加表示スコア", optional_candidates, "word_optional_scores")

    filtered = word_df[word_df["word"].isin(selected_words)].copy() if selected_words else word_df.iloc[0:0].copy()
    return filtered, optional_scores


def filter_phoneme_view(phoneme_df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 音素フィルタ")

        all_words = sorted(phoneme_df["target_word"].dropna().unique().tolist()) if "target_word" in phoneme_df.columns else []
        selected_words = checkbox_selector("対象単語", all_words, "phoneme_words")

        filtered = phoneme_df[phoneme_df["target_word"].isin(selected_words)].copy() if selected_words else phoneme_df.iloc[0:0].copy()

        phonemes = sorted(filtered["phoneme"].dropna().unique().tolist()) if "phoneme" in filtered.columns else []
        selected_phonemes = checkbox_selector("表示する音素", phonemes, "phoneme_filter")

        formant_mode = st.radio(
            "フォルマント表示",
            ["F1-F2", "F1", "F2", "F3", "F3(min)", "F3-F2 gap(min)"],
            horizontal=True,
        )

    filtered = filtered[filtered["phoneme"].isin(selected_phonemes)].copy() if selected_phonemes else filtered.iloc[0:0].copy()
    return filtered, formant_mode


def render_word_view(word_df: pd.DataFrame, phoneme_df: pd.DataFrame):
    st.markdown("## 単語ビュー")
    filtered_df, optional_scores = filter_word_view(word_df)

    if filtered_df.empty:
        st.warning("表示対象の単語がありません。")
        return

    c1, c2 = st.columns(2)
    with c1:
        if "accuracy" in filtered_df.columns:
            build_interactive_line_chart(
                filtered_df, "time", "accuracy", "word",
                "単語ごとの accuracy 推移",
                ["recognized_text", "wav_path"],
                "word_accuracy_chart",
            )
    with c2:
        if "pron" in filtered_df.columns:
            build_interactive_line_chart(
                filtered_df, "time", "pron", "word",
                "単語ごとの PronScore 推移",
                ["recognized_text", "wav_path"],
                "word_pron_chart",
            )

    if optional_scores:
        st.markdown("### 追加スコア")
        cols = st.columns(len(optional_scores))
        for i, score_col in enumerate(optional_scores):
            with cols[i]:
                build_interactive_line_chart(
                    filtered_df, "time", score_col, "word",
                    f"単語ごとの {score_col} 推移",
                    ["recognized_text", "wav_path"],
                    f"word_optional_{score_col}",
                )

    build_audio_panel(filtered_df, "word")
    render_selected_files_comparison(phoneme_df, "word_compare")


def render_phoneme_view(phoneme_df: pd.DataFrame):
    st.markdown("## 音素ビュー")
    filtered_df, formant_mode = filter_phoneme_view(phoneme_df)

    if filtered_df.empty:
        st.warning("表示対象の音素がありません。")
        return

    c1, c2 = st.columns(2)
    with c1:
        build_interactive_line_chart(
            filtered_df, "time", "accuracy", "phoneme",
            "音素ごとの accuracy 推移",
            ["target_word", "wav_path"],
            "phoneme_accuracy_chart",
        )
    with c2:
        plot_phoneme_formants(filtered_df, formant_mode, "phoneme_formants")

    build_audio_panel(filtered_df, "phoneme")
    render_confusion_tables(filtered_df, "phoneme_view")
    render_selected_files_comparison(filtered_df, "phoneme_compare")


def make_word_stats(word_df: pd.DataFrame) -> pd.DataFrame:
    if word_df.empty:
        return pd.DataFrame()

    agg = word_df.groupby("word", dropna=False).agg(
        samples=("word", "count"),
        accuracy_avg=("accuracy", "mean"),
        accuracy_best=("accuracy", "max"),
        accuracy_worst=("accuracy", "min"),
        pron_avg=("pron", "mean"),
        pron_best=("pron", "max"),
        pron_worst=("pron", "min"),
        last_accuracy=("accuracy", "last"),
        last_pron=("pron", "last"),
    ).reset_index()

    return agg.sort_values(["accuracy_avg", "pron_avg"], ascending=[False, False])


def make_phoneme_stats(phoneme_df: pd.DataFrame) -> pd.DataFrame:
    if phoneme_df.empty:
        return pd.DataFrame()

    agg = phoneme_df.groupby("phoneme", dropna=False).agg(
        samples=("phoneme", "count"),
        accuracy_avg=("accuracy", "mean"),
        accuracy_best=("accuracy", "max"),
        accuracy_worst=("accuracy", "min"),
        last_accuracy=("accuracy", "last"),
    ).reset_index()

    return agg.sort_values(["accuracy_avg"], ascending=[False])


def make_progress_stats(df: pd.DataFrame, group_col: str, score_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns or score_col not in df.columns or "time" not in df.columns:
        return pd.DataFrame()

    target = df.dropna(subset=[group_col, score_col, "time"]).sort_values("time").copy()
    if target.empty:
        return pd.DataFrame()

    rows = []
    for item, grp in target.groupby(group_col, dropna=False):
        grp = grp.sort_values("time")
        rows.append(
            {
                group_col: item,
                "samples": len(grp),
                "avg_score": float(grp[score_col].mean()),
                "first_score": float(grp[score_col].iloc[0]),
                "last_score": float(grp[score_col].iloc[-1]),
                "improvement": float(grp[score_col].iloc[-1] - grp[score_col].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def pick_focus_item(stats_df: pd.DataFrame, label_col: str, score_col: str, min_samples: int = 1, ascending: bool = True) -> dict | None:
    if stats_df.empty:
        return None

    target = stats_df[stats_df["samples"] >= min_samples].sort_values([score_col, "samples"], ascending=[ascending, False])
    if target.empty:
        return None

    row = target.iloc[0]
    return {"label": row[label_col], "score": row[score_col], "samples": row["samples"]}


def make_daily_accuracy_trend(word_df: pd.DataFrame, phoneme_df: pd.DataFrame) -> pd.DataFrame:
    source = word_df if not word_df.empty and "accuracy" in word_df.columns else phoneme_df
    if source.empty or "time" not in source.columns or "accuracy" not in source.columns:
        return pd.DataFrame()

    target = source.dropna(subset=["time", "accuracy"]).copy()
    if target.empty:
        return pd.DataFrame()

    target["date"] = target["time"].dt.date
    trend = (
        target.groupby("date", dropna=False)
        .agg(accuracy_avg=("accuracy", "mean"), samples=("accuracy", "count"))
        .reset_index()
        .sort_values("date")
    )
    trend["date"] = pd.to_datetime(trend["date"])
    return trend


def render_insight_panel(title: str, items: list[dict[str, str]]):
    st.markdown(f"### {title}")
    if not items:
        st.info("表示できる統計がありません。")
        return

    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)


def render_confusion_tables(phoneme_df: pd.DataFrame, key_prefix: str):
    confusion_stats = make_confusion_stats(phoneme_df)
    overview_df = make_phoneme_confusion_overview(confusion_stats)

    st.markdown("### 誤認傾向")
    if confusion_stats.empty:
        st.info("誤認候補の統計を出せるデータがまだありません。")
        return

    summary = st.columns(3)
    best_pair = confusion_stats.iloc[0]
    weakest_phoneme = overview_df.iloc[0] if not overview_df.empty else None
    summary[0].metric("最高 Top1 誤認率", f"{best_pair['rank1_rate_pct']:.1f}%")
    summary[1].metric("最高 ペア出現率", f"{best_pair['pair_rate_pct']:.1f}%")
    summary[2].metric(
        "最も崩れやすい音素",
        str(weakest_phoneme["phoneme"]) if weakest_phoneme is not None else "-",
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### 誤認ペア")
        st.dataframe(
            confusion_stats.head(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                "phoneme": "正解音素",
                "candidate_phoneme": "誤認候補",
                "source_samples": "正解音素サンプル数",
                "pair_samples": "該当ペア件数",
                "pair_rate_pct": "ペア出現率(%)",
                "avg_candidate_score": "候補スコア平均",
                "avg_accuracy": "元音素 accuracy 平均",
                "rank1_hits": "Top1 だった件数",
                "rank1_rate_pct": "Top1 誤認率(%)",
                "unique_recordings": "録音数",
            },
        )
    with right:
        st.markdown("#### 音素ごとの誤認されやすさ")
        st.dataframe(
            overview_df.head(20),
            use_container_width=True,
            hide_index=True,
            column_config={
                "phoneme": "音素",
                "source_samples": "サンプル数",
                "distinct_confusions": "候補種類数",
                "most_confused_with": "最頻候補",
                "top_pair_rate_pct": "最多ペア出現率(%)",
                "top1_confusion_rate_pct": "Top1 誤認率(%)",
                "top_pair_score": "最多ペアの平均スコア",
            },
        )

    with st.expander("誤認ペア全件を見る", expanded=False):
        st.dataframe(
            confusion_stats,
            use_container_width=True,
            hide_index=True,
            column_config={
                "phoneme": "正解音素",
                "candidate_phoneme": "誤認候補",
                "source_samples": "正解音素サンプル数",
                "pair_samples": "該当ペア件数",
                "pair_rate_pct": "ペア出現率(%)",
                "avg_candidate_score": "候補スコア平均",
                "avg_accuracy": "元音素 accuracy 平均",
                "rank1_hits": "Top1 だった件数",
                "rank1_rate_pct": "Top1 誤認率(%)",
                "unique_recordings": "録音数",
            },
            key=f"{key_prefix}_confusion_pairs",
        )


def _legacy_render_statistics_view(word_df: pd.DataFrame, phoneme_df: pd.DataFrame):
    st.markdown("## 統計ビュー")

    word_stats = make_word_stats(word_df)
    phoneme_stats = make_phoneme_stats(phoneme_df)

    tab1, tab2 = st.tabs(["単語統計", "音素統計"])

    with tab1:
        if word_stats.empty:
            st.info("単語統計に表示できるデータがありません。")
        else:
            l, r = st.columns(2)
            with l:
                st.markdown("### うまい単語ランキング")
                st.dataframe(word_stats.sort_values(["accuracy_avg", "pron_avg"], ascending=[False, False]).head(20), use_container_width=True, hide_index=True)
            with r:
                st.markdown("### 下手単語ランキング")
                st.dataframe(word_stats.sort_values(["accuracy_avg", "pron_avg"], ascending=[True, True]).head(20), use_container_width=True, hide_index=True)
            st.markdown("### 単語統計一覧")
            st.dataframe(word_stats, use_container_width=True, hide_index=True)

    with tab2:
        if phoneme_stats.empty:
            st.info("音素統計に表示できるデータがありません。")
        else:
            l, r = st.columns(2)
            with l:
                st.markdown("### うまい音素ランキング")
                st.dataframe(phoneme_stats.sort_values(["accuracy_avg"], ascending=[False]).head(20), use_container_width=True, hide_index=True)
            with r:
                st.markdown("### 下手音素ランキング")
                st.dataframe(phoneme_stats.sort_values(["accuracy_avg"], ascending=[True]).head(20), use_container_width=True, hide_index=True)
            st.markdown("### 音素統計一覧")
            st.dataframe(phoneme_stats, use_container_width=True, hide_index=True)


def render_statistics_dashboard(word_df: pd.DataFrame, phoneme_df: pd.DataFrame):
    st.markdown("## 統計ダッシュボード")

    word_stats = make_word_stats(word_df)
    phoneme_stats = make_phoneme_stats(phoneme_df)
    word_progress = make_progress_stats(word_df, "word", "accuracy")
    phoneme_progress = make_progress_stats(phoneme_df, "phoneme", "accuracy")
    daily_trend = make_daily_accuracy_trend(word_df, phoneme_df)

    overall_accuracy = word_df["accuracy"].mean() if not word_df.empty and "accuracy" in word_df.columns else phoneme_df["accuracy"].mean()
    overall_pron = word_df["pron"].mean() if not word_df.empty and "pron" in word_df.columns else float("nan")
    total_recordings = int(len(word_df)) if not word_df.empty else 0
    total_practice_days = int(word_df["time"].dt.date.nunique()) if not word_df.empty and "time" in word_df.columns else 0

    hardest_word = pick_focus_item(word_stats, "word", "accuracy_avg")
    hardest_phoneme = pick_focus_item(phoneme_stats, "phoneme", "accuracy_avg")
    improved_word = pick_focus_item(word_progress[word_progress["samples"] >= 2], "word", "improvement", min_samples=2, ascending=False) if not word_progress.empty else None
    improved_phoneme = pick_focus_item(phoneme_progress[phoneme_progress["samples"] >= 2], "phoneme", "improvement", min_samples=2, ascending=False) if not phoneme_progress.empty else None
    word_sample_threshold = max(2, int(word_stats["samples"].median())) if not word_stats.empty else 2
    phoneme_sample_threshold = max(2, int(phoneme_stats["samples"].median())) if not phoneme_stats.empty else 2
    low_avg_many_word = pick_focus_item(word_stats[word_stats["samples"] >= word_sample_threshold], "word", "accuracy_avg", min_samples=word_sample_threshold) if not word_stats.empty else None
    low_avg_many_phoneme = pick_focus_item(phoneme_stats[phoneme_stats["samples"] >= phoneme_sample_threshold], "phoneme", "accuracy_avg", min_samples=phoneme_sample_threshold) if not phoneme_stats.empty else None

    summary = st.columns(4)
    summary[0].metric("平均 accuracy", "-" if pd.isna(overall_accuracy) else f"{overall_accuracy:.1f}")
    summary[1].metric("平均 PronScore", "-" if pd.isna(overall_pron) else f"{overall_pron:.1f}")
    summary[2].metric("総録音回数", total_recordings)
    summary[3].metric("総練習日数", total_practice_days)

    focus_col, improve_col = st.columns(2)
    with focus_col:
        focus_items = []
        if hardest_word:
            focus_items.append(("最も苦手な単語", f"{hardest_word['label']}\n平均 accuracy: {hardest_word['score']:.1f}\nサンプル数: {hardest_word['samples']}回"))
        if hardest_phoneme:
            focus_items.append(("最も苦手な音素", f"{hardest_phoneme['label']}\n平均 accuracy: {hardest_phoneme['score']:.1f}\nサンプル数: {hardest_phoneme['samples']}回"))
        if low_avg_many_word:
            focus_items.append(("回数が多いのに平均が低い単語", f"{low_avg_many_word['label']}\n平均 accuracy: {low_avg_many_word['score']:.1f}"))
        if low_avg_many_phoneme:
            focus_items.append(("回数が多いのに平均が低い音素", f"{low_avg_many_phoneme['label']}\n平均 accuracy: {low_avg_many_phoneme['score']:.1f}"))
        render_insight_panel("苦手ポイント", focus_items)

    with improve_col:
        improve_items = []
        if improved_word:
            improve_items.append(("改善幅が大きかった単語", f"{improved_word['label']}\n改善幅: {improved_word['score']:.1f}"))
        if improved_phoneme:
            improve_items.append(("改善幅が大きかった音素", f"{improved_phoneme['label']}\n改善幅: {improved_phoneme['score']:.1f}"))
        if not improve_items:
            improve_items.append(("改善傾向", "改善幅を計算するには各項目で2回以上のサンプルが必要です。"))
        render_insight_panel("改善傾向", improve_items)

    st.markdown("### 日別 average accuracy")
    if daily_trend.empty:
        st.info("日付単位の accuracy 推移を表示できるデータがありません。")
    else:
        fig = px.line(
            daily_trend,
            x="date",
            y="accuracy_avg",
            markers=True,
            hover_data=["samples"],
            title="期間における平均 accuracy の変化（日付単位）",
        )
        fig.update_layout(height=360, showlegend=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 詳細テーブル")
    left, right = st.columns(2)
    with left:
        st.markdown("#### 単語")
        if word_stats.empty:
            st.info("単語の統計データがありません。")
        else:
            st.dataframe(
                word_stats.sort_values(["accuracy_avg", "pron_avg"], ascending=[True, True]),
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.markdown("#### 音素")
        if phoneme_stats.empty:
            st.info("音素の統計データがありません。")
        else:
            st.dataframe(
                phoneme_stats.sort_values(["accuracy_avg", "samples"], ascending=[True, False]),
                use_container_width=True,
                hide_index=True,
            )


def render_statistics_dashboard_v2(word_df: pd.DataFrame, phoneme_df: pd.DataFrame):
    st.markdown("## 統計ダッシュボード")

    word_stats = make_word_stats(word_df)
    phoneme_stats = make_phoneme_stats(phoneme_df)
    word_progress = make_progress_stats(word_df, "word", "accuracy")
    phoneme_progress = make_progress_stats(phoneme_df, "phoneme", "accuracy")
    daily_trend = make_daily_accuracy_trend(word_df, phoneme_df)

    overall_accuracy = word_df["accuracy"].mean() if not word_df.empty and "accuracy" in word_df.columns else phoneme_df["accuracy"].mean()
    overall_pron = word_df["pron"].mean() if not word_df.empty and "pron" in word_df.columns else float("nan")
    total_recordings = int(len(word_df)) if not word_df.empty else 0
    total_practice_days = int(word_df["time"].dt.date.nunique()) if not word_df.empty and "time" in word_df.columns else 0

    hardest_word = pick_focus_item(word_stats, "word", "accuracy_avg")
    hardest_phoneme = pick_focus_item(phoneme_stats, "phoneme", "accuracy_avg")
    improved_word = pick_focus_item(
        word_progress[word_progress["samples"] >= 2], "word", "improvement", min_samples=2, ascending=False
    ) if not word_progress.empty else None
    improved_phoneme = pick_focus_item(
        phoneme_progress[phoneme_progress["samples"] >= 2], "phoneme", "improvement", min_samples=2, ascending=False
    ) if not phoneme_progress.empty else None
    word_sample_threshold = max(2, int(word_stats["samples"].median())) if not word_stats.empty else 2
    phoneme_sample_threshold = max(2, int(phoneme_stats["samples"].median())) if not phoneme_stats.empty else 2
    low_avg_many_word = pick_focus_item(
        word_stats[word_stats["samples"] >= word_sample_threshold], "word", "accuracy_avg", min_samples=word_sample_threshold
    ) if not word_stats.empty else None
    low_avg_many_phoneme = pick_focus_item(
        phoneme_stats[phoneme_stats["samples"] >= phoneme_sample_threshold], "phoneme", "accuracy_avg", min_samples=phoneme_sample_threshold
    ) if not phoneme_stats.empty else None

    summary = st.columns(4)
    summary[0].metric("平均 accuracy", "-" if pd.isna(overall_accuracy) else f"{overall_accuracy:.1f}")
    summary[1].metric("平均 PronScore", "-" if pd.isna(overall_pron) else f"{overall_pron:.1f}")
    summary[2].metric("総録音回数", total_recordings)
    summary[3].metric("総練習日数", total_practice_days)

    focus_items: list[dict[str, str]] = []
    if hardest_word:
        focus_items.append({"項目": "最も苦手な単語", "対象": str(hardest_word["label"]), "値": f"平均 accuracy {hardest_word['score']:.1f}", "回数": f"{hardest_word['samples']}回"})
    if hardest_phoneme:
        focus_items.append({"項目": "最も苦手な音素", "対象": str(hardest_phoneme["label"]), "値": f"平均 accuracy {hardest_phoneme['score']:.1f}", "回数": f"{hardest_phoneme['samples']}回"})
    if low_avg_many_word:
        focus_items.append({"項目": "回数が多いのに平均が低い単語", "対象": str(low_avg_many_word["label"]), "値": f"平均 accuracy {low_avg_many_word['score']:.1f}", "回数": f"{low_avg_many_word['samples']}回"})
    if low_avg_many_phoneme:
        focus_items.append({"項目": "回数が多いのに平均が低い音素", "対象": str(low_avg_many_phoneme["label"]), "値": f"平均 accuracy {low_avg_many_phoneme['score']:.1f}", "回数": f"{low_avg_many_phoneme['samples']}回"})

    improve_items: list[dict[str, str]] = []
    if improved_word:
        improve_items.append({"項目": "改善幅が大きかった単語", "対象": str(improved_word["label"]), "値": f"改善幅 {improved_word['score']:.1f}", "回数": f"{improved_word['samples']}回"})
    if improved_phoneme:
        improve_items.append({"項目": "改善幅が大きかった音素", "対象": str(improved_phoneme["label"]), "値": f"改善幅 {improved_phoneme['score']:.1f}", "回数": f"{improved_phoneme['samples']}回"})
    if not improve_items:
        improve_items.append({"項目": "改善傾向", "対象": "-", "値": "改善幅を計算するには各項目で2回以上のサンプルが必要です。", "回数": "-"})

    focus_col, improve_col = st.columns(2)
    with focus_col:
        render_insight_panel("苦手ポイント", focus_items)
    with improve_col:
        render_insight_panel("改善傾向", improve_items)

    st.markdown("### 日別 average accuracy")
    if daily_trend.empty:
        st.info("日付単位の accuracy 推移を表示できるデータがありません。")
    else:
        fig = px.line(
            daily_trend,
            x="date",
            y="accuracy_avg",
            markers=True,
            hover_data=["samples"],
            title="期間における平均 accuracy の変化（日付単位）",
        )
        fig.update_layout(height=360, showlegend=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 詳細テーブル")
    left, right = st.columns(2)
    with left:
        st.markdown("#### 単語")
        if word_stats.empty:
            st.info("単語の統計データがありません。")
        else:
            st.dataframe(
                word_stats.sort_values(["accuracy_avg", "pron_avg"], ascending=[True, True]),
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.markdown("#### 音素")
        if phoneme_stats.empty:
            st.info("音素の統計データがありません。")
        else:
            st.dataframe(
                phoneme_stats.sort_values(["accuracy_avg", "samples"], ascending=[True, False]),
                use_container_width=True,
                hide_index=True,
            )
    render_confusion_tables(phoneme_df, "statistics")


def main():
    init_state()

    word_df, phoneme_df = load_data()

    word_df = normalize_numeric(
        word_df,
        ["pron", "accuracy", "fluency", "completeness", "f1", "f2", "f3"],
    )
    phoneme_df = normalize_numeric(
        phoneme_df,
        [
            "accuracy", "offset", "duration", "offset_sec", "duration_sec",
            "f1", "f2", "f3", "num_samples",
            "f1_mean", "f2_mean", "f3_mean",
            "f1_min", "f2_min", "f3_min",
            "f1_max", "f2_max", "f3_max",
            "f3_f2_gap_mean", "f3_f2_gap_min", "f3_f2_gap_max",
        ],
    )

    if word_df.empty and phoneme_df.empty:
        st.error("results/history.csv と results/phoneme_history.csv が見つかりません。")
        return

    start_date, end_date = render_global_filters(word_df, phoneme_df)

    word_df = apply_date_filter(word_df, start_date, end_date)
    phoneme_df = apply_date_filter(phoneme_df, start_date, end_date)
    phoneme_df = enrich_phoneme_misrecognitions(phoneme_df)
    word_df = attach_recording_misrecognitions(word_df, phoneme_df)

    st.markdown("---")

    page = st.radio(
        "ページ",
        ["単語単位", "音素単位", "統計"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "単語単位":
        render_word_view(word_df, phoneme_df)
    elif page == "音素単位":
        render_phoneme_view(phoneme_df)
    else:
        render_statistics_dashboard_v2(word_df, phoneme_df)


if __name__ == "__main__":
    main()
