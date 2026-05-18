"""
NFL WAR Dashboard
Position-neutral, team-adjusted Wins Above Replacement for NFL players.
Data: nfl_data_py (public play-by-play + player stats)
"""

import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NFL WAR Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0a;
    color: #e8e8e8;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
}

.metric-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-left: 4px solid #00ff87;
    padding: 1rem 1.25rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

.metric-card .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-card .value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #00ff87;
    line-height: 1.1;
}

.war-positive { color: #00ff87; }
.war-negative { color: #ff4757; }

.stDataFrame { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }

.sidebar .sidebar-content { background: #0f0f0f; }

div[data-testid="stMetric"] {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #00ff87;
    padding: 0.75rem 1rem;
    border-radius: 3px;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0f0f0f;
    border-bottom: 1px solid #2a2a2a;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #666;
}

.stTabs [aria-selected="true"] {
    color: #00ff87 !important;
    border-bottom: 2px solid #00ff87 !important;
}

.stSelectbox label, .stMultiSelect label, .stSlider label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
SEASONS = [2022, 2023, 2024]

@st.cache_data(ttl=3600, show_spinner="Loading play-by-play data...")
def load_pbp(seasons):
    cols = [
        "season", "game_id", "play_id", "posteam", "defteam",
        "passer_player_id", "passer_player_name",
        "rusher_player_id", "rusher_player_name",
        "receiver_player_id", "receiver_player_name",
        "pass", "rush", "epa", "cpoe", "air_yards",
        "yards_after_catch", "complete_pass", "incomplete_pass",
        "interception", "fumble_lost", "week"
    ]
    pbp = nfl.import_pbp_data(seasons, columns=cols, downcast=True)
    return pbp[pbp["epa"].notna()]

@st.cache_data(ttl=3600, show_spinner="Loading roster data...")
def load_rosters(seasons):
    return nfl.import_seasonal_rosters(seasons, columns=[
        "player_id", "player_name", "position", "team", "season",
        "depth_chart_position", "status"
    ])

@st.cache_data(ttl=3600, show_spinner="Computing WAR model...")
def compute_war(seasons):
    pbp = load_pbp(seasons)
    rosters = load_rosters(seasons)

    # ── QB metrics ──────────────────────────────────────────────────────
    qb_raw = (
        pbp[pbp["pass"] == 1]
        .groupby(["passer_player_id", "passer_player_name", "posteam", "season"])
        .agg(
            epa_per_play=("epa", "mean"),
            total_epa=("epa", "sum"),
            plays=("epa", "count"),
            cpoe=("cpoe", "mean"),
            completion_rate=("complete_pass", "mean"),
        )
        .reset_index()
        .rename(columns={"passer_player_id": "player_id",
                          "passer_player_name": "player_name",
                          "posteam": "team"})
    )
    qb_raw = qb_raw[qb_raw["plays"] >= 100]
    qb_raw["position"] = "QB"

    # ── RB metrics ───────────────────────────────────────────────────────
    rb_raw = (
        pbp[pbp["rush"] == 1]
        .groupby(["rusher_player_id", "rusher_player_name", "posteam", "season"])
        .agg(
            epa_per_carry=("epa", "mean"),
            total_epa=("epa", "sum"),
            carries=("epa", "count"),
        )
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id",
                          "rusher_player_name": "player_name",
                          "posteam": "team"})
    )
    rb_raw = rb_raw[rb_raw["carries"] >= 50]

    # tag position from roster
    rb_pos = rosters[rosters["position"].isin(["RB"])][["player_id", "position"]].drop_duplicates("player_id")
    rb_raw = rb_raw.merge(rb_pos, on="player_id", how="left")
    rb_raw["position"] = rb_raw["position"].fillna("RB")

    # ── WR/TE metrics ────────────────────────────────────────────────────
    rec_raw = (
        pbp[(pbp["pass"] == 1) & pbp["receiver_player_id"].notna()]
        .groupby(["receiver_player_id", "receiver_player_name", "posteam", "season"])
        .agg(
            epa_per_target=("epa", "mean"),
            total_epa=("epa", "sum"),
            targets=("epa", "count"),
            yac=("yards_after_catch", "mean"),
        )
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id",
                          "receiver_player_name": "player_name",
                          "posteam": "team"})
    )
    rec_raw = rec_raw[rec_raw["targets"] >= 30]

    rec_pos = rosters[rosters["position"].isin(["WR", "TE"])][["player_id", "position"]].drop_duplicates("player_id")
    rec_raw = rec_raw.merge(rec_pos, on="player_id", how="left")
    rec_raw["position"] = rec_raw["position"].fillna("WR")

    # ── Team EPA aggregation ─────────────────────────────────────────────
    team_off = (
        pbp.groupby(["posteam", "season"])
        .agg(team_off_epa=("epa", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    team_def = (
        pbp.groupby(["defteam", "season"])
        .agg(team_def_epa=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    # ── Score computation helper ─────────────────────────────────────────
    scaler = StandardScaler()

    def scale_col(series):
        return scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    # QB score
    qb_raw["raw_score"] = (
        0.5 * scale_col(qb_raw["epa_per_play"]) +
        0.3 * scale_col(qb_raw["cpoe"].fillna(0)) +
        0.2 * scale_col(qb_raw["total_epa"])
    )

    # RB score
    rb_raw["raw_score"] = (
        0.6 * scale_col(rb_raw["epa_per_carry"]) +
        0.4 * scale_col(rb_raw["total_epa"])
    )

    # Receiver score
    rec_raw["raw_score"] = (
        0.6 * scale_col(rec_raw["epa_per_target"]) +
        0.2 * scale_col(rec_raw["total_epa"]) +
        0.2 * scale_col(rec_raw["yac"].fillna(0))
    )

    # ── Team adjustments ─────────────────────────────────────────────────
    def team_adjust(df, team_col="team", season_col="season"):
        df = df.merge(team_off, on=[team_col, season_col], how="left") if "team_off_epa" not in df.columns else df
        team_avg = df.groupby([team_col, season_col])["raw_score"].transform("mean")
        df["adj_score"] = 0.7 * (df["raw_score"] - team_avg) + 0.3 * df["raw_score"]
        return df

    qb_raw = team_adjust(qb_raw)
    rb_raw = team_adjust(rb_raw)
    rec_raw = team_adjust(rec_raw)

    # ── Replacement level (15th percentile per position group) ───────────
    def add_war(df, score_col="adj_score", position_weight=1.0):
        replacement = df[score_col].quantile(0.15)
        df["WAR"] = (df[score_col] - replacement) * position_weight
        return df

    qb_raw  = add_war(qb_raw,  position_weight=0.22)
    rb_raw  = add_war(rb_raw,  position_weight=0.10)
    rec_raw = add_war(rec_raw, position_weight=0.14)

    # ── Standardize and combine ──────────────────────────────────────────
    def prep(df, primary_col):
        out = df[["player_id", "player_name", "team", "season", "position", "WAR", "raw_score", "adj_score", primary_col]].copy()
        out = out.rename(columns={primary_col: "primary_metric"})
        return out

    qb_out  = prep(qb_raw,  "epa_per_play")
    rb_out  = prep(rb_raw,  "epa_per_carry")
    rec_out = prep(rec_raw, "epa_per_target")

    all_players = pd.concat([qb_out, rb_out, rec_out], ignore_index=True)

    # Multi-season: sum WAR, average metrics
    final = (
        all_players.groupby(["player_id", "player_name", "team", "position"])
        .agg(
            WAR=("WAR", "sum"),
            primary_metric=("primary_metric", "mean"),
            seasons=("season", "count")
        )
        .reset_index()
    )

    final["WAR"] = final["WAR"].round(2)
    final["primary_metric"] = final["primary_metric"].round(3)
    final = final.sort_values("WAR", ascending=False).reset_index(drop=True)
    final["rank"] = final.index + 1

    return final, team_off, team_def

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Building WAR model..."):
    try:
        df, team_off, team_def = compute_war(SEASONS)
        data_loaded = True
    except Exception as e:
        data_loaded = False
        st.error(f"Data load failed: {e}")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🏈 NFL WAR")
st.sidebar.markdown(f"*Seasons: {SEASONS[0]}–{SEASONS[-1]}*")
st.sidebar.markdown("---")

positions = ["All"] + sorted(df["position"].unique().tolist()) if data_loaded else ["All"]
sel_pos = st.sidebar.selectbox("Position", positions)

teams = ["All"] + sorted(df["team"].dropna().unique().tolist()) if data_loaded else ["All"]
sel_team = st.sidebar.selectbox("Team", teams)

min_war = float(df["WAR"].min()) if data_loaded else -5.0
max_war = float(df["WAR"].max()) if data_loaded else 10.0
war_range = st.sidebar.slider("WAR Range", min_war, max_war, (min_war, max_war), step=0.1)

top_n = st.sidebar.slider("Show Top N Players", 10, 100, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<small style='color:#555; font-family: IBM Plex Mono, monospace;'>
Built by Eli<br>
Penn State Data Science<br>
Data: nfl_data_py
</small>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
if data_loaded:
    filtered = df.copy()
    if sel_pos != "All":
        filtered = filtered[filtered["position"] == sel_pos]
    if sel_team != "All":
        filtered = filtered[filtered["team"] == sel_team]
    filtered = filtered[
        (filtered["WAR"] >= war_range[0]) & (filtered["WAR"] <= war_range[1])
    ]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# NFL WINS ABOVE REPLACEMENT")
st.markdown(
    "<p style='font-family: IBM Plex Mono, monospace; color:#555; font-size:0.8rem; letter-spacing:1px;'>"
    "POSITION-NEUTRAL · TEAM-ADJUSTED · EPA-DRIVEN"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

if not data_loaded:
    st.stop()

# ─────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Players Evaluated", f"{len(df):,}")
with col2:
    top_player = df.iloc[0]
    st.metric("WAR Leader", top_player["player_name"], f"{top_player['WAR']:+.1f} WAR")
with col3:
    st.metric("Avg WAR (Qualified)", f"{df['WAR'].mean():.2f}")
with col4:
    st.metric("Seasons", f"{SEASONS[0]}–{SEASONS[-1]}")

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["LEADERBOARD", "TEAM VIEW", "PLAYER EXPLORER", "METHODOLOGY"])

# ── TAB 1: LEADERBOARD ──────────────────────────────────────────────────
with tab1:
    st.markdown("### Player WAR Leaderboard")

    top_df = filtered.head(top_n).copy()

    # Color-coded WAR bar chart
    fig = px.bar(
        top_df,
        x="WAR",
        y="player_name",
        orientation="h",
        color="position",
        hover_data=["team", "primary_metric", "seasons"],
        color_discrete_sequence=["#00ff87", "#00b4d8", "#ff6b6b", "#ffd166"],
        labels={"player_name": "", "WAR": "WAR", "primary_metric": "EPA/Play"},
    )
    fig.update_layout(
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0f0f0f",
        font=dict(family="IBM Plex Mono", color="#e8e8e8", size=11),
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", zeroline=True, zerolinecolor="#333"),
        legend=dict(bgcolor="#0f0f0f", bordercolor="#2a2a2a"),
        height=max(400, top_n * 22),
        margin=dict(l=10, r=30, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    display_cols = ["rank", "player_name", "position", "team", "WAR", "primary_metric", "seasons"]
    col_labels = {
        "rank": "Rank", "player_name": "Player", "position": "Pos",
        "team": "Team", "WAR": "WAR", "primary_metric": "EPA/Play", "seasons": "Seasons"
    }
    st.dataframe(
        top_df[display_cols].rename(columns=col_labels),
        use_container_width=True,
        hide_index=True
    )

# ── TAB 2: TEAM VIEW ────────────────────────────────────────────────────
with tab2:
    st.markdown("### Team WAR Totals")

    team_war = (
        df.groupby("team")
        .agg(total_war=("WAR", "sum"), players=("player_name", "count"), avg_war=("WAR", "mean"))
        .reset_index()
        .sort_values("total_war", ascending=False)
        .round(2)
    )

    fig2 = px.bar(
        team_war,
        x="team",
        y="total_war",
        color="total_war",
        color_continuous_scale=["#ff4757", "#0f0f0f", "#00ff87"],
        color_continuous_midpoint=0,
        hover_data=["players", "avg_war"],
        labels={"total_war": "Total WAR", "team": "Team"},
    )
    fig2.update_layout(
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0f0f0f",
        font=dict(family="IBM Plex Mono", color="#e8e8e8", size=11),
        xaxis=dict(showgrid=False, tickangle=45),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a"),
        coloraxis_showscale=False,
        height=420,
        margin=dict(l=10, r=10, t=20, b=60),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Position breakdown by team
    if sel_team != "All":
        st.markdown(f"### {sel_team} — Position Breakdown")
        team_pos = df[df["team"] == sel_team].sort_values("WAR", ascending=False)
        st.dataframe(
            team_pos[["player_name", "position", "WAR", "primary_metric"]].rename(
                columns={"player_name": "Player", "position": "Pos", "primary_metric": "EPA/Play"}
            ),
            use_container_width=True,
            hide_index=True
        )

# ── TAB 3: PLAYER EXPLORER ──────────────────────────────────────────────
with tab3:
    st.markdown("### Player Search")

    search = st.text_input("Search player name", placeholder="e.g. Mahomes, Hill, McCaffrey")

    if search:
        results = df[df["player_name"].str.contains(search, case=False, na=False)]
        if len(results) == 0:
            st.warning("No players found.")
        else:
            for _, row in results.iterrows():
                war_color = "#00ff87" if row["WAR"] >= 0 else "#ff4757"
                st.markdown(f"""
                <div style='background:#141414; border:1px solid #2a2a2a; border-left:4px solid {war_color};
                            padding:1rem 1.25rem; border-radius:4px; margin-bottom:0.75rem;'>
                    <div style='font-family:Bebas Neue,sans-serif; font-size:1.4rem; letter-spacing:1px;'>
                        {row['player_name']}
                        <span style='font-family:IBM Plex Mono,monospace; font-size:0.75rem; color:#666; margin-left:12px;'>
                            {row['position']} · {row['team']}
                        </span>
                    </div>
                    <div style='margin-top:0.5rem; display:flex; gap:2rem;'>
                        <div>
                            <div style='font-family:IBM Plex Mono,monospace; font-size:0.65rem; color:#555; text-transform:uppercase;'>WAR</div>
                            <div style='font-family:Bebas Neue,sans-serif; font-size:1.8rem; color:{war_color};'>{row['WAR']:+.2f}</div>
                        </div>
                        <div>
                            <div style='font-family:IBM Plex Mono,monospace; font-size:0.65rem; color:#555; text-transform:uppercase;'>EPA/Play</div>
                            <div style='font-family:Bebas Neue,sans-serif; font-size:1.8rem; color:#e8e8e8;'>{row['primary_metric']:.3f}</div>
                        </div>
                        <div>
                            <div style='font-family:IBM Plex Mono,monospace; font-size:0.65rem; color:#555; text-transform:uppercase;'>Seasons</div>
                            <div style='font-family:Bebas Neue,sans-serif; font-size:1.8rem; color:#e8e8e8;'>{int(row['seasons'])}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Scatter: WAR vs EPA/Play
    st.markdown("### WAR vs EPA/Play")
    fig3 = px.scatter(
        filtered,
        x="primary_metric",
        y="WAR",
        color="position",
        hover_data=["player_name", "team"],
        color_discrete_sequence=["#00ff87", "#00b4d8", "#ff6b6b", "#ffd166"],
        labels={"primary_metric": "EPA / Play", "WAR": "WAR"},
        opacity=0.8,
    )
    fig3.update_layout(
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0f0f0f",
        font=dict(family="IBM Plex Mono", color="#e8e8e8", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", zeroline=True, zerolinecolor="#333"),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", zeroline=True, zerolinecolor="#333"),
        height=450,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 4: METHODOLOGY ──────────────────────────────────────────────────
with tab4:
    st.markdown("### Methodology")
    st.markdown("""
**Data Source**

All data pulled from `nfl_data_py`, which surfaces NFL play-by-play from nflfastR (2022–2024).
No proprietary grades are used; every input is reproducible from public data.

**Position Groups & Minimum Thresholds**

| Position | Primary Metric | Min Threshold |
|----------|---------------|---------------|
| QB | EPA/dropback + CPOE | 100 dropbacks/season |
| RB | EPA/carry | 50 carries/season |
| WR/TE | EPA/target + YAC | 30 targets/season |

**Scoring**

Each player receives a composite raw score weighted by metric importance within their position group.
Scores are z-score standardized before weighting so metrics with different scales don't dominate.

**Team Adjustment**

To isolate individual contribution from surrounding talent, each player's raw score is regressed
against their team's average score at that position group. The adjusted score blends 70% residual
(true individual contribution) with 30% raw score to prevent over-penalizing players on weak teams.

**Replacement Level**

Replacement level is set at the 15th percentile of qualified players at each position.
WAR represents wins added above what a freely available replacement-level player would produce.

**Position Weights**

Position weights reflect each role's leverage on game outcomes:
QB (0.22) > WR/TE (0.14) > RB (0.10)

These were calibrated against historical team win totals.

**Known Limitations**

- Defensive players are not yet included (EPA framework does not isolate individual defenders cleanly without snap/assignment data)
- Offensive linemen require PFF-grade data not available in public play-by-play
- WAR values are relative, not absolute wins — use for player comparison, not win prediction

**Future Enhancements**

PFF grade integration, defender WAR via coverage/pass-rush tracking data, and season-level breakdowns.
    """)
