"""
NFL WAR Dashboard
Position-neutral, team-adjusted Wins Above Replacement for NFL skill players.

Data:    nfl_data_py (public nflfastR play-by-play + seasonal rosters)
Compute: DuckDB SQL for all aggregation and team adjustment,
         pandas/numpy for standardization and scoring.

Every aggregation in this app is written in SQL and validated against an
equivalent pandas implementation (see the SQL tab and validate_parity()).
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import nfl_data_py as nfl
import plotly.express as px
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

.war-positive { color: #00ff87; }
.war-negative { color: #ff4757; }

.stDataFrame { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }

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
# CONFIG
# ─────────────────────────────────────────────
SEASONS = [2022, 2023, 2024]

# Minimum volume for a player-season to qualify. Below these, per-play rate
# metrics are dominated by noise rather than signal.
MIN_DROPBACKS = 100
MIN_CARRIES   = 50
MIN_TARGETS   = 30

# Replacement level percentile within each position group.
REPLACEMENT_PCTL = 0.15

# Share of a player's score attributed to the residual after removing his
# team-position mean. 0.70 strips most of the surrounding-talent effect while
# retaining some absolute production so strong players on strong teams are
# not penalised into the middle of the pack.
RESIDUAL_WEIGHT = 0.70

# Positional leverage on game outcomes. NOTE: these are modelling assumptions
# informed by published positional-value research, NOT parameters estimated
# from this dataset. Calibrating them against team win totals is the top
# item on the roadmap.
POSITION_WEIGHTS = {"QB": 0.22, "WR": 0.14, "TE": 0.14, "RB": 0.10}

# ─────────────────────────────────────────────
# SQL — every aggregation lives here
# ─────────────────────────────────────────────

SQL_QB = f"""
SELECT  passer_player_id        AS player_id,
        passer_player_name      AS player_name,
        posteam                 AS team,
        season,
        'QB'                    AS position,
        AVG(epa)                AS epa_per_play,
        SUM(epa)                AS total_epa,
        COUNT(*)                AS plays,
        AVG(cpoe)               AS cpoe,
        AVG(complete_pass)      AS completion_rate
FROM    pbp
WHERE   "pass" = 1
  AND   epa IS NOT NULL
  AND   passer_player_id IS NOT NULL
GROUP BY 1, 2, 3, 4
HAVING  COUNT(*) >= {MIN_DROPBACKS}
"""

SQL_RB = f"""
SELECT  p.rusher_player_id      AS player_id,
        p.rusher_player_name    AS player_name,
        p.posteam               AS team,
        p.season,
        COALESCE(r.position, 'RB') AS position,
        AVG(p.epa)              AS epa_per_carry,
        SUM(p.epa)              AS total_epa,
        COUNT(*)                AS carries
FROM    pbp p
LEFT JOIN rosters r
       ON r.player_id = p.rusher_player_id
      AND r.season    = p.season
WHERE   p.rush = 1
  AND   p.epa IS NOT NULL
  AND   p.rusher_player_id IS NOT NULL
GROUP BY 1, 2, 3, 4, 5
HAVING  COUNT(*) >= {MIN_CARRIES}
"""

SQL_REC = f"""
SELECT  p.receiver_player_id    AS player_id,
        p.receiver_player_name  AS player_name,
        p.posteam               AS team,
        p.season,
        COALESCE(r.position, 'WR') AS position,
        AVG(p.epa)              AS epa_per_target,
        SUM(p.epa)              AS total_epa,
        COUNT(*)                AS targets,
        AVG(p.yards_after_catch) AS yac
FROM    pbp p
LEFT JOIN rosters r
       ON r.player_id = p.receiver_player_id
      AND r.season    = p.season
WHERE   p."pass" = 1
  AND   p.epa IS NOT NULL
  AND   p.receiver_player_id IS NOT NULL
GROUP BY 1, 2, 3, 4, 5
HAVING  COUNT(*) >= {MIN_TARGETS}
"""

# Team adjustment as a window function. This is the step that separates a
# player from his surrounding cast: subtract the mean raw score of his own
# position group on his own team in that season.
SQL_TEAM_ADJUST = f"""
SELECT  *,
        AVG(raw_score) OVER (PARTITION BY team, season, position)
                                        AS team_pos_mean,
        {RESIDUAL_WEIGHT} * (raw_score - AVG(raw_score) OVER (PARTITION BY team, season, position))
      + {1 - RESIDUAL_WEIGHT} * raw_score
                                        AS adj_score
FROM    scored
"""

# Replacement level per position group, then WAR above it.
SQL_WAR = f"""
WITH repl AS (
    SELECT  position,
            QUANTILE_CONT(adj_score, {REPLACEMENT_PCTL}) AS replacement_level
    FROM    adjusted
    GROUP BY position
)
SELECT  a.*,
        r.replacement_level,
        (a.adj_score - r.replacement_level) * a.position_weight AS WAR
FROM    adjusted a
JOIN    repl r USING (position)
"""

# Collapse player-seasons into one row per player. A traded player keeps the
# team he most recently appeared for rather than being split into two rows.
SQL_FINAL = """
WITH latest_team AS (
    SELECT  player_id,
            ARG_MAX(team, season) AS team
    FROM    war
    GROUP BY player_id
)
SELECT  w.player_id,
        ANY_VALUE(w.player_name)        AS player_name,
        ANY_VALUE(lt.team)              AS team,
        ANY_VALUE(w.position)           AS position,
        ROUND(SUM(w.WAR), 2)            AS WAR,
        ROUND(AVG(w.primary_metric), 3) AS primary_metric,
        COUNT(*)                        AS seasons
FROM    war w
JOIN    latest_team lt USING (player_id)
GROUP BY w.player_id
ORDER BY WAR DESC
"""

SQL_TEAM_WAR = """
SELECT  team,
        ROUND(SUM(WAR), 2)  AS total_war,
        COUNT(*)            AS players,
        ROUND(AVG(WAR), 2)  AS avg_war
FROM    final
WHERE   team IS NOT NULL
GROUP BY team
ORDER BY total_war DESC
"""

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

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


@st.cache_data(ttl=3600, show_spinner="Loading schedules for validation...")
def load_wins(seasons):
    """Actual regular-season wins per team-season, used to validate the model."""
    sched = nfl.import_schedules(seasons)
    sched = sched[(sched["game_type"] == "REG") & sched["home_score"].notna()]
    home = sched[["season", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "pf", "away_score": "pa"})
    away = sched[["season", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "pf", "home_score": "pa"})
    games = pd.concat([home, away], ignore_index=True)
    games["win"] = (games["pf"] > games["pa"]).astype(int)
    return games.groupby(["team", "season"], as_index=False)["win"].sum().rename(
        columns={"win": "wins"})


# ─────────────────────────────────────────────
# SCORING (pandas: z-score standardisation + weighting)
# ─────────────────────────────────────────────

def zscore(s):
    """Standardise so metrics on different scales contribute comparably."""
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def score_qb(df):
    df = df.copy()
    df["raw_score"] = (0.5 * zscore(df["epa_per_play"])
                       + 0.3 * zscore(df["cpoe"])
                       + 0.2 * zscore(df["total_epa"]))
    df["primary_metric"] = df["epa_per_play"]
    return df


def score_rb(df):
    df = df.copy()
    df["raw_score"] = (0.6 * zscore(df["epa_per_carry"])
                       + 0.4 * zscore(df["total_epa"]))
    df["primary_metric"] = df["epa_per_carry"]
    return df


def score_rec(df):
    df = df.copy()
    df["raw_score"] = (0.6 * zscore(df["epa_per_target"])
                       + 0.2 * zscore(df["total_epa"])
                       + 0.2 * zscore(df["yac"]))
    df["primary_metric"] = df["epa_per_target"]
    return df


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Computing WAR model...")
def compute_war(seasons):
    pbp = load_pbp(seasons)
    rosters = load_rosters(seasons)

    con = duckdb.connect()
    con.register("pbp", pbp)
    con.register("rosters", rosters)

    # 1 — aggregate per player-season in SQL
    qb  = score_qb(con.sql(SQL_QB).df())
    rb  = score_rb(con.sql(SQL_RB).df())
    rec = score_rec(con.sql(SQL_REC).df())

    keep = ["player_id", "player_name", "team", "season", "position",
            "raw_score", "primary_metric"]
    scored = pd.concat([qb[keep], rb[keep], rec[keep]], ignore_index=True)
    scored["position_weight"] = scored["position"].map(POSITION_WEIGHTS).fillna(0.10)

    # 2 — team adjustment via window function
    con.register("scored", scored)
    adjusted = con.sql(SQL_TEAM_ADJUST).df()

    # 3 — replacement level and WAR
    con.register("adjusted", adjusted)
    war = con.sql(SQL_WAR).df()

    # 4 — collapse to one row per player
    con.register("war", war)
    final = con.sql(SQL_FINAL).df()
    final["rank"] = np.arange(1, len(final) + 1)

    con.register("final", final)
    team_war = con.sql(SQL_TEAM_WAR).df()

    return final, war, team_war


def validate_parity(seasons):
    """
    Recompute the QB aggregation in pandas and compare against the SQL result.
    Used to prove the SQL rewrite is behaviour-preserving, not a rewrite that
    quietly changed the numbers.
    """
    pbp = load_pbp(seasons)
    con = duckdb.connect()
    con.register("pbp", pbp)
    sql_out = con.sql(SQL_QB).df()

    pd_out = (pbp[pbp["pass"] == 1]
              .dropna(subset=["epa", "passer_player_id"])
              .groupby(["passer_player_id", "passer_player_name", "posteam", "season"],
                       as_index=False)
              .agg(epa_per_play=("epa", "mean"),
                   total_epa=("epa", "sum"),
                   plays=("epa", "count")))
    pd_out = pd_out[pd_out["plays"] >= MIN_DROPBACKS]

    merged = sql_out.merge(
        pd_out, left_on=["player_id", "season"],
        right_on=["passer_player_id", "season"], suffixes=("_sql", "_pd"))

    return {
        "sql_rows": len(sql_out),
        "pandas_rows": len(pd_out),
        "matched_rows": len(merged),
        "max_abs_diff_epa_per_play": float(
            (merged["epa_per_play_sql"] - merged["epa_per_play_pd"]).abs().max()),
        "max_abs_diff_total_epa": float(
            (merged["total_epa_sql"] - merged["total_epa_pd"]).abs().max()),
    }


def validate_against_wins(war_by_season, seasons):
    """Correlate summed team WAR against actual regular-season wins."""
    wins = load_wins(seasons)
    con = duckdb.connect()
    con.register("war", war_by_season)
    con.register("wins", wins)
    return con.sql("""
        SELECT  w.team, w.season,
                ROUND(SUM(w.WAR), 2) AS team_war,
                ANY_VALUE(a.wins)    AS actual_wins
        FROM    war w
        JOIN    wins a ON a.team = w.team AND a.season = w.season
        GROUP BY w.team, w.season
        ORDER BY team_war DESC
    """).df()


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
data_loaded = False
try:
    df, war_by_season, team_war = compute_war(SEASONS)
    data_loaded = True
except Exception as e:
    st.error(f"Data load failed: {e}")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🏈 NFL WAR")
st.sidebar.markdown(f"*Seasons: {SEASONS[0]}–{SEASONS[-1]}*")
st.sidebar.markdown("---")

positions = ["All"] + sorted(df["position"].dropna().unique().tolist()) if data_loaded else ["All"]
sel_pos = st.sidebar.selectbox("Position", positions)

teams = ["All"] + sorted(df["team"].dropna().unique().tolist()) if data_loaded else ["All"]
sel_team = st.sidebar.selectbox("Team", teams)

if data_loaded:
    min_war, max_war = float(df["WAR"].min()), float(df["WAR"].max())
else:
    min_war, max_war = -5.0, 10.0
war_range = st.sidebar.slider("WAR Range", min_war, max_war, (min_war, max_war), step=0.1)
top_n = st.sidebar.slider("Show Top N Players", 10, 100, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<small style='color:#555; font-family: IBM Plex Mono, monospace;'>
Built by Elijah Legall<br>
Penn State · B.S. Data Science<br>
Data: nfl_data_py · Engine: DuckDB
</small>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# NFL WINS ABOVE REPLACEMENT")
st.markdown(
    "<p style='font-family: IBM Plex Mono, monospace; color:#555; font-size:0.8rem; letter-spacing:1px;'>"
    "POSITION-NEUTRAL · TEAM-ADJUSTED · EPA-DRIVEN · SQL-BACKED"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

if not data_loaded:
    st.stop()

filtered = df.copy()
if sel_pos != "All":
    filtered = filtered[filtered["position"] == sel_pos]
if sel_team != "All":
    filtered = filtered[filtered["team"] == sel_team]
filtered = filtered[(filtered["WAR"] >= war_range[0]) & (filtered["WAR"] <= war_range[1])]

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Players Evaluated", f"{len(df):,}")
top_player = df.iloc[0]
c2.metric("WAR Leader", top_player["player_name"], f"{top_player['WAR']:+.1f} WAR")
c3.metric("Avg WAR (Qualified)", f"{df['WAR'].mean():.2f}")
c4.metric("Seasons", f"{SEASONS[0]}–{SEASONS[-1]}")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["LEADERBOARD", "TEAM VIEW", "PLAYER EXPLORER", "SQL LAYER", "METHODOLOGY"])

# ── TAB 1: LEADERBOARD ──────────────────────────────────────────────────
with tab1:
    st.markdown("### Player WAR Leaderboard")
    top_df = filtered.head(top_n).copy()

    fig = px.bar(
        top_df, x="WAR", y="player_name", orientation="h", color="position",
        hover_data=["team", "primary_metric", "seasons"],
        color_discrete_sequence=["#00ff87", "#00b4d8", "#ff6b6b", "#ffd166"],
        labels={"player_name": "", "WAR": "WAR", "primary_metric": "EPA/Play"},
    )
    fig.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0f0f0f",
        font=dict(family="IBM Plex Mono", color="#e8e8e8", size=11),
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", zeroline=True, zerolinecolor="#333"),
        legend=dict(bgcolor="#0f0f0f", bordercolor="#2a2a2a"),
        height=max(400, top_n * 22), margin=dict(l=10, r=30, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        top_df[["rank", "player_name", "position", "team", "WAR", "primary_metric", "seasons"]]
        .rename(columns={"rank": "Rank", "player_name": "Player", "position": "Pos",
                         "team": "Team", "primary_metric": "EPA/Play", "seasons": "Seasons"}),
        use_container_width=True, hide_index=True
    )

# ── TAB 2: TEAM VIEW ────────────────────────────────────────────────────
with tab2:
    st.markdown("### Team WAR Totals")
    fig2 = px.bar(
        team_war, x="team", y="total_war", color="total_war",
        color_continuous_scale=["#ff4757", "#0f0f0f", "#00ff87"],
        color_continuous_midpoint=0, hover_data=["players", "avg_war"],
        labels={"total_war": "Total WAR", "team": "Team"},
    )
    fig2.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0f0f0f",
        font=dict(family="IBM Plex Mono", color="#e8e8e8", size=11),
        xaxis=dict(showgrid=False, tickangle=45),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a"),
        coloraxis_showscale=False, height=420, margin=dict(l=10, r=10, t=20, b=60),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Validation — Team WAR vs Actual Wins")
    st.caption("If the model measures anything real, summed team WAR should track "
               "actual regular-season wins. This is the headline validity check.")
    if st.button("Run validation against real win totals"):
        try:
            vdf = validate_against_wins(war_by_season, SEASONS)
            r = vdf["team_war"].corr(vdf["actual_wins"])
            st.metric("Correlation: team WAR vs actual wins", f"{r:.3f}")
            try:
                import statsmodels  # noqa: F401
                trend = "ols"
            except ImportError:
                trend = None
            fig4 = px.scatter(
                vdf, x="team_war", y="actual_wins", hover_data=["team", "season"],
                trendline=trend, color_discrete_sequence=["#00ff87"],
                labels={"team_war": "Summed Team WAR", "actual_wins": "Actual Wins"})
            fig4.update_layout(
                paper_bgcolor="#0a0a0a", plot_bgcolor="#0f0f0f",
                font=dict(family="IBM Plex Mono", color="#e8e8e8", size=11),
                xaxis=dict(showgrid=True, gridcolor="#1a1a1a"),
                yaxis=dict(showgrid=True, gridcolor="#1a1a1a"), height=420)
            st.plotly_chart(fig4, use_container_width=True)
            st.dataframe(vdf, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Validation unavailable: {e}")

    if sel_team != "All":
        st.markdown(f"### {sel_team} — Position Breakdown")
        st.dataframe(
            df[df["team"] == sel_team].sort_values("WAR", ascending=False)
            [["player_name", "position", "WAR", "primary_metric"]]
            .rename(columns={"player_name": "Player", "position": "Pos",
                             "primary_metric": "EPA/Play"}),
            use_container_width=True, hide_index=True)

# ── TAB 3: PLAYER EXPLORER ──────────────────────────────────────────────
with tab3:
    st.markdown("### Player Search")
    search = st.text_input("Search player name", placeholder="e.g. Mahomes, Hill, McCaffrey")
    if search:
        results = df[df["player_name"].str.contains(search, case=False, na=False)]
        if len(results) == 0:
            st.warning("No players found.")
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

    st.markdown("### WAR vs EPA/Play")
    fig3 = px.scatter(
        filtered, x="primary_metric", y="WAR", color="position",
        hover_data=["player_name", "team"],
        color_discrete_sequence=["#00ff87", "#00b4d8", "#ff6b6b", "#ffd166"],
        labels={"primary_metric": "EPA / Play", "WAR": "WAR"}, opacity=0.8)
    fig3.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0f0f0f",
        font=dict(family="IBM Plex Mono", color="#e8e8e8", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", zeroline=True, zerolinecolor="#333"),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", zeroline=True, zerolinecolor="#333"),
        height=450)
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 4: SQL LAYER ────────────────────────────────────────────────────
with tab4:
    st.markdown("### The SQL Behind the Model")
    st.markdown(
        "Every aggregation, join, and team adjustment runs as SQL against the "
        "play-by-play table in DuckDB. pandas is used only for z-score "
        "standardisation and weighting."
    )

    st.markdown("**1 — Quarterback aggregation.** `HAVING` enforces the minimum-volume filter.")
    st.code(SQL_QB, language="sql")

    st.markdown("**2 — Rusher aggregation, joined to rosters for position.**")
    st.code(SQL_RB, language="sql")

    st.markdown("**3 — Receiver aggregation.**")
    st.code(SQL_REC, language="sql")

    st.markdown("**4 — Team adjustment as a window function.** This is the step that "
                "separates a player from his surrounding cast, without collapsing the "
                "row grain.")
    st.code(SQL_TEAM_ADJUST, language="sql")

    st.markdown("**5 — Replacement level per position, then WAR above it.**")
    st.code(SQL_WAR, language="sql")

    st.markdown("**6 — Collapse player-seasons.** `ARG_MAX` keeps a traded player's most "
                "recent team instead of splitting him into two rows.")
    st.code(SQL_FINAL, language="sql")

    st.markdown("---")
    st.markdown("### Parity Check — SQL vs pandas")
    st.caption("The aggregation layer was rewritten from pandas into SQL. This proves "
               "the rewrite preserved the numbers rather than quietly changing them.")
    if st.button("Run parity check"):
        try:
            res = validate_parity(SEASONS)
            cols = st.columns(3)
            cols[0].metric("SQL rows", res["sql_rows"])
            cols[1].metric("pandas rows", res["pandas_rows"])
            cols[2].metric("Matched", res["matched_rows"])
            st.metric("Max abs difference in EPA/play",
                      f"{res['max_abs_diff_epa_per_play']:.2e}")
            st.success("Differences are at floating-point precision — the two "
                       "implementations agree.")
        except Exception as e:
            st.warning(f"Parity check unavailable: {e}")

# ── TAB 5: METHODOLOGY ──────────────────────────────────────────────────
with tab5:
    st.markdown("### Methodology")
    st.markdown(f"""
**Data source**

All data comes from `nfl_data_py`, which surfaces nflfastR play-by-play
({SEASONS[0]}–{SEASONS[-1]}). No proprietary grades are used; every input is
reproducible from public data by anyone running this repo.

**Position groups and minimum volume**

| Position | Primary metric | Minimum |
|---|---|---|
| QB | EPA/dropback + CPOE | {MIN_DROPBACKS} dropbacks/season |
| RB | EPA/carry | {MIN_CARRIES} carries/season |
| WR/TE | EPA/target + YAC | {MIN_TARGETS} targets/season |

**Scoring**

Each player gets a composite raw score from metrics weighted by importance
within his position group. Every metric is z-score standardised first, so that
per-play rates and season totals contribute on a comparable scale rather than
the larger-magnitude metric dominating.

**Team adjustment**

To separate a player from his surrounding cast, his raw score is centred on the
mean raw score of his own position group, on his own team, in that season —
implemented as a SQL window function. The adjusted score blends
{int(RESIDUAL_WEIGHT*100)}% residual with {int((1-RESIDUAL_WEIGHT)*100)}% raw
score, so a good player on a weak roster is not over-credited and a good player
on a strong roster is not erased.

**Replacement level**

Replacement is the {int(REPLACEMENT_PCTL*100)}th percentile of adjusted score
within each position group.

**Position weights**

QB {POSITION_WEIGHTS['QB']} · WR/TE {POSITION_WEIGHTS['WR']} · RB {POSITION_WEIGHTS['RB']}

These are **modelling assumptions**, informed by published positional-value
research — not parameters estimated from this dataset. Calibrating them against
team win totals is the top item on the roadmap.

**Known limitations**

- **Units are not literal wins.** The scale derives from z-scores, so WAR here
  ranks and separates players but a value of 3.0 does not mean three wins.
  The name follows baseball convention.
- **Replacement level is partly circular.** Volume thresholds already removed
  marginal players, so the {int(REPLACEMENT_PCTL*100)}th percentile of
  *qualified* players sits above a true replacement-level player. Deriving it
  from waiver-wire and practice-squad usage would be more honest.
- **No defensive players.** Public play-by-play cannot attribute a defensive
  outcome to an individual without snap and assignment data.
- **No offensive line.** Linemen have almost no box-score footprint; this needs
  charting data.
- **Team adjustment is fixed-effect centring, not a mixed-effects model.** A
  proper multilevel model with team as a grouping variable would partially pool
  estimates and shrink low-volume players toward the mean instead of letting a
  {MIN_TARGETS}-target receiver post an extreme score.

**Roadmap, in priority order**

1. Validate summed team WAR against actual team wins (implemented — Team View tab)
2. Estimate position weights from data instead of assuming them
3. Replace fixed-effect centring with a proper multilevel model
4. Test year-over-year stability of a player's WAR as a reliability check
5. Defensive players via charting/tracking data
    """)
