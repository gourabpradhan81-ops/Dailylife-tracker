"""
Daybook — Daily Routine Tracker
--------------------------------
A Streamlit habit tracker for a rolling 30-day cycle.

Run with:  streamlit run app.py
Requires : streamlit, pandas, numpy, matplotlib, seaborn, plotly

Persistence note: browser localStorage isn't available to a Python/Streamlit
process, so this app persists state to a local JSON file (routine_data.json,
created next to app.py) instead — it survives restarts the same way.
"""

import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Daybook — Routine Tracker",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CYCLE_LEN = 30
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "routine_data.json")

HABITS = [
    ("💧", "Drink 2L Water"),
    ("🏃", "Exercise 30 min"),
    ("📖", "Read 20 min"),
    ("🌙", "Sleep 8 Hours"),
    ("🧘", "Meditate"),
    ("🥗", "Healthy Eating"),
    ("🗒️", "Plan the Day"),
    ("📵", "Screen-free Hour"),
    ("🎯", "Learn a Skill"),
    ("✍️", "Journaling"),
]
HABIT_LABELS = [h[1] for h in HABITS]

MOSS = "#6D28D9"   # primary violet — "Productive" / checked states
CLAY = "#BE185D"   # magenta — "Incomplete" states
GOLD = "#C026D3"   # fuchsia accent — target line / highlights
SAGE = "#C4B5FD"   # light violet — mid-range shade
AMBER = "#F3D9FA"  # soft pink-lavender — partial-completion shade
PAPER = "#F6F1FB"  # page background — soft lavender
CARD = "#FFFFFF"   # card background
INK = "#2E1A47"    # primary text — deep purple-black
INK_SOFT = "#6B5B8A"  # secondary text — muted violet-grey
RULE = "#DCC9F0"   # borders / dividers

# ----------------------------------------------------------------------
# CUSTOM CSS — dynamic themed UI
# ----------------------------------------------------------------------
st.markdown(f"""
<style>
    /* Base app background + default text color (fixes white-on-light invisible text) */
    .stApp {{ background-color: {PAPER}; color: {INK}; }}

    /* Force readable text color across every Streamlit text element, regardless of
       light/dark client theme — this is what was rendering white-on-light before. */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp li, .stApp td, .stApp th, .stApp a,
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stCaption, small,
    .stCheckbox label p, .stCheckbox label span,
    div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"],
    div[data-testid="stMetricDelta"],
    section[data-testid="stSidebar"] * ,
    .stExpander, .stExpander p, .stExpander span,
    div[data-testid="stDataFrame"] * {{
        color: {INK} !important;
    }}

    section[data-testid="stSidebar"] {{ background-color: {CARD}; border-right: 1px solid {RULE}; }}
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] {{ color: {INK_SOFT} !important; }}

    div[data-testid="stMetric"] {{
        background: {CARD}; border: 1px solid {RULE}; border-radius: 6px;
        padding: 14px 16px; box-shadow: 0 1px 2px rgba(46,26,71,0.06);
    }}
    div[data-testid="stMetricLabel"] {{ color: {INK_SOFT} !important; font-weight: 500; }}
    div[data-testid="stMetricValue"] {{ color: {MOSS} !important; font-weight: 700; }}

    .badge {{
        display:inline-block; padding:6px 14px; border-radius:100px;
        font-weight:600; font-size:13px; letter-spacing:0.03em;
    }}
    .badge-productive {{ background:#EDE4FC; color:{MOSS} !important; border:1px solid {MOSS}; }}
    .badge-incomplete {{ background:#FBE1EC; color:{CLAY} !important; border:1px solid {CLAY}; }}

    .habit-row {{ padding:6px 10px; border-radius:5px; margin-bottom:2px; }}
    .habit-row:hover {{ background: rgba(109,40,217,0.06); }}

    /* Buttons */
    .stButton button {{
        border: 1px solid {MOSS} !important; color: {MOSS} !important; background: {CARD} !important;
    }}
    .stButton button:hover {{ background: {MOSS} !important; color: #FFFFFF !important; }}
    .stButton button[kind="primary"] {{ background: {MOSS} !important; color: #FFFFFF !important; }}
    .stButton button[kind="primary"]:hover {{ background: #5B21B6 !important; }}

    /* Sidebar caption / helper text */
    .stCaption p, div[data-testid="stCaptionContainer"] {{ color: {INK_SOFT} !important; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# DATA LAYER (JSON file = local equivalent of localStorage)
# ----------------------------------------------------------------------
def default_state():
    return {"cycle_start": date.today().isoformat(), "days": {}}


def load_state() -> dict:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
            if "cycle_start" in data and "days" in data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return default_state()


def save_state(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


if "data" not in st.session_state:
    st.session_state.data = load_state()

data = st.session_state.data


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def today_str() -> str:
    return date.today().isoformat()


def cycle_dates(cycle_start: str) -> list:
    start = date.fromisoformat(cycle_start)
    return [(start + timedelta(days=i)).isoformat() for i in range(CYCLE_LEN)]


def score_of(habit_arr) -> float:
    """(completed / 10) * 100 using numpy."""
    if habit_arr is None:
        return np.nan
    arr = np.array(habit_arr, dtype=bool)
    return float(np.round((arr.sum() / len(HABIT_LABELS)) * 100, 1))


def build_cycle_df(data: dict) -> pd.DataFrame:
    dates = cycle_dates(data["cycle_start"])
    rows = []
    for i, d in enumerate(dates):
        habits = data["days"].get(d)
        rows.append({
            "day_num": i + 1,
            "date": d,
            "score": score_of(habits) if habits is not None else np.nan,
            "completed": int(np.sum(habits)) if habits is not None else 0,
            "logged": habits is not None,
        })
    return pd.DataFrame(rows)


def cycle_index_of_today(cycle_start: str) -> int:
    return (date.today() - date.fromisoformat(cycle_start)).days


def toggle_habit(idx: int):
    t = today_str()
    arr = data["days"].get(t, [False] * 10).copy()
    arr[idx] = not arr[idx]
    data["days"][t] = arr
    save_state(data)


def reset_today():
    data["days"][today_str()] = [False] * 10
    save_state(data)


def start_new_month():
    st.session_state.data = default_state()
    save_state(st.session_state.data)


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
st.sidebar.title("🗓️ Daybook")
st.sidebar.caption("Daily Routine Ledger")
st.sidebar.markdown("---")

cyc_idx = cycle_index_of_today(data["cycle_start"])
cycle_complete = cyc_idx >= CYCLE_LEN
day_label = f"Day {min(cyc_idx, CYCLE_LEN-1) + 1} of {CYCLE_LEN}" if not cycle_complete else "Cycle complete"

st.sidebar.metric("Cycle Progress", day_label)
st.sidebar.write(f"**Cycle start:** {data['cycle_start']}")
st.sidebar.write(f"**Today:** {today_str()}")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Manage")

if st.sidebar.button("↺ Reset Today's Habits", use_container_width=True, disabled=cycle_complete):
    reset_today()
    st.rerun()

confirm_reset = st.sidebar.checkbox("Confirm: I want to start a new 30-day cycle (this clears all data)")
if st.sidebar.button("🗑️ Start New Month", use_container_width=True, type="primary", disabled=not confirm_reset):
    start_new_month()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Data stored locally at `routine_data.json`")

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("🗓️ Daybook — Daily Routine Tracker")
st.caption(f"{date.today().strftime('%A, %B %d, %Y')} · Tracking 10 daily habits across a 30-day cycle")

df_cycle = build_cycle_df(data)
today_habits = data["days"].get(today_str())
today_score = score_of(today_habits) if today_habits is not None else 0.0
today_completed = int(np.sum(today_habits)) if today_habits is not None else 0
is_productive_day = today_score == 100.0

# Monthly average — unlogged days count as 0, using numpy
scores_for_avg = df_cycle["score"].fillna(0).to_numpy()
month_avg = float(np.round(scores_for_avg.mean(), 1))
is_productive_month = month_avg >= 80.0
logged_days = int(df_cycle["logged"].sum())

st.markdown("---")

# ----------------------------------------------------------------------
# TOP METRICS
# ----------------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Today's Score", f"{today_score:.0f}%", f"{today_completed}/10 habits")
m2.metric("Today's Status", "Productive" if is_productive_day else "Incomplete")
m3.metric("Days Logged", f"{logged_days}/{CYCLE_LEN}")
m4.metric("Running Monthly Avg", f"{month_avg:.1f}%")
m5.metric("Month Grade", "Productive" if is_productive_month else "Needs Improvement")

st.markdown("---")

# ----------------------------------------------------------------------
# MAIN LAYOUT
# ----------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.3])

# ---------------- LEFT: Today's checklist + gauge ----------------
with col_left:
    st.subheader("✅ Today's Habits")

    if cycle_complete:
        st.info("This 30-day cycle has ended. Start a new month from the sidebar to keep logging.")
    else:
        arr = today_habits if today_habits is not None else [False] * 10
        for idx, (icon, label) in enumerate(HABITS):
            checked = st.checkbox(
                f"{icon}  {label}",
                value=bool(arr[idx]),
                key=f"habit_{idx}",
                on_change=toggle_habit,
                args=(idx,),
            )

    st.markdown(
        f'<span class="badge {"badge-productive" if is_productive_day else "badge-incomplete"}">'
        f'{"● Productive Day" if is_productive_day else "● Incomplete"}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("###")
    # Plotly gauge for today's score
    gauge_color = MOSS if is_productive_day else CLAY
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=today_score,
        number={"suffix": "%", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": gauge_color, "thickness": 0.32},
            "bgcolor": CARD,
            "borderwidth": 1,
            "bordercolor": RULE,
            "steps": [
                {"range": [0, 50], "color": "#FBE1EC"},
                {"range": [50, 99], "color": "#E9E0FB"},
                {"range": [99, 100], "color": "#D5C4F7"},
            ],
            "threshold": {"line": {"color": INK, "width": 3}, "thickness": 0.85, "value": 100},
        },
        title={"text": "Today's Productivity Score", "font": {"size": 14}},
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10),
                             paper_bgcolor="rgba(0,0,0,0)", font_color=INK)
    st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------- RIGHT: Cycle trend ----------------
with col_right:
    st.subheader("📈 30-Day Cycle Trend")

    plot_df = df_cycle.copy()
    plot_df["status"] = np.select(
        [plot_df["score"] == 100, plot_df["score"].notna() & (plot_df["score"] < 100), plot_df["score"].isna()],
        ["Productive", "Incomplete", "Not logged"],
        default="Not logged",
    )
    fig_trend = px.bar(
        plot_df, x="day_num", y=plot_df["score"].fillna(0), color="status",
        color_discrete_map={"Productive": MOSS, "Incomplete": CLAY, "Not logged": "#E4DAF5"},
        labels={"day_num": "Day of Cycle", "y": "Score (%)"},
        template="plotly_white",
    )
    fig_trend.add_hline(y=80, line_dash="dash", line_color=GOLD,
                         annotation_text="80% monthly target", annotation_position="top left")
    fig_trend.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             legend_title_text="", font_color=INK)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown(
        f'<span class="badge {"badge-productive" if is_productive_month else "badge-incomplete"}">'
        f'{"● Productive Month" if is_productive_month else "● Needs Improvement"} '
        f'— {month_avg:.1f}% average</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ----------------------------------------------------------------------
# CALENDAR HEATMAP (matplotlib + seaborn)
# ----------------------------------------------------------------------
st.subheader("🔥 Monthly Calendar Heatmap")

heat_values = df_cycle["score"].to_numpy(dtype=float)
heat_grid = np.full(30, np.nan)
heat_grid[:len(heat_values)] = heat_values
heat_grid = heat_grid.reshape(6, 5)  # 6 rows x 5 cols = 30 days

day_labels_grid = np.arange(1, 31).reshape(6, 5)
annot = np.array([
    [f"D{day_labels_grid[r,c]}\n{'' if np.isnan(heat_grid[r,c]) else str(int(heat_grid[r,c]))+'%'}"
     for c in range(5)] for r in range(6)
])

from matplotlib.colors import LinearSegmentedColormap
purple_cmap = LinearSegmentedColormap.from_list(
    "daybook_purple", ["#BE185D", "#F3D9FA", "#6D28D9"]  # low -> mid -> high score
)

fig_heat, ax = plt.subplots(figsize=(11, 5.2))
fig_heat.patch.set_facecolor(PAPER)
sns.heatmap(
    heat_grid, annot=False, cmap=purple_cmap, vmin=0, vmax=100,
    linewidths=2, linecolor=PAPER, cbar_kws={"label": "Score (%)"},
    mask=np.isnan(heat_grid), ax=ax, square=True,
)

# Manually place annotations so text color adapts to cell brightness
# (both ends of the purple scale are dark, the middle is light).
for r in range(6):
    for c in range(5):
        val = heat_grid[r, c]
        if np.isnan(val):
            continue
        txt_color = INK if 30 <= val <= 70 else "#FFFFFF"
        ax.text(c + 0.5, r + 0.5, annot[r, c], ha="center", va="center",
                 fontsize=9, color=txt_color, linespacing=1.6)

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_color(INK)
cbar.ax.tick_params(colors=INK)
ax.set_facecolor("#EDE4FB")
ax.set_title("Score by Day — Current 30-Day Cycle", fontsize=13, pad=12, color=INK)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel("")
ax.set_ylabel("")
st.pyplot(fig_heat)

st.markdown("---")

# ----------------------------------------------------------------------
# HABIT-LEVEL BREAKDOWN (which habits get skipped most)
# ----------------------------------------------------------------------
st.subheader("📊 Habit Completion Breakdown (this cycle)")

habit_matrix = []
for d in cycle_dates(data["cycle_start"]):
    habits = data["days"].get(d)
    habit_matrix.append(habits if habits is not None else [np.nan] * 10)

habit_df = pd.DataFrame(habit_matrix, columns=HABIT_LABELS)
completion_rate = habit_df.mean(skipna=True).mul(100).round(1).sort_values(ascending=True)

c1, c2 = st.columns([1.3, 1])
with c1:
    fig_bar = px.bar(
        x=completion_rate.values, y=completion_rate.index, orientation="h",
        labels={"x": "Completion Rate (%)", "y": ""},
        color=completion_rate.values, color_continuous_scale=["#BE185D", "#C4B5FD", "#6D28D9"],
        template="plotly_white",
    )
    fig_bar.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10),
                           coloraxis_showscale=False, font_color=INK,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.markdown("**Weakest habits this cycle**")
    weakest = completion_rate.head(3)
    for label, rate in weakest.items():
        st.write(f"• {label} — {rate:.0f}% completion")
    st.markdown("**Strongest habits this cycle**")
    strongest = completion_rate.tail(3).sort_values(ascending=False)
    for label, rate in strongest.items():
        st.write(f"• {label} — {rate:.0f}% completion")

st.markdown("---")

# ----------------------------------------------------------------------
# RAW DATA / EXPORT
# ----------------------------------------------------------------------
with st.expander("📋 View & export raw cycle data"):
    display_df = df_cycle.copy()
    display_df["status"] = np.where(
        display_df["score"] == 100, "Productive",
        np.where(display_df["logged"], "Incomplete", "Not logged")
    )
    st.dataframe(display_df[["day_num", "date", "completed", "score", "status"]],
                 use_container_width=True, hide_index=True)
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Cycle Data as CSV", data=csv_bytes,
                        file_name="routine_cycle_data.csv", mime="text/csv")

st.caption("Score = (completed habits ÷ 10) × 100 · 100% marks a Productive Day · "
           "≥80% cycle average marks a Productive Month")
