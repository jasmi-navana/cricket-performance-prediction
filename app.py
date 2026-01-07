import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import numpy as np

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Cricket Player Performance Prediction",
    layout="wide"
)

# =================================================
# PDF-STYLE THEME
# =================================================
st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #eef3f9 0%, #ffffff 100%);
}
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #0b2c4a;
    margin-bottom: 6px;
}
.sub-title {
    text-align: center;
    color: #5c6f82;
    font-size: 15px;
    margin-bottom: 30px;
}
.card {
    background: #ffffff;
    padding: 24px 26px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(13,38,76,0.10);
    margin-bottom: 26px;
}

.metric-big {
    font-size: 44px;
    font-weight: 700;
    color: #1f77b4;
}
.conf {
    color: #2ca02c;
    font-weight: 600;
}
hr {
    margin-top: 25px;
    margin-bottom: 25px;
}
.section-gap {
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# =================================================
# LOAD MODELS & DATA
# =================================================
@st.cache_resource
def load_runs_model():
    return joblib.load("full_pipeline.joblib")

@st.cache_resource
def load_wickets_model():
    return joblib.load("wickets_pipeline.joblib")

@st.cache_data
def load_mapping():
    return pd.read_csv("datasets/mapping_data.csv")

@st.cache_data
def load_runs_dataset():
    return pd.read_csv("datasets/dataset.csv")

@st.cache_data
def load_wickets_dataset():
    return pd.read_csv("datasets/wicket_dataset.csv")
@st.cache_data
def load_player_roles():
    return pd.read_csv("datasets/player_role.csv")
@st.cache_data
def get_all_teams(mapping_df, wickets_df):
    bat_teams = mapping_df["team"].dropna().unique()
    bowl_teams = wickets_df["team"].dropna().unique()
    return sorted(set(bat_teams).union(set(bowl_teams)))



roles_df = load_player_roles()
player_role_dict = dict(zip(roles_df["player"], roles_df["role"]))

runs_model = load_runs_model()
wickets_model = load_wickets_model()

mapping_df = load_mapping()
dataset_df = load_runs_dataset()
wickets_df = load_wickets_dataset()
all_teams = get_all_teams(mapping_df, wickets_df)

preprocessor = runs_model.named_steps["preprocessor"]
xgb_runs = runs_model.named_steps["model"]
wkt_scaler = wickets_model.named_steps["scaler"]
xgb_wkts = wickets_model.named_steps["model"]




RUN_FEATURES = [
    "runs_last5",
    "sr_last5",
    "venue_avg_runs",
    "career_avg",
    "career_sr"
]

WICKET_FEATURES = [
    "wkts_last5",
    "eco_last5",
    "balls",
    "economy"
]

# =================================================
# HEADER
# =================================================
st.markdown("<div class='main-title'>CRICKET PLAYER PERFORMANCE PREDICTION</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>AI-based prediction of player performance in upcoming IPL matches</div>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# =================================================
# LAYOUT
# =================================================
left, center, right = st.columns([1.2, 1.8, 1.4])

# =================================================
# INPUT PANEL
# =================================================
players = sorted(mapping_df["batter"].unique())
venues = sorted(mapping_df["venue"].dropna().unique())
def normalize(values):
    values = np.array(values, dtype=float)
    return values / (values.max() + 1e-6)

# =================================================
# FEATURE ENGINEERING
# =================================================
def compute_run_features(player, venue):
    df = mapping_df[mapping_df["batter"] == player]
    df = df[df["balls"] > 0].sort_values("match_id")

    last5 = df.tail(5)

    return pd.DataFrame([{
        "runs_last5": last5["runs"].mean(),
        "sr_last5": last5["strike_rate"].mean(),
        "venue_avg_runs": (
            df[df["venue"] == venue]["runs"].mean()
            if venue in df["venue"].values else df["runs"].mean()
        ),
        "career_avg": df["runs"].mean(),
        "career_sr": df["strike_rate"].mean()
    }])

def compute_wicket_features(player):
    df = wickets_df[wickets_df["bowler"] == player]

    if df.empty:
        return None

    last5 = df.sort_values("match_id").tail(5)

    return pd.DataFrame([{
        "wkts_last5": last5["wkts_last5"].mean(),
        "eco_last5": last5["eco_last5"].mean(),
        "balls": last5["balls"].mean(),
        "economy": last5["economy"].mean()
    }])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔢 Input Parameters")

    player = st.selectbox("Select Player", players)

    # Detect role
    role = player_role_dict.get(player, "batsman")

    # Auto-detect player's own team
    if role == "batsman":
        player_team = (
            mapping_df[mapping_df["batter"] == player]["team"]
            .dropna()
            .unique()
        )
    elif role == "bowler":
        player_team = (
            wickets_df[wickets_df["bowler"] == player]["team"]
            .dropna()
            .unique()
        )
    else:
        bat_teams = mapping_df[mapping_df["batter"] == player]["team"].dropna().unique()
        bowl_teams = wickets_df[wickets_df["bowler"] == player]["team"].dropna().unique()
        player_team = list(set(bat_teams).union(set(bowl_teams)))

    player_team = player_team[0] if len(player_team) > 0 else "Unknown"

    # ---- OPPONENT TEAM INPUT ✅
    opponent_teams = [t for t in all_teams if t != player_team]

    opponent_team = st.selectbox(
        "Opponent Team",
        opponent_teams
    )

    venue = st.selectbox("Select Venue", venues)
    predict_btn = st.button("🔮 Predict Performance")

    st.markdown("</div>", unsafe_allow_html=True)


    # ---- Detect player role early (for all panels)
    role = player_role_dict.get(player, "batsman")

    # ---- Player Role Info
    if predict_btn:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧍 Player Role")

        role_label = {
            "batsman": "🏏 Specialist Batsman",
            "bowler": "🎯 Specialist Bowler",
            "allrounder": "🔄 All-Rounder"
        }

        st.markdown(f"**Role:** {role_label.get(role, role.title())}")
        st.markdown(f"**Team:** {player_team}")
        st.markdown("</div>", unsafe_allow_html=True)
    # ---- Match Context Card (SAFE VERSION)
    if predict_btn:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🏟️ Match Context")

        st.markdown(f"""
        **Player:** {player}  
        **Role:** {role.upper()}  
        **Team:** {player_team}  
        **Venue:** {venue}  
        **Match Type:** IPL League Match
        """)

        # Compute run features locally (only if relevant)
        if role in ["batsman", "allrounder"]:
            X_run_ctx = compute_run_features(player, venue)

            venue_hint = (
                "Batting-friendly venue"
                if X_run_ctx["venue_avg_runs"].values[0] >
                X_run_ctx["career_avg"].values[0]
                else "Balanced venue"
            )

            st.markdown(f"**Venue Insight:** {venue_hint}")

        st.markdown("</div>", unsafe_allow_html=True)


    # ---- Quick Player Snapshot (SAFE VERSION)
    if predict_btn:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Quick Snapshot")

        # Compute features locally (lightweight & safe)
        X_run_snap = compute_run_features(player, venue)
        X_wkt_snap = compute_wicket_features(player)

        if role in ["batsman", "allrounder"]:
            st.markdown(
                f"- **Avg Runs:** {round(X_run_snap['career_avg'].values[0], 1)}"
            )
            st.markdown(
                f"- **Strike Rate:** {round(X_run_snap['career_sr'].values[0], 1)}"
            )

        if role in ["bowler", "allrounder"] and X_wkt_snap is not None:
            st.markdown(
                f"- **Avg Economy:** {round(X_wkt_snap['economy'].values[0], 2)}"
            )
            st.markdown(
                f"- **Wickets (Last 5):** {round(X_wkt_snap['wkts_last5'].values[0], 1)}"
            )

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)


    # ---- Help / Guide Section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ℹ️ How to Read This Dashboard")

    st.markdown("""
    - **Predicted Runs / Wickets** are model outputs for the next match  
    - **Trends** show recent form (last 5 matches)  
    - **SHAP plots** explain which features influenced predictions  
    - **Radar chart** summarizes overall performance  

    Role-based filtering ensures only relevant analytics are shown.
    """)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)








# =================================================
# CENTER PANEL
# =================================================
with center:
    if predict_btn:
        # ---- Detect player role
        role = player_role_dict.get(player, "batsman")

        # ---- Runs prediction
        X_run = compute_run_features(player, venue)
        predicted_runs = runs_model.predict(X_run)[0]

        # ---- Wickets prediction
        X_wkt = compute_wicket_features(player)
        predicted_wkts = (
            wickets_model.predict(X_wkt)[0]
            if X_wkt is not None else 0
        )

        # ---- ROLE-BASED FILTERING
        if role == "batsman":
            predicted_wkts = 0

        elif role == "bowler":
            predicted_runs = 0

        # allrounder → keep both


        # ---- Prediction Card (RUNS + WICKETS)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        st.markdown(f"**Player Role:** `{role.upper()}`")


        with col1:
            st.markdown("### Predicted Runs")
            st.markdown(f"<div class='metric-big'>{int(predicted_runs)}</div>", unsafe_allow_html=True)
            st.markdown("<div class='conf'>High Confidence</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### Predicted Wickets")
            if predicted_wkts is not None:
                st.markdown(f"<div class='metric-big'>{predicted_wkts:.1f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='conf'>High Confidence</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='metric-big'>N/A</div>", unsafe_allow_html=True)
                st.markdown("<div style='color:#777'>No bowling data</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Player Form
        # ---- Player Form (ONLY for batsman / allrounder)
        if role in ["batsman", "allrounder"]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Player Form: Last 5 Batting Innings")

            form_df = (
                mapping_df[mapping_df["batter"] == player]
                .query("balls > 0")
                .sort_values("match_id")
                .tail(5)
                .reset_index(drop=True)
            )
            form_df["Inning"] = range(1, len(form_df) + 1)

            fig_form = px.line(
                form_df,
                x="Inning",
                y="runs",
                markers=True,
                labels={"runs": "Runs"}
            )
            st.plotly_chart(fig_form, use_container_width=True)
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

        # ---- Bowling Form: Last 5 Matches (ONLY for bowler / allrounder)
        if role in ["bowler", "allrounder"]:

            bowl_form_df = (
                wickets_df[wickets_df["bowler"] == player]
                .sort_values("match_id")
                .tail(5)
                .reset_index(drop=True)
            )

            if not bowl_form_df.empty:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Bowling Form: Last 5 Matches")

                bowl_form_df["Match"] = range(1, len(bowl_form_df) + 1)

                fig_bowl = px.line(
                    bowl_form_df,
                    x="Match",
                    y="wkts_last5",
                    markers=True,
                    labels={"wkts_last5": "Wickets"}
                )

                st.plotly_chart(fig_bowl, use_container_width=True)
                st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            else:
                st.info("No recent bowling data available.")
        # ---- Bowling Economy Trend: Last 5 Matches
        if role in ["bowler", "allrounder"]:

            eco_form_df = (
                wickets_df[wickets_df["bowler"] == player]
                .sort_values("match_id")
                .tail(5)
                .reset_index(drop=True)
            )

            if not eco_form_df.empty:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Bowling Economy Trend: Last 5 Matches")

                eco_form_df["Match"] = range(1, len(eco_form_df) + 1)

                fig_eco = px.line(
                    eco_form_df,
                    x="Match",
                    y="economy",
                    markers=True,
                    labels={"economy": "Economy Rate"}
                )

                st.plotly_chart(fig_eco, use_container_width=True)
                st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            else:
                st.info("No recent economy data available.")
            # ---- Combined Performance Radar Chart
    if predict_btn:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔵 Combined Performance Radar Chart")

        if role == "batsman":
            labels = [
                "Runs (Last 5)",
                "Strike Rate (Last 5)",
                "Venue Avg Runs",
                "Career Avg"
            ]

            values = [
                X_run["runs_last5"].values[0],
                X_run["sr_last5"].values[0],
                X_run["venue_avg_runs"].values[0],
                X_run["career_avg"].values[0]
            ]

        elif role == "bowler" and X_wkt is not None:
            labels = [
                "Wickets (Last 5)",
                "Economy (Inverse)",
                "Balls Bowled",
                "Career Economy (Inverse)"
            ]

            values = [
                X_wkt["wkts_last5"].values[0],
                10 - X_wkt["eco_last5"].values[0],   # inverse for radar
                X_wkt["balls"].values[0],
                10 - X_wkt["economy"].values[0]
            ]

        elif role == "allrounder" and X_wkt is not None:
            labels = [
                "Runs (Last 5)",
                "Strike Rate",
                "Wickets (Last 5)",
                "Economy (Inverse)",
                "Career Avg"
            ]

            values = [
                X_run["runs_last5"].values[0],
                X_run["sr_last5"].values[0],
                X_wkt["wkts_last5"].values[0],
                10 - X_wkt["economy"].values[0],
                X_run["career_avg"].values[0]
            ]
        else:
            st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
            st.stop()

        values_norm = normalize(values)
        values_norm = np.append(values_norm, values_norm[0])
        labels = labels + [labels[0]]

        fig = px.line_polar(
            r=values_norm,
            theta=labels,
            line_close=True
        )

        fig.update_traces(fill="toself")
        fig.update_layout(showlegend=False)

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)




# =================================================
# RIGHT PANEL
# =================================================
with right:
    if predict_btn:
        # ---- Input Feature Values (Runs)
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown("### Input Feature Values (Runs)")

        fig_feat = px.bar(
            x=RUN_FEATURES,
            y=X_run.iloc[0],
            labels={"x": "Feature", "y": "Value"}
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
                # ---- Prediction Summary Table
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Prediction Summary")

        # Safe display for wickets
        if predicted_wkts is not None:
            wkts_display = round(predicted_wkts, 1)
        else:
            wkts_display = "N/A"

        summary_df = pd.DataFrame([{
                "Player": player,
                "Player Team": player_team,
                "Opponent Team": opponent_team,
                "Venue": venue,
                "Predicted Runs": int(predicted_runs),
                "Predicted Wickets": wkts_display,
                "Confidence": "High"
        }])


        st.dataframe(summary_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


        # ---- Actual vs Predicted Runs
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Actual vs Predicted Runs")

        X_all = dataset_df[RUN_FEATURES]
        y_true = dataset_df["next_match_runs"]
        y_pred = runs_model.predict(X_all)

        fig_scatter = px.scatter(
            x=y_true,
            y=y_pred,
            labels={"x": "Actual Runs", "y": "Predicted Runs"}
        )

        fig_scatter.add_shape(
            type="line",
            x0=0, y0=0,
            x1=max(y_true), y1=max(y_true),
            line=dict(color="red", dash="dash")
        )

        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        
        # ---- SHAP Feature Importance (ROLE-BASED)
        if role in ["batsman", "allrounder"]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### SHAP Feature Importance (Runs)")
            try:
                # Background data from training (small sample)
                X_bg = dataset_df[RUN_FEATURES].sample(50, random_state=42)
                X_bg_proc = preprocessor.transform(X_bg)

                explainer = shap.Explainer(
                    xgb_runs.predict,
                    X_bg_proc
                )

                shap_vals = explainer(preprocessor.transform(X_run)).values

                fig, ax = plt.subplots(figsize=(6, 4))
                shap.summary_plot(
                    shap_vals,
                    X_bg_proc,
                    feature_names=RUN_FEATURES,
                    plot_type="bar",
                    show=False
                )

                st.pyplot(fig)

            except Exception:
                st.warning("SHAP explanation unavailable for this prediction.")


            st.markdown("</div>", unsafe_allow_html=True)
        # ---- SHAP Feature Importance (WICKETS)
        if role in ["bowler", "allrounder"] and X_wkt is not None:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### SHAP Feature Importance (Wickets)")

            # Apply the same scaling used during training
            X_wkt_scaled = wkt_scaler.transform(X_wkt)

            # SHAP explainer for bowling model
            try:
                # Background data (multiple rows)
                X_bg = wickets_df[WICKET_FEATURES].sample(100, random_state=42)
                X_bg_scaled = wkt_scaler.transform(X_bg)

                explainer_wkt = shap.Explainer(
                    xgb_wkts.predict,
                    X_bg_scaled
                )

                shap_vals_wkt = explainer_wkt(X_bg_scaled).values

                fig, ax = plt.subplots(figsize=(6, 4))
                shap.summary_plot(
                    shap_vals_wkt,
                    X_bg_scaled,
                    feature_names=WICKET_FEATURES,
                    plot_type="bar",
                    show=False
                )

                st.pyplot(fig)

            except Exception:
                st.warning("SHAP explanation unavailable for bowling model.")


            st.markdown("</div>", unsafe_allow_html=True)
        # ---- Key Insights Summary (Bottom Right)
        if predict_btn:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🔎 Key Insights")

            if role == "batsman":
                st.markdown(f"""
                • Player is a **specialist batsman**  
                • Recent batting form is **{'strong' if X_run['runs_last5'].values[0] > 30 else 'moderate'}**  
                • Venue conditions are **{'favorable' if X_run['venue_avg_runs'].values[0] > X_run['career_avg'].values[0] else 'neutral'}**
                """)

            elif role == "bowler" and X_wkt is not None:
                st.markdown(f"""
                • Player is a **specialist bowler**  
                • Recent wicket-taking form is **{'good' if X_wkt['wkts_last5'].values[0] > 1.5 else 'average'}**  
                • Economy trend indicates **{'good control' if X_wkt['economy'].values[0] < 8 else 'high run rate'}**
                """)

            else:  # allrounder
                st.markdown("""
                • Player contributes in **both batting and bowling**  
                • Overall performance profile is **balanced**  
                • Suitable for multi-dimensional impact in match
                """)

            st.markdown("</div>", unsafe_allow_html=True)

# =================================================
# FOOTER
# =================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
**Selected Player:** {player}  
**Venue:** {venue}
""")
