import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import fetch_california_housing

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StatPulse Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- TITLE & DESCRIPTION ---
st.title("📊 StatPulse: Interactive Data Analytics & ML Dashboard")
st.markdown(
    "A sleek, interactive tool for exploratory data analysis, dynamic visualization, and machine learning simulation.")

# --- SIDEBAR: DATA INPUT ---
st.sidebar.header("📁 Data Input")
data_source = st.sidebar.radio("Choose Data Source:", ("Sample Dataset (Housing)", "Upload CSV"))


@st.cache_data
def load_sample_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    # Take a manageable sample for responsiveness
    return df.sample(n=500, random_state=42).reset_index(drop=True)


if data_source == "Sample Dataset (Housing)":
    df = load_sample_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file or switch to the sample dataset.")
        st.stop()

# Ensure unique column names upfront to prevent downstream plotting issues
df = df.loc[:, ~df.columns.duplicated()]

# --- TAB NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "📈 Bivariate Analysis", "🤖 Predictive Simulator"])

# ==========================================
# TAB 1: DATA OVERVIEW
# ==========================================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().T, use_container_width=True)
    with col2:
        st.subheader("Missing Values")
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ["Column", "Missing Count"]
        st.dataframe(missing_data, use_container_width=True)

# ==========================================
# TAB 2: BIVARIATE ANALYSIS & VISUALIZATION
# ==========================================
with tab2:
    st.subheader("Interactive Visualizations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        col_x, col_y, col_color = st.columns(3)
        x_var = col_x.selectbox("X-Axis Variable", numeric_cols, index=0)
        y_var = col_y.selectbox("Y-Axis Variable", numeric_cols, index=min(1, len(numeric_cols) - 1))

        categorical_cols = ["None"] + df.select_dtypes(include=["object", "category"]).columns.tolist()
        color_var = col_color.selectbox("Color By (Optional)", categorical_cols)

        color_arg = None if color_var == "None" else color_var

        # Prepare a clean plot DataFrame to prevent duplicate key errors when x_var == y_var
        plot_df = pd.DataFrame({
            x_var: df[x_var],
            f"{y_var}_y" if x_var == y_var else y_var: df[y_var]
        })

        y_col_name = f"{y_var}_y" if x_var == y_var else y_var

        if color_arg:
            plot_df[color_arg] = df[color_arg]

        # Stylish Scatter Plot with Soft Muted Colors
        fig = px.scatter(
            plot_df,
            x=x_var,
            y=y_col_name,
            color=color_arg,
            trendline="ols" if color_var == "None" else None,
            trendline_color_override="#C46857",  # Soft Terracotta trendline
            title=f"<b>{y_var}</b> vs. <b>{x_var}</b>",
            labels={y_col_name: y_var},
            template="plotly_dark"
        )

        # Style points with faded slate gray if no category selected
        if color_var == "None":
            fig.update_traces(
                marker=dict(size=7, color="#7B8B9A", opacity=0.6, line=dict(width=0.5, color="#D0D7DE"))
            )

        st.plotly_chart(fig, use_container_width=True)

        # Elegant Neutral Correlation Heatmap
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()

        # Faded Neutral Palette (Muted Charcoal -> Soft Cream -> Faded Taupe)
        neutral_palette = [
            [0.0, "#334155"],  # Muted Dark Slate (-1)
            [0.5, "#F8FAFC"],  # Soft Cream / Neutral White (0)
            [1.0, "#C5A089"]  # Faded Earth Taupe (+1)
        ]

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=neutral_palette,
            zmin=-1, zmax=1,
            aspect="auto",
            template="plotly_dark"
        )
        fig_corr.update_layout(
            margin=dict(l=40, r=40, t=30, b=30),
            coloraxis_showscale=True
        )

        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Not enough numerical columns available to display bivariate visualizations.")

# ==========================================
# TAB 3: PREDICTIVE SIMULATOR
# ==========================================
with tab3:
    st.subheader("Real-Time Machine Learning Simulation")
    st.info("Dynamic model training and prediction module configured for real-time exploratory analysis.")