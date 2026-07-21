import ssl
from typing import cast
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn.datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

# Handle SSL certificate verification for dataset fetching
ssl._create_default_https_context = ssl._create_unverified_context

# Page Configuration
st.set_page_config(
    page_title="StatPulse | Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title & Intro
st.title("📊 StatPulse: Interactive Statistical & Predictive Dashboard")
st.markdown("""
*Welcome to the live demonstration dashboard!*  
Upload your own CSV or use the built-in sample dataset to explore real-time exploratory data analysis and predictive machine learning simulations.
""")

st.sidebar.header("📁 Data Input")

# 1. Data Selection / Upload
data_source = st.sidebar.radio("Choose Data Source:", ("Sample Dataset (housing)", "Upload CSV"))


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    sample_data = sklearn.datasets.fetch_california_housing(as_frame=True)
    sample_df = cast(pd.DataFrame, sample_data.frame)
    sample_df.columns = [col.replace(" ", "_") for col in sample_df.columns]
    return cast(pd.DataFrame, sample_df.sample(1000, random_state=42).reset_index(drop=True))


if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df: pd.DataFrame = cast(pd.DataFrame, pd.read_csv(uploaded_file))
    else:
        st.info("👆 Please upload a CSV file from the sidebar to proceed. Defaulting to sample data.")
        df = load_sample_data()
else:
    df = load_sample_data()

# 2. Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Data Preview & EDA", "📈 Interactive Visualizations", "🤖 Predictive Simulator"])

# --- TAB 1: EDA ---
with tab1:
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head(10), width="stretch")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T, width="stretch")

# --- TAB 2: VISUALIZATIONS ---
with tab2:
    st.subheader("Dynamic Feature Analysis")

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

        fig = px.scatter(
            plot_df,
            x=x_var,
            y=y_col_name,
            color=color_arg,
            trendline="ols" if color_var == "None" else None,
            title=f"{y_var} vs. {x_var}",
            labels={y_col_name: y_var},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for dynamic plotting.")

# --- TAB 3: PREDICTIVE SIMULATOR ---
with tab3:
    st.subheader("Real-Time Machine Learning Simulation")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        target_var = st.selectbox("Select Target Variable to Predict", numeric_cols, index=len(numeric_cols) - 1)
        feature_vars = st.multiselect("Select Feature Variables", numeric_cols,
                                      default=[c for c in numeric_cols if c != target_var])

        if feature_vars:
            st.sidebar.markdown("---")
            st.sidebar.header("⚙️ Model Parameters")
            test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20) / 100.0
            n_estimators = st.sidebar.slider("Number of Trees (Random Forest)", 10, 200, 50, step=10)

            X = df[feature_vars]
            y = df[target_var]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.markdown("### Model Metrics")
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
            m_col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

            # Actual vs Predicted Plot
            results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            fig_pred = px.scatter(
                results_df, x="Actual", y="Predicted",
                title="Actual vs Predicted Values",
                labels={"Actual": f"Actual {target_var}", "Predicted": f"Predicted {target_var}"},
                template="plotly_white"
            )

            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                               line=dict(color="Red", dash="dash"))

            st.plotly_chart(fig_pred, width="stretch")
        else:
            st.warning("Please select at least one feature variable to train the model.")