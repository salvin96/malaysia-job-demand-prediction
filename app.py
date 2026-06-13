import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

# ============================================================
# 1. APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Job Demand Prediction",
    page_icon="📊",
    layout="wide"
)

MODEL_DIR = "saved_ft_transformer_enhanced_model"


# ============================================================
# 2. FT-TRANSFORMER MODEL CLASS
# Must match training architecture exactly
# ============================================================

class FTTransformer(nn.Module):
    def __init__(
        self,
        categories,
        num_numeric,
        dim=64,
        depth=3,
        heads=4,
        dropout=0.15
    ):
        super().__init__()

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(category_size, dim)
            for category_size in categories
        ])

        self.num_weight = nn.Parameter(torch.randn(num_numeric, dim))
        self.num_bias = nn.Parameter(torch.randn(num_numeric, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x_cat, x_num):
        batch_size = x_cat.size(0)

        cat_tokens = []

        for i, emb in enumerate(self.cat_embeddings):
            cat_tokens.append(emb(x_cat[:, i]).unsqueeze(1))

        cat_tokens = torch.cat(cat_tokens, dim=1)

        num_tokens = (
            x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0)
            + self.num_bias.unsqueeze(0)
        )

        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)

        x = torch.cat([cls_tokens, cat_tokens, num_tokens], dim=1)

        x = self.transformer(x)

        cls_output = x[:, 0]

        output = self.regressor(cls_output)

        return output


# ============================================================
# 3. LOAD MODEL AND ARTIFACTS
# ============================================================

@st.cache_resource
def load_model_and_artifacts():
    state_encoder = joblib.load(os.path.join(MODEL_DIR, "state_encoder.pkl"))
    category_encoder = joblib.load(os.path.join(MODEL_DIR, "category_encoder.pkl"))
    num_scaler = joblib.load(os.path.join(MODEL_DIR, "num_scaler.pkl"))
    target_scaler_log = joblib.load(os.path.join(MODEL_DIR, "target_scaler_log.pkl"))

    cat_features = joblib.load(os.path.join(MODEL_DIR, "cat_features.pkl"))
    num_features = joblib.load(os.path.join(MODEL_DIR, "num_features.pkl"))

    categories = [
        len(state_encoder.classes_),
        len(category_encoder.classes_)
    ]

    model = FTTransformer(
        categories=categories,
        num_numeric=len(num_features),
        dim=64,
        depth=3,
        heads=4,
        dropout=0.15
    )

    model_path = os.path.join(
        MODEL_DIR,
        "best_ft_transformer_enhanced_model.pth"
    )

    if not os.path.exists(model_path):
        seed_model_path = os.path.join(
            MODEL_DIR,
            "seed_models",
            "ft_transformer_enhanced_seed_21.pth"
        )

        if os.path.exists(seed_model_path):
            model_path = seed_model_path
        else:
            st.error("Model file not found. Please check your saved model folder.")
            st.stop()

    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )

    model.eval()

    return {
        "model": model,
        "state_encoder": state_encoder,
        "category_encoder": category_encoder,
        "num_scaler": num_scaler,
        "target_scaler_log": target_scaler_log,
        "cat_features": cat_features,
        "num_features": num_features
    }


artifacts = load_model_and_artifacts()

@st.cache_data
def load_dashboard_data():
    df = pd.read_excel("jobstreet_cleaned_v2.xlsx")

    df = df[df["state"].notna()]
    df = df[df["state"].str.strip().str.lower() != "unknown"]

    if "month" not in df.columns:
        df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")
        df["month"] = df["year_month"].dt.month

    df = df[["state", "category", "year", "month"]].dropna()

    df["state"] = df["state"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    df_grouped = (
        df.groupby(["state", "category", "year", "month"])
        .size()
        .reset_index(name="job_count")
    )

    df_grouped["year_month"] = pd.to_datetime(
        df_grouped["year"].astype(str) + "-" + df_grouped["month"].astype(str) + "-01"
    )

    df_grouped["year_month_label"] = df_grouped["year_month"].dt.strftime("%b %Y")

    return df_grouped
    

# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================

def predict_job_demand(state, category, year, month):
    state_encoded = artifacts["state_encoder"].transform([state])[0]
    category_encoded = artifacts["category_encoder"].transform([category])[0]

    x_cat = np.array([[state_encoded, category_encoded]])

    x_num_raw = np.array([[year, month]])
    x_num_scaled = artifacts["num_scaler"].transform(x_num_raw)

    x_cat_tensor = torch.tensor(x_cat, dtype=torch.long)
    x_num_tensor = torch.tensor(x_num_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = artifacts["model"](x_cat_tensor, x_num_tensor).numpy()

    pred_log = artifacts["target_scaler_log"].inverse_transform(pred_scaled)

    pred_original = np.expm1(pred_log)
    pred_original = max(0, pred_original[0][0])

    return round(pred_original)


# ============================================================
# 5. UI
# ============================================================

st.title("Malaysia Job Demand Prediction System")

tab1, tab2 = st.tabs(["Prediction Model", "Dashboard"])


# ============================================================
# 6. PREDICTION TAB
# ============================================================

month_map = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

with tab1:
    st.header("Predict Job Demand")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox(
            "Select State",
            sorted(artifacts["state_encoder"].classes_)
        )

        year = st.selectbox(
            "Select Year",
            list(range(2023, 2031)),
            index=1
        )

    with col2:
        category = st.selectbox(
            "Select Job Category",
            sorted(artifacts["category_encoder"].classes_)
        )

        month_name = st.selectbox(
            "Select Month",
            list(month_map.keys())
        )

    predict_button = st.button("Predict Job Demand")

    if predict_button:
        month = month_map[month_name]

        prediction = predict_job_demand(
            state=state,
            category=category,
            year=year,
            month=month
        )

        if prediction < 20:
            st.error(f"Predicted Job Demand: {prediction}")
            st.write(
                f"Job availability for **{category}** in **{state}** "
                f"during **{month_name} {year}** is very low. "
                f"This category has limited job opportunities."
            )

        elif prediction <= 50:
            st.warning(f"Predicted Job Demand: {prediction}")
            st.write(
                f"Job demand for **{category}** in **{state}** "
                f"during **{month_name} {year}** is moderate. "
                f"There are some opportunities, but the demand is not very high."
            )

        elif prediction > 200:
            st.success(f"Predicted Job Demand: {prediction}")
            st.write(
                f"Excellent choice of job category. Demand for **{category}** "
                f"in **{state}** during **{month_name} {year}** is very strong."
            )

        else:
            st.success(f"Predicted Job Demand: {prediction}")
            st.write(
                f"Job opportunity is quite good for **{category}** "
                f"in **{state}** during **{month_name} {year}**."
            )


# ============================================================
# 7. DASHBOARD TAB
# ============================================================

with tab2:
    st.header("Job Demand Dashboard")

    dashboard_df = load_dashboard_data()

    st.subheader("Dashboard Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_states = st.multiselect(
            "Filter by State",
            sorted(dashboard_df["state"].unique()),
            default=sorted(dashboard_df["state"].unique())
        )

    with col2:
        selected_categories = st.multiselect(
            "Filter by Category",
            sorted(dashboard_df["category"].unique()),
            default=sorted(dashboard_df["category"].unique())
        )

    with col3:
        selected_years = st.multiselect(
            "Filter by Year",
            sorted(dashboard_df["year"].unique()),
            default=sorted(dashboard_df["year"].unique())
        )

    filtered_df = dashboard_df[
        (dashboard_df["state"].isin(selected_states)) &
        (dashboard_df["category"].isin(selected_categories)) &
        (dashboard_df["year"].isin(selected_years))
    ]

    st.divider()

    total_jobs = int(filtered_df["job_count"].sum())
    total_states = filtered_df["state"].nunique()
    total_categories = filtered_df["category"].nunique()

    m1, m2, m3 = st.columns(3)

    m1.metric("Total Job Demand", total_jobs)
    m2.metric("States Covered", total_states)
    m3.metric("Categories Covered", total_categories)

    st.divider()

    # Chart 1: Job Demand by State
    state_df = (
        filtered_df.groupby("state")["job_count"]
        .sum()
        .reset_index()
        .sort_values("job_count", ascending=False)
    )

    fig_state = px.bar(
        state_df,
        x="job_count",
        y="state",
        orientation="h",
        title="Job Demand by State",
        labels={"job_count": "Job Demand", "state": "State"}
    )

    st.plotly_chart(fig_state, use_container_width=True)

    # Chart 2: Job Demand by Category
    category_df = (
        filtered_df.groupby("category")["job_count"]
        .sum()
        .reset_index()
        .sort_values("job_count", ascending=False)
    )

    fig_category = px.bar(
        category_df,
        x="job_count",
        y="category",
        orientation="h",
        title="Job Demand by Category",
        labels={"job_count": "Job Demand", "category": "Category"}
    )

    st.plotly_chart(fig_category, use_container_width=True)

    # Chart 3: Year-Month Job Demand Trend
    trend_df = (
        filtered_df.groupby(["year_month", "year_month_label"])["job_count"]
        .sum()
        .reset_index()
        .sort_values("year_month")
    )

    fig_trend = px.line(
        trend_df,
        x="year_month_label",
        y="job_count",
        markers=True,
        title="Year-Month Job Demand Trend",
        labels={"year_month_label": "Year-Month", "job_count": "Job Demand"}
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # Chart 4: State vs Category Heatmap
    heatmap_df = (
        filtered_df.groupby(["state", "category"])["job_count"]
        .sum()
        .reset_index()
    )

    fig_heatmap = px.density_heatmap(
        heatmap_df,
        x="category",
        y="state",
        z="job_count",
        title="State vs Category Job Demand Heatmap",
        labels={
            "category": "Category",
            "state": "State",
            "job_count": "Job Demand"
        }
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)