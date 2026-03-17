import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")

st.markdown(
    "Predict whether a transaction is fraudulent and explore fraud insights."
)

# -------------------------------------------------
# LOAD MODEL FILES
# -------------------------------------------------

model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_list = pickle.load(open("feature_list.pkl", "rb"))

# -------------------------------------------------
# LOAD DATASET FOR INSIGHTS
# -------------------------------------------------

dftrain = None

try:
    dftrain = pd.read_csv("fraudTrain.csv")

    if "trans_date_trans_time" in dftrain.columns:
        dftrain["trans_date_trans_time"] = pd.to_datetime(
            dftrain["trans_date_trans_time"]
        )
        dftrain["hour"] = dftrain["trans_date_trans_time"].dt.hour

except:
    dftrain = None

# -------------------------------------------------
# EXTRACT OPTIONS FROM MODEL FEATURES
# -------------------------------------------------

available_states = sorted(
    [col.replace("state_", "") for col in feature_list if col.startswith("state_")]
)

available_categories = sorted(
    [col.replace("category_", "") for col in feature_list if col.startswith("category_")]
)

# -------------------------------------------------
# SIDEBAR INPUT
# -------------------------------------------------

st.sidebar.header("Transaction Input")

amt = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)

hour = st.sidebar.slider("Transaction Hour", 0, 23, 12)

day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

month = st.sidebar.slider("Month", 1, 12, 6)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

state = st.sidebar.selectbox("State", available_states)

category = st.sidebar.selectbox("Transaction Category", available_categories)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.sidebar.button("Predict Fraud"):

    input_data = pd.DataFrame(columns=feature_list)
    input_data.loc[0] = 0

    for col in input_data.columns:

        if col == "amt":
            input_data.at[0, col] = amt

        elif col == "hour":
            input_data.at[0, col] = hour

        elif col == "day_of_week":
            input_data.at[0, col] = day_of_week

        elif col == "month":
            input_data.at[0, col] = month

    gender_col = f"gender_{gender}"
    if gender_col in input_data.columns:
        input_data.at[0, gender_col] = 1

    state_col = f"state_{state}"
    if state_col in input_data.columns:
        input_data.at[0, state_col] = 1

    category_col = f"category_{category}"
    if category_col in input_data.columns:
        input_data.at[0, category_col] = 1

    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]

    st.header("Prediction Result")

    st.write(f"Fraud Probability: **{prob:.2%}**")

    if prob < 0.01:
        st.success("🟢 Low Risk Transaction")

    elif prob < 0.05:
        st.warning("🟡 Medium Risk Transaction")

    else:
        st.error("🔴 High Risk Transaction")

    # -------------------------------------------------
    # FRAUD RISK GAUGE
    # -------------------------------------------------

    st.subheader("Fraud Risk Meter")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Fraud Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 10], "color": "lightgreen"},
                    {"range": [10, 30], "color": "yellow"},
                    {"range": [30, 100], "color": "salmon"},
                ],
            },
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------

    st.subheader("Top 10 Features Influencing Fraud")

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame(
        {"Feature": feature_list, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    st.bar_chart(feature_importance_df.head(10).set_index("Feature"))

# -------------------------------------------------
# FRAUD INSIGHTS DASHBOARD
# -------------------------------------------------

if dftrain is not None:

    st.header("Fraud Insights Dashboard")

    col1, col2 = st.columns(2)

    # -------------------------------------------------
    # AMOUNT DISTRIBUTION
    # -------------------------------------------------

    with col1:

        st.subheader("Transaction Amount Distribution")

        fig, ax = plt.subplots()

        ax.hist(dftrain["amt"], bins=50)

        ax.set_xlabel("Amount")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    # -------------------------------------------------
    # FRAUD BY HOUR
    # -------------------------------------------------

    with col2:

        if "hour" in dftrain.columns:

            st.subheader("Fraud Rate by Hour")

            fraud_hour = dftrain.groupby("hour")["is_fraud"].mean()

            fig, ax = plt.subplots()

            ax.plot(fraud_hour.index, fraud_hour.values)

            ax.set_xlabel("Hour")
            ax.set_ylabel("Fraud Rate")

            st.pyplot(fig)

    # -------------------------------------------------
    # FRAUD BY CATEGORY
    # -------------------------------------------------

    if "category" in dftrain.columns:

        st.subheader("Fraud Rate by Category")

        category_fraud = (
            dftrain.groupby("category")["is_fraud"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(category_fraud)

else:

    st.warning(
        "fraudTrain.csv was not found in the project folder. Insight graphs cannot be displayed."
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")
st.markdown("Machine Learning Model for Credit Card Fraud Detection")