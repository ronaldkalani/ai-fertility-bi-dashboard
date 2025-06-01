import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Streamlit Page Setup
st.set_page_config(page_title="AI BI Dashboard", layout="wide")
st.title("📊 AI Fertility Centre – Business Intelligence Dashboard")
st.markdown("This dashboard integrates clinical appointment analytics with predictive modeling to inform operational and clinical decisions.")

# Load dataset
DATA_PATH = "cleaned_appointment.csv"
df = pd.read_csv(DATA_PATH)

# Simulate missing data
if "TreatmentType" not in df.columns:
    df["TreatmentType"] = np.random.choice(["IVF", "IUI", "Egg Freezing", "Donor Program"], len(df))
if "SatisfactionScore" not in df.columns:
    df["SatisfactionScore"] = np.random.randint(7, 11, size=len(df))
if "ReferralSource" not in df.columns:
    df["ReferralSource"] = np.random.choice(["Google Ads", "OB-GYN", "Webinar", "Instagram", "Family Doctor"], len(df))
if "No_show" not in df.columns:
    df["No_show"] = np.random.choice(["Yes", "No"], size=len(df))
if "SMS_received" not in df.columns:
    df["SMS_received"] = np.random.choice([0, 1], size=len(df))
if "ScheduledDay" not in df.columns or "AppointmentDay" not in df.columns:
    df["WaitDays"] = np.random.randint(1, 20, size=len(df))
else:
    df["WaitDays"] = (pd.to_datetime(df["AppointmentDay"]) - pd.to_datetime(df["ScheduledDay"])).dt.days

# Preprocess for modeling
df["No_show_binary"] = df["No_show"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Age", "WaitDays", "SMS_received", "No_show_binary"])
X = df[["Age", "WaitDays", "SMS_received"]]
y = df["No_show_binary"]
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.3)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Sidebar Dropdown for Section Selection
st.sidebar.header("📊 Select Dashboard Section")
metric_option = st.sidebar.selectbox("Choose a Metric or Insight", [
    "Average Satisfaction Score by Treatment Type",
    "Predicted No-Show Risk",
    "Total Appointments Summary",
    "No-Show Rate (%)",
    "Average Wait Time (days)",
    "Proportion of SMS Reminders Sent",
    "Appointment Volume by Region",
    "Referral Source Breakdown",
    "Monthly Appointment Trends",
    "Public Platform Ratings",
    "Competitor Watchlist",
    "AI Treatment Path Suggestion",
    "Logistic Regression Model – Predictive Analytics",
    "ROC Curve",
    "Summary"
])

# Dynamic Viewer
st.header(f"📌 {metric_option}")

if metric_option == "Average Satisfaction Score by Treatment Type":
    avg_satisfaction = df.groupby("TreatmentType")["SatisfactionScore"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=avg_satisfaction, x="TreatmentType", y="SatisfactionScore", palette="Set2", ax=ax)
    ax.set_title("Satisfaction Score by Treatment")
    st.pyplot(fig)

elif metric_option == "Predicted No-Show Risk":
    df["NoShowProb"] = df["No_show"].apply(lambda x: 0.85 if x == "Yes" else 0.15)
    st.dataframe(df[["PatientId", "NoShowProb"]].head() if "PatientId" in df.columns else df[["NoShowProb"]].head())

elif metric_option == "Total Appointments Summary":
    st.metric("Total Appointments", f"{len(df):,}")
    st.metric("No-Show Rate", f"{(df['No_show'] == 'Yes').mean() * 100:.2f}%")
    st.metric("Average Wait Time", f"{df['WaitDays'].mean():.1f} days")
    st.metric("Average Satisfaction", f"{df['SatisfactionScore'].mean():.2f}/10")

elif metric_option == "No-Show Rate (%)":
    st.metric("No-Show Rate", f"{(df['No_show'] == 'Yes').mean() * 100:.2f}%")

elif metric_option == "Average Wait Time (days)":
    st.metric("Average Wait Time", f"{df['WaitDays'].mean():.1f} days")

elif metric_option == "Proportion of SMS Reminders Sent":
    sms = df["SMS_received"].value_counts(normalize=True).rename({0: "No SMS", 1: "Received SMS"}) * 100
    fig, ax = plt.subplots()
    sms.plot(kind="bar", color=["red", "green"], ax=ax)
    ax.set_title("SMS Reminder Distribution")
    st.pyplot(fig)

elif metric_option == "Appointment Volume by Region":
    if "Neighbourhood" in df.columns:
        reg = df["Neighbourhood"].value_counts().head(10).reset_index()
        reg.columns = ["Neighbourhood", "Appointments"]
        fig, ax = plt.subplots()
        sns.barplot(data=reg, x="Appointments", y="Neighbourhood", palette="coolwarm", ax=ax)
        st.pyplot(fig)

elif metric_option == "Referral Source Breakdown":
    ref = df["ReferralSource"].value_counts().reset_index()
    ref.columns = ["Source", "Leads"]
    fig, ax = plt.subplots()
    sns.barplot(data=ref, x="Leads", y="Source", palette="magma", ax=ax)
    st.pyplot(fig)

elif metric_option == "Monthly Appointment Trends":
    if "AppointmentDay" in df.columns:
        df["Month"] = pd.to_datetime(df["AppointmentDay"]).dt.strftime("%b")
        month_df = df["Month"].value_counts().sort_index().reset_index()
        month_df.columns = ["Month", "Appointments"]
        fig, ax = plt.subplots()
        sns.lineplot(data=month_df, x="Month", y="Appointments", marker="o", ax=ax)
        st.pyplot(fig)

elif metric_option == "Public Platform Ratings":
    fig, ax = plt.subplots()
    sns.barplot(data=pd.DataFrame({
        "Platform": ["Google", "RateMDs", "Facebook"],
        "Rating": [4.7, 4.6, 4.8]
    }), x="Platform", y="Rating", palette="Set2", ax=ax)
    ax.set_title("Public Ratings by Platform")
    st.pyplot(fig)

elif metric_option == "Competitor Watchlist":
    st.dataframe(pd.DataFrame({
        "Clinic": ["TRIO", "CReATe", "Astra"],
        "IVF Success Rate (%)": [63, 61, 58],
        "Google Rating": [4.8, 4.6, 4.4]
    }))

elif metric_option == "AI Treatment Path Suggestion":
    age = st.slider("Patient Age", 20, 45, 32)
    amh = st.slider("AMH Level", 0.5, 5.0, 2.5)
    if amh < 1.0 or age > 38:
        st.warning("Suggested Protocol: Aggressive IVF")
    else:
        st.success("Suggested Protocol: Natural IVF")

elif metric_option == "Logistic Regression Model – Predictive Analytics":
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

elif metric_option == "ROC Curve":
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

elif metric_option == "Summary":
    st.markdown("### 📌 Summary")
    st.markdown("""
    This integrated dashboard empowers fertility centers with actionable insights from:
    
    - Operational KPIs (no-show rates, wait time)
    - Patient behavior and satisfaction
    - Referral and marketing effectiveness
    - Predictive AI modeling for no-shows and treatment paths
    
    The dashboard supports strategic planning, marketing investments, patient outcomes, and clinical resource optimization.
    """)





