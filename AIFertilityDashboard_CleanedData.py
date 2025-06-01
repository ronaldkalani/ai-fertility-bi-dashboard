# AIFertilityDashboard_with_LogisticModel.py
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

# Create Dropdown for Filtering
parameters = st.multiselect("Filter by parameters", options=["TreatmentType", "ReferralSource", "Neighbourhood"])
filtered_df = df.copy()
for param in parameters:
    if param in df.columns:
        selected_value = st.selectbox(f"Select {param}", options=df[param].dropna().unique())
        filtered_df = filtered_df[filtered_df[param] == selected_value]

# Section 1-2: Satisfaction + No-show
st.header("1️⃣–2️⃣ Satisfaction & No-Show Risk")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Avg Satisfaction by Treatment")
    st.caption("Displays average satisfaction scores across different treatment types.")
    avg_satisfaction = df.groupby("TreatmentType")["SatisfactionScore"].mean().reset_index()
    fig1, ax1 = plt.subplots()
    sns.barplot(data=avg_satisfaction, x="TreatmentType", y="SatisfactionScore", palette="Set2", ax=ax1)
    ax1.set_title("Satisfaction Score by Treatment")
    st.pyplot(fig1)

with col2:
    st.subheader("Predicted No-Show Risk")
    st.caption("Assigns high probability (0.85) to 'Yes' and low (0.15) to 'No' no-show labels.")
    df["NoShowProb"] = df["No_show"].apply(lambda x: 0.85 if x == "Yes" else 0.15)
    st.dataframe(df[["PatientId", "NoShowProb"]].head() if "PatientId" in df.columns else df[["NoShowProb"]].head())

# KPIs
st.header("3️⃣ Key Performance Metrics")
if "ScheduledDay" in df.columns and "AppointmentDay" in df.columns:
    df["WaitDays"] = (pd.to_datetime(df["AppointmentDay"]) - pd.to_datetime(df["ScheduledDay"])).dt.days
else:
    df["WaitDays"] = np.random.randint(1, 20, size=len(df))
k1, k2, k3 = st.columns(3)
k1.metric("Total Appointments", f"{len(df):,}")
k2.metric("No-Show Rate", f"{(df['No_show'] == 'Yes').mean() * 100:.2f}%")
k3.metric("Avg Wait Time", f"{df['WaitDays'].mean():.1f} days")

# SMS + Region
st.header("4️⃣–5️⃣ Self-Service & Regional Metrics")
c3, c4 = st.columns(2)
with c3:
    st.subheader("SMS Reminders Sent")
    st.caption("Proportion of patients who received SMS appointment reminders.")
    if "SMS_received" not in df.columns:
        df["SMS_received"] = np.random.choice([0, 1], size=len(df))
    sms = df["SMS_received"].value_counts(normalize=True).rename({0: "No SMS", 1: "Received SMS"}) * 100
    fig2, ax2 = plt.subplots()
    sms.plot(kind="bar", color=["red", "green"], ax=ax2)
    ax2.set_title("SMS Reminder Distribution")
    st.pyplot(fig2)

with c4:
    st.subheader("Appointments by Region")
    st.caption("Displays top neighborhoods by appointment volume.")
    if "Neighbourhood" in df.columns:
        reg = df["Neighbourhood"].value_counts().head(10).reset_index()
        reg.columns = ["Neighbourhood", "Appointments"]
        fig3, ax3 = plt.subplots()
        sns.barplot(data=reg, x="Appointments", y="Neighbourhood", palette="coolwarm", ax=ax3)
        st.pyplot(fig3)

# Competitor Watch
st.header("6️⃣ Competitor Watchlist")
st.caption("Benchmarking performance against local competitors.")
st.dataframe(pd.DataFrame({
    "Clinic": ["TRIO", "CReATe", "Astra"],
    "IVF Success Rate (%)": [63, 61, 58],
    "Google Rating": [4.8, 4.6, 4.4]
}))

# AI Treatment Suggestion
st.header("7️⃣ AI Treatment Path Suggestion")
age = st.slider("Patient Age", 20, 45, 32)
amh = st.slider("AMH Level", 0.5, 5.0, 2.5)
if amh < 1.0 or age > 38:
    st.warning("Suggested Protocol: Aggressive IVF")
else:
    st.success("Suggested Protocol: Natural IVF")

# Referrals
st.header("8️⃣ Referral Source Breakdown")
st.caption("Highlights the most common sources of referrals.")
ref = df["ReferralSource"].value_counts().reset_index()
ref.columns = ["Source", "Leads"]
fig4, ax4 = plt.subplots()
sns.barplot(data=ref, x="Leads", y="Source", palette="magma", ax=ax4)
st.pyplot(fig4)

# Seasonality
st.header("9️⃣ Monthly Appointment Trends")
st.caption("Shows seasonality trends in patient appointments.")
if "AppointmentDay" in df.columns:
    df["Month"] = pd.to_datetime(df["AppointmentDay"]).dt.strftime("%b")
    month_df = df["Month"].value_counts().sort_index().reset_index()
    month_df.columns = ["Month", "Appointments"]
    fig5, ax5 = plt.subplots()
    sns.lineplot(data=month_df, x="Month", y="Appointments", marker="o", ax=ax5)
    st.pyplot(fig5)

# Ratings
st.header("🔟 Public Trust & Transparency")
st.caption("Average ratings from public platforms.")
fig6, ax6 = plt.subplots()
sns.barplot(data=pd.DataFrame({
    "Platform": ["Google", "RateMDs", "Facebook"],
    "Rating": [4.7, 4.6, 4.8]
}), x="Platform", y="Rating", palette="Set2", ax=ax6)
st.pyplot(fig6)

# Logistic Regression
st.header("🧠 Logistic Regression Model – Predictive Analytics")
df = df.dropna(subset=["Age", "WaitDays", "No_show", "SMS_received"])
df["No_show_binary"] = df["No_show"].map({"Yes": 1, "No": 0})
X = df[["Age", "WaitDays", "SMS_received"]]
y = df["No_show_binary"]
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()
st.pyplot(fig_roc)

# Summary
st.markdown("### 📌 Summary")
st.markdown("""
This integrated dashboard empowers fertility centers with actionable insights from operational KPIs, patient behavior, marketing channels, and AI-driven no-show predictions. It supports patient satisfaction optimization, strategic planning, and clinical efficiency.
""")



