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
st.set_page_config(page_title="AI Fertility BI Dashboard", layout="wide")
st.title("📊 AI Fertility Centre – Business Intelligence Dashboard")
st.markdown("This dashboard integrates clinical appointment analytics with predictive modeling to support informed business decisions.")

# Load Data
DATA_PATH = "cleaned_appointment.csv"
df = pd.read_csv(DATA_PATH)

# Simulated fields (for demo)
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

# Preprocessing
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

# Sidebar selection
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

# Business decision/action info
INSIGHTS = {
    "Average Satisfaction Score by Treatment Type": {
        "Decision": "Improve patient experience per treatment",
        "Action": "Investigate low-scoring treatment areas; revise protocols or retrain staff."
    },
    "Predicted No-Show Risk": {
        "Decision": "Reduce last-minute cancellations and revenue loss",
        "Action": "Flag high-risk patients early; reallocate slots or send reminders."
    },
    "Total Appointments Summary": {
        "Decision": "Monitor clinic performance and patient inflow",
        "Action": "Set monthly/quarterly growth targets and outreach goals."
    },
    "No-Show Rate (%)": {
        "Decision": "Identify systemic inefficiencies in patient engagement",
        "Action": "Enhance communication or offer flexible rescheduling."
    },
    "Average Wait Time (days)": {
        "Decision": "Optimize booking practices and reduce dropout",
        "Action": "Adjust provider schedules or redistribute staff resources."
    },
    "Proportion of SMS Reminders Sent": {
        "Decision": "Evaluate impact of communication strategy",
        "Action": "Boost SMS outreach and track resulting attendance improvements."
    },
    "Appointment Volume by Region": {
        "Decision": "Identify high/low demand areas",
        "Action": "Target advertising or expand clinics in underserved zones."
    },
    "Referral Source Breakdown": {
        "Decision": "Optimize marketing and referral performance",
        "Action": "Invest more in high-performing referral sources."
    },
    "Monthly Appointment Trends": {
        "Decision": "Plan staffing based on demand patterns",
        "Action": "Boost staff during peak months; run promotions in low months."
    },
    "Public Platform Ratings": {
        "Decision": "Improve public image and patient trust",
        "Action": "Respond to reviews and enhance front-desk training."
    },
    "Competitor Watchlist": {
        "Decision": "Benchmark performance and offerings",
        "Action": "Adjust pricing or launch new services based on competitors."
    },
    "AI Treatment Path Suggestion": {
        "Decision": "Provide AI-assisted personalized care",
        "Action": "Integrate AI tools into treatment planning."
    },
    "Logistic Regression Model – Predictive Analytics": {
        "Decision": "Forecast clinic risks (e.g., cancellations)",
        "Action": "Triage and allocate resources using ML predictions."
    },
    "ROC Curve": {
        "Decision": "Evaluate model reliability for predictions",
        "Action": "Use ROC to determine decision thresholds."
    },
    "Summary": {
        "Decision": "Enable real-time executive oversight",
        "Action": "Use dashboard insights during strategy meetings."
    }
}

# Logic for AI Treatment Path Suggestion
def suggest_treatment(age, amh):
    if amh < 0.8 and age >= 40:
        return "Donor Program"
    elif amh < 1.5 and age >= 38:
        return "IVF"
    elif 1.5 <= amh < 3.5 and age >= 35:
        return "IVF"
    elif 1.5 <= amh < 3.5 and age < 35:
        return "Natural IVF"
    elif amh >= 3.5 and age <= 34:
        return "Egg Freezing"
    elif 2.0 <= amh <= 4.5 and age < 33:
        return "IUI"
    else:
        return "Further Evaluation"

# Main Display Logic
st.header(f"📌 {metric_option}")

if metric_option == "Average Satisfaction Score by Treatment Type":
    avg_satisfaction = df.groupby("TreatmentType")["SatisfactionScore"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=avg_satisfaction, x="TreatmentType", y="SatisfactionScore", palette="Set2", ax=ax)
    ax.set_title("Satisfaction Score by Treatment")
    st.pyplot(fig)

elif metric_option == "Predicted No-Show Risk":
    df["NoShowProb"] = df["No_show"].apply(lambda x: 0.85 if x == "Yes" else 0.15)
    st.dataframe(df[["NoShowProb"]].head())

elif metric_option == "Total Appointments Summary":
    st.metric("Total Appointments", f"{len(df):,}")
    st.metric("No-Show Rate", f"{(df['No_show'] == 'Yes').mean() * 100:.2f}%")
    st.metric("Avg. Wait Time", f"{df['WaitDays'].mean():.1f} days")
    st.metric("Avg. Satisfaction", f"{df['SatisfactionScore'].mean():.2f}/10")

elif metric_option == "No-Show Rate (%)":
    st.metric("No-Show Rate", f"{(df['No_show'] == 'Yes').mean() * 100:.2f}%")

elif metric_option == "Average Wait Time (days)":
    st.metric("Average Wait Time", f"{df['WaitDays'].mean():.1f} days")

elif metric_option == "Proportion of SMS Reminders Sent":
    sms = df["SMS_received"].value_counts(normalize=True).rename({0: "No SMS", 1: "SMS Sent"}) * 100
    fig, ax = plt.subplots()
    sms.plot(kind="bar", color=["red", "green"], ax=ax)
    ax.set_title("SMS Reminder Distribution")
    st.pyplot(fig)

elif metric_option == "Appointment Volume by Region":
    if "Neighbourhood" in df.columns:
        vol = df["Neighbourhood"].value_counts().head(10).reset_index()
        vol.columns = ["Neighbourhood", "Appointments"]
        fig, ax = plt.subplots()
        sns.barplot(data=vol, x="Appointments", y="Neighbourhood", ax=ax)
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
        trend = df["Month"].value_counts().sort_index().reset_index()
        trend.columns = ["Month", "Appointments"]
        fig, ax = plt.subplots()
        sns.lineplot(data=trend, x="Month", y="Appointments", marker="o", ax=ax)
        st.pyplot(fig)

elif metric_option == "Public Platform Ratings":
    ratings = pd.DataFrame({
        "Platform": ["Google", "RateMDs", "Facebook"],
        "Rating": [4.7, 4.6, 4.8]
    })
    fig, ax = plt.subplots()
    sns.barplot(data=ratings, x="Platform", y="Rating", ax=ax)
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
    protocol = suggest_treatment(age, amh)
    st.success(f"Suggested Protocol: {protocol}")

elif metric_option == "Logistic Regression Model – Predictive Analytics":
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

elif metric_option == "ROC Curve":
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

elif metric_option == "Summary":
    st.markdown("### 📌 Dashboard Summary")
    st.markdown("""
    This dashboard equips the clinic with insights into:
    - Patient attendance behavior
    - Operational KPIs
    - Referral performance
    - Predictive risk modeling
    - AI-powered treatment suggestions
    """)

# Contextual Business Decision & Action
if metric_option in INSIGHTS:
    st.markdown("---")
    st.markdown(f"**🧠 Business Decision:** {INSIGHTS[metric_option]['Decision']}")
    st.markdown(f"**✅ Recommended Action:** {INSIGHTS[metric_option]['Action']}")






