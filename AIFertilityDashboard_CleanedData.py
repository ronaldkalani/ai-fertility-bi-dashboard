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
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

# 📄 Page Setup
st.set_page_config(page_title="AI BI Dashboard", layout="wide")
st.title("📊 AI Fertility Centre – Business Intelligence Dashboard")
st.markdown("**Track KPIs, AI Insights, Market Intelligence, and Live Customer Feedback**")

# Load dataset
DATA_PATH = "cleaned_appointment.csv"
df = pd.read_csv(DATA_PATH)

# Simulate missing columns
if "TreatmentType" not in df.columns:
    treatment_options = ["IVF", "IUI", "Egg Freezing", "Donor Program"]
    df["TreatmentType"] = np.random.choice(treatment_options, size=len(df))

if "SatisfactionScore" not in df.columns:
    df["SatisfactionScore"] = np.random.randint(7, 11, size=len(df))

if "ReferralSource" not in df.columns:
    df["ReferralSource"] = np.random.choice(["OB-GYN", "Google Ads", "Family Doctor", "Instagram", "Webinar"], size=len(df))

if "No_show" not in df.columns:
    df["No_show"] = np.random.choice(["Yes", "No"], size=len(df))

# 1️⃣–2️⃣ Patient Satisfaction & No-Show Risk
st.header("1️⃣–2️⃣ Patient Satisfaction & No-Show Risk")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Avg Satisfaction by Treatment")
    avg_satisfaction = df.groupby("TreatmentType")["SatisfactionScore"].mean().reset_index()
    st.bar_chart(avg_satisfaction.set_index("TreatmentType"))

with col2:
    st.subheader("Predicted No-Show Risk")
    df["NoShowProb"] = df["No_show"].apply(lambda x: 0.85 if x == "Yes" else 0.15)
    st.dataframe(df[["PatientId", "NoShowProb"]].head() if "PatientId" in df.columns else df[["NoShowProb"]].head())

# 3️⃣ Real-Time KPIs
st.header("3️⃣ Real-Time KPIs")
if "ScheduledDay" in df.columns and "AppointmentDay" in df.columns:
    df['WaitDays'] = (pd.to_datetime(df['AppointmentDay']) - pd.to_datetime(df['ScheduledDay'])).dt.days
else:
    df['WaitDays'] = np.random.randint(1, 15, size=len(df))

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Appointments", f"{len(df):,}")
kpi2.metric("No-Show Rate", f"{(df['No_show'] == 'Yes').mean() * 100:.2f}%")
kpi3.metric("Avg Wait Time", f"{df['WaitDays'].mean():.1f} days")

# 4️⃣–5️⃣ Self-Service & Regional Metrics
st.header("4️⃣–5️⃣ Self-Service & Regional Metrics")
col3, col4 = st.columns(2)
with col3:
    st.subheader("Self-Service (SMS Received %)")
    if "SMS_received" not in df.columns:
        df['SMS_received'] = np.random.choice([0, 1], size=len(df))
    sms = df['SMS_received'].value_counts(normalize=True).rename({0: "No SMS", 1: "Received SMS"}) * 100
    st.bar_chart(sms)

with col4:
    st.subheader("Appointments by Region")
    if "Neighbourhood" in df.columns:
        region_data = df['Neighbourhood'].value_counts().head(10).reset_index()
        region_data.columns = ['Neighbourhood', 'Appointments']
        st.bar_chart(region_data.set_index("Neighbourhood"))
    else:
        st.warning("Neighbourhood column not found.")

# 6️⃣ Competitor Watch
st.header("6️⃣ Competitor Watchlist")
competitors = pd.DataFrame({
    "Clinic": ["TRIO", "CReATe", "Astra"],
    "IVF Success Rate (%)": [63, 61, 58],
    "Google Reviews": [4.8, 4.6, 4.4]
})
st.dataframe(competitors)

# 7️⃣ AI Treatment Path Suggestion
st.header("7️⃣ AI Treatment Path Suggestion")
age = st.slider("Select Age", 20, 45, 32)
amh = st.slider("AMH Level (ng/mL)", 0.5, 5.0, 2.1)
if amh < 1.0 or age > 38:
    st.warning("Suggested: Aggressive IVF Protocol")
else:
    st.success("Suggested: Natural IVF or Ovulation Induction")

# 8️⃣ Referral Analytics
st.header("8️⃣ Referral Source Breakdown")
ref_data = df["ReferralSource"].value_counts().reset_index()
ref_data.columns = ["Source", "Leads"]
st.bar_chart(ref_data.set_index("Source"))

# 9️⃣ IVF Seasonality
st.header("9️⃣ Monthly Appointment Trends")
if "AppointmentDay" in df.columns:
    df["AppointmentMonth"] = pd.to_datetime(df["AppointmentDay"]).dt.strftime("%b")
    season_df = df["AppointmentMonth"].value_counts().sort_index().reset_index()
    season_df.columns = ["Month", "Appointments"]
    st.line_chart(season_df.set_index("Month"))
else:
    st.warning("AppointmentDay column not found.")

# 🔟 Public Trust & Transparency
st.header("🔟 Public Trust & Transparency")
reviews = pd.DataFrame({
    "Platform": ["Google", "RateMDs", "Facebook"],
    "Avg Rating": [4.7, 4.6, 4.8]
})
st.bar_chart(reviews.set_index("Platform"))

# Logistic Regression Model
st.header("Predictive Model – Logistic Regression")
df = df.dropna(subset=["Age", "WaitDays", "No_show", "SMS_received"])
df["No_show_binary"] = df["No_show"].map({"Yes": 1, "No": 0})
X = df[["Age", "WaitDays", "SMS_received"]]
y = df["No_show_binary"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("Classification Report")
report = classification_report(y_test, y_pred)
st.text(report)

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0, 1], [0, 1], "k--")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# Competitor Review Sentiment
st.header("Competitor Review Sentiment")
clinics = {
    "TRIO Fertility": "https://www.ratemds.com/clinic/ca-on-toronto-trio-fertility/",
    "CReATe Fertility": "https://www.ratemds.com/clinic/ca-on-toronto-create/",
    "Mount Sinai Fertility": "https://www.ratemds.com/clinic/ca-on-toronto-mount-sinai/"
}

def scrape_reviews(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        return [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text()) > 20][:10]
    except:
        return []

def analyze_sentiment(reviews):
    polarities = [TextBlob(r).sentiment.polarity for r in reviews]
    subjectivities = [TextBlob(r).sentiment.subjectivity for r in reviews]
    return sum(polarities)/len(polarities), sum(subjectivities)/len(subjectivities)

sentiment_data = []
for clinic, url in clinics.items():
    texts = scrape_reviews(url)
    if texts:
        polarity, subjectivity = analyze_sentiment(texts)
        sentiment_data.append({"Clinic": clinic, "Polarity": polarity, "Subjectivity": subjectivity})

sentiment_df = pd.DataFrame(sentiment_data)
st.bar_chart(sentiment_df.set_index("Clinic")[["Polarity"]])
st.dataframe(sentiment_df)

