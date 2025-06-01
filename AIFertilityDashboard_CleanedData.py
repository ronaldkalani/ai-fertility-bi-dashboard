
# 📦 Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

# 📄 Page Setup
st.set_page_config(page_title="AI BI Dashboard", layout="wide")
st.title("📊 AI Fertility Centre – Business Intelligence Dashboard")
st.markdown("**Track KPIs, AI Insights, Market Intelligence, and Live Customer Feedback**")

# 📂 Load Cleaned Dataset
DATA_PATH = "C:/Users/HP/FertilityClinic/Cleaned_Appointment_Data.csv"
df = pd.read_csv(DATA_PATH)

# 1️⃣ Patient Satisfaction Monitoring
st.header("1️⃣ Patient Satisfaction Monitoring – Voice of the Customer (VoC)")
df['SatisfactionScore'] = np.random.randint(7, 11, size=len(df))
avg_satisfaction = df.groupby("TreatmentType")['SatisfactionScore'].mean().reset_index()
st.bar_chart(avg_satisfaction.set_index("TreatmentType"))

# 2️⃣ No-Show Risk Prediction
st.header("2️⃣ Predictive Analytics – No-Show Risk Model")
no_show_prob = df[['PatientId', 'No_show']].copy()
no_show_prob['NoShowProb'] = no_show_prob['No_show'].apply(lambda x: 0.85 if x == 'Yes' else 0.15)
st.dataframe(no_show_prob[['PatientId', 'NoShowProb']].head())

# 3️⃣ KPI Dashboard
st.header("3️⃣ Real-Time KPIs")
total_appointments = len(df)
no_show_rate = (df['No_show'] == 'Yes').mean() * 100
df['WaitDays'] = (pd.to_datetime(df['AppointmentDay']) - pd.to_datetime(df['ScheduledDay'])).dt.days
avg_wait = df['WaitDays'].mean()

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Appointments", f"{total_appointments:,}")
kpi2.metric("No-Show Rate", f"{no_show_rate:.2f}%")
kpi3.metric("Avg Wait Time", f"{avg_wait:.1f} days")

# 4️⃣ Self-Service Trends (simulated from SMS_received)
st.header("4️⃣ Patient Portal & Telehealth Usage")
self_service = df['SMS_received'].value_counts(normalize=True).rename({0: "No SMS", 1: "Received SMS"}) * 100
st.bar_chart(self_service)

# 5️⃣ Market Segmentation by Region
st.header("5️⃣ Market Segmentation by Region")
region_data = df['Neighbourhood'].value_counts().head(10).reset_index()
region_data.columns = ['Neighbourhood', 'Appointments']
st.bar_chart(region_data.set_index("Neighbourhood"))

# 6️⃣ Competitor Watch (Simulated)
st.header("6️⃣ Competitor Watchlist")
competitors = pd.DataFrame({
    "Clinic": ["TRIO", "CReATe", "Astra"],
    "IVF Success Rate (%)": [63, 61, 58],
    "Google Reviews": [4.8, 4.6, 4.4]
})
st.dataframe(competitors)

# 7️⃣ AI Treatment Suggestion (based on Age + simulated AMH)
st.header("7️⃣ AI Treatment Path Suggestion")
age = st.slider("Select Age", 20, 45, 32)
amh = st.slider("AMH Level (ng/mL)", 0.5, 5.0, 2.1)
if amh < 1.0 or age > 38:
    st.warning("Suggested: Aggressive IVF Protocol")
else:
    st.success("Suggested: Natural IVF or Ovulation Induction")

# 8️⃣ Referral Analytics (Simulated)
st.header("8️⃣ Outreach & Referral Pipeline")
df['ReferralSource'] = np.random.choice(["OB-GYN", "Google Ads", "Family Doctor", "Instagram", "Webinar"], size=len(df))
ref_df = df['ReferralSource'].value_counts().reset_index()
ref_df.columns = ['Source', 'Leads']
st.bar_chart(ref_df.set_index("Source"))

# 9️⃣ IVF Seasonality (by Appointment Month)
st.header("9️⃣ IVF Cycle Trends Over Months")
df['AppointmentMonth'] = pd.to_datetime(df['AppointmentDay']).dt.strftime('%b')
cycle_df = df['AppointmentMonth'].value_counts().sort_index().reset_index()
cycle_df.columns = ['Month', 'Appointments']
st.line_chart(cycle_df.set_index("Month"))

# 🔟 Reputation Summary (Simulated external ratings)
st.header("🔟 Public Trust & Transparency")
reviews = pd.DataFrame({
    'Platform': ['Google', 'RateMDs', 'Facebook'],
    'Avg Rating': [4.7, 4.6, 4.8]
})
st.bar_chart(reviews.set_index('Platform'))

# 🕵️ Competitor Review Intelligence (Sentiment Analysis)
st.header("🕵️ Competitor Review Intelligence")
clinics = {
    'TRIO Fertility': 'https://www.ratemds.com/clinic/ca-on-toronto-trio-fertility/',
    'CReATe Fertility': 'https://www.ratemds.com/clinic/ca-on-toronto-create/',
    'Mount Sinai Fertility': 'https://www.ratemds.com/clinic/ca-on-toronto-mount-sinai/'
}

def scrape_reviews(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        reviews = soup.find_all('p')
        return [r.get_text(strip=True) for r in reviews if len(r.get_text()) > 20][:10]
    except:
        return []

def analyze_sentiment(reviews):
    polarities = [TextBlob(r).sentiment.polarity for r in reviews]
    subjectivities = [TextBlob(r).sentiment.subjectivity for r in reviews]
    return sum(polarities)/len(polarities), sum(subjectivities)/len(subjectivities)

results = []
for name, url in clinics.items():
    texts = scrape_reviews(url)
    if texts:
        polarity, subjectivity = analyze_sentiment(texts)
        results.append({'Clinic': name, 'Polarity': polarity, 'Subjectivity': subjectivity})

df_sentiment = pd.DataFrame(results)
st.bar_chart(df_sentiment.set_index("Clinic")[['Polarity']])
st.table(df_sentiment)
