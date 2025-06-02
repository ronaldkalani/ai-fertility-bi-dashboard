## Project Title
AI-Enhanced Fertility BI Dashboard: Predictive Insights & Personalized Treatment Planning

## Goal
To empower fertility clinic decision-makers with a real-time, interactive Business Intelligence (BI) and Artificial Intelligence (AI) platform that:

- Improves operational efficiency

- Enhances patient satisfaction

- Predicts no-shows

- Personalizes treatment pathways

- Benchmarks against competitors

## Intended Audience

1. Fertility Clinic Executives and Managers

2. Clinical Operations Teams

3. Business Analysts & Data Scientists in Healthcare

4. Marketing & Outreach Coordinators

5. Physicians and Fertility Counselors

## Strategy & Pipeline Steps

# 1. Data Integration
- Load cleaned appointment records (cleaned_appointment.csv)

- Simulate missing fields like treatment type, SMS received, referral source

# 2. Preprocessing
- Encode no-show labels into binary

- Engineer features: Age, WaitDays, SMS_received

- Normalize features using StandardScaler

# 3. Machine Learning
- Train logistic regression model to predict no-show risk

- Evaluate using classification report, confusion matrix, and ROC curve

- Generate AI treatment suggestions using rules based on age + AMH levels

# 4. Visualization – Options
A. Streamlit
- Dynamic sidebar with metric selection

- Visualizations: bar charts, line charts, ROC curve, treatment suggestor

- Hosted live: https://ai-fertility-bi-dashboard.streamlit.app

B. Tableau
Alternate dashboard version using Tableau for:

- Satisfaction score breakdown

- Monthly trend analysis

- Referral source and regional demand heatmaps

- Publish via Tableau Public or Server for executive use

# 5. Insights & Automation
- Actionable business decisions tied to each metric

- Real-time feedback for front-desk, marketing, and clinical teams

## Challenges
- Class imbalance in no-show prediction reduced model precision

- Limited AMH/patient data constrained treatment path ML automation

- Visual consistency between Tableau and Streamlit needed additional UI normalization

- External review data (Google, RateMDs) lacked structured formats for NLP

## Problem Statement
Fertility clinics face high no-show rates, uneven referral performance, and limited decision support tools for optimizing treatment planning and operational strategy. This leads to patient dissatisfaction, lost revenue, and inefficient staff utilization.

## Dataset
cleaned_appointment.csv

Key fields: Age, WaitDays, SMS_received, TreatmentType, ReferralSource, No_show, SatisfactionScore, AppointmentDay

## MACHINE LEARNING PREDICTION & OUTCOMES
- Model Used: Logistic Regression

- Features: Age, WaitDays, SMS_received

- Target: No_show_binary

- Accuracy: ~50% (initial, before rebalancing)

Insights:

- SMS reminders correlated with lower no-shows

- Longer wait time associated with higher no-shows

- Younger patients showed lower satisfaction with donor programs

Suggestions:

- Apply SMOTE to rebalance dataset

- Explore XGBoost or Gradient Boosting for higher precision

- Implement patient re-engagement automation using model outputs

## Trailer Documentation
- Summary Document: [✓ Loaded: AIFertilityDashboard_Summary.txt]

- Classification Report & ROC Curve: Displayed via Streamlit Dashboard

- Competitor Sentiment: Shown through positive/negative analysis of public review excerpts

- Satisfaction Trend Tracker: Interactive charts across treatment modalities

## Conceptual Enhancement – AGI Integration
Looking forward, Artificial General Intelligence (AGI) could:

- Interpret multi-modal data (lab results, ultrasound images, genomics)

- Evolve dynamic treatment protocols based on global fertility data

- Engage patients through empathetic, personalized chatbots for ongoing fertility tracking

- Auto-optimize clinic scheduling using real-time traffic, hormone cycles, and staff burnout models

## Reference
- Scikit-learn Documentation

- Streamlit Documentation

- Tableau Public

- Imbalanced-learn (SMOTE)

- Hugging Face Transformers

- Fertility KPIs – CFAS Benchmarks & CARTR+ Database
