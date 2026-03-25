import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Global Sleep Pattern Analysis", layout="wide")

st.title("Global Sleep Pattern Analysis")
st.markdown("Explore sleep patterns, predict disorders, and analyze lifestyle clusters.")
st.divider()

try:
    df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    st.error("dataset.csv not found.")
    st.stop()

df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.divider()

df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
df['Systolic_BP'] = df['Systolic_BP'].astype(int)
df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)
df = df.drop(columns=["Person ID", "Blood Pressure"])

df_ml = df.copy()

encoders = {}
for col in ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    encoders[col] = le

st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.scatterplot(x="Sleep Duration", y="Quality of Sleep", data=df, ax=ax1)
    st.pyplot(fig1, width="content")

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.boxplot(x="Gender", y="Sleep Duration", data=df, ax=ax2)
    st.pyplot(fig2, width="content")

with col2:
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.scatterplot(x="Stress Level", y="Sleep Duration", data=df, ax=ax3)
    st.pyplot(fig3, width="content")

    fig4, ax4 = plt.subplots(figsize=(5, 3))
    sns.boxplot(x="Sleep Disorder", y="Sleep Duration", data=df, ax=ax4)
    st.pyplot(fig4, width="content")

st.divider()

st.subheader("🤒 Sleep Disorder Prediction")

X = df_ml.drop(columns=["Sleep Disorder"])
y = df_ml["Sleep Disorder"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

st.write(f"Accuracy: {acc:.2f}")
st.text("Classification Report")
st.text(classification_report(y_test, pred))

st.divider()

st.subheader("🧠 Sleep Lifestyle Clustering")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ml)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig5, ax5 = plt.subplots(figsize=(4, 2))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=clusters,
    palette="viridis",
    s=30,
    ax=ax5
)
ax5.set_title("Sleep Lifestyle Clusters")

st.pyplot(fig5, width="content")

st.divider()

st.subheader("🧾 Personal Sleep Check")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", df["Gender"].unique())
    occupation = st.selectbox("Occupation", df["Occupation"].unique())
    bmi = st.selectbox("BMI Category", df["BMI Category"].unique())

with col2:
    age = st.slider("Age", 10, 100, 30)
    sleep_duration = st.slider("Sleep Duration", 3.0, 10.0, 7.0)
    stress = st.slider("Stress Level", 1, 10, 5)

with col3:
    quality = st.slider("Quality of Sleep", 1, 10, 5)
    heart_rate = st.number_input("Heart Rate", value=int(df["Heart Rate"].mean()))
    steps = st.number_input("Daily Steps", value=int(df["Daily Steps"].mean()))
    systolic = st.number_input("Systolic BP", value=120)
    diastolic = st.number_input("Diastolic BP", value=80)

if st.button("Predict Sleep Health"):
    input_data = pd.DataFrame([{
        "Gender": encoders["Gender"].transform([gender])[0],
        "Age": age,
        "Occupation": encoders["Occupation"].transform([occupation])[0],
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality,
        "Physical Activity Level": df["Physical Activity Level"].mean(),
        "Stress Level": stress,
        "BMI Category": encoders["BMI Category"].transform([bmi])[0],
        "Heart Rate": heart_rate,
        "Daily Steps": steps,
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic
    }])

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)[0]
    result = encoders["Sleep Disorder"].inverse_transform([prediction])[0]

    if result in ["None", "No Disorder", "Healthy"]:
        st.success("✅ You have healthy sleep patterns")
    else:
        st.error(f"⚠️ Possible sleep issue detected: {result}")

st.sidebar.title("About")
st.sidebar.info("This app analyzes global sleep patterns using ML and clustering.")
