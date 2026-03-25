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

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("❤️ Heart Disease Prediction & Analysis")
st.markdown("Explore heart health patterns, predict disease risk, and analyze lifestyle clusters.")
st.divider()

# ── Load Dataset ──────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    st.error("dataset.csv not found. Please place your dataset in the same folder.")
    st.stop()

# Drop ID column if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Standardise target column name
target_candidates = [c for c in df.columns if "heart" in c.lower() or "target" in c.lower() or "disease" in c.lower()]
TARGET = target_candidates[0] if target_candidates else df.columns[-1]

df[TARGET] = df[TARGET].fillna(0)

st.subheader("Dataset Preview")
st.dataframe(df.head())
st.divider()

# ── Encode categoricals ───────────────────────────────────────────────────────
df_ml = df.copy()
encoders = {}
cat_cols = df_ml.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = le

# Encode target if object
if df_ml[TARGET].dtype == object:
    le_target = LabelEncoder()
    df_ml[TARGET] = le_target.fit_transform(df_ml[TARGET].astype(str))
    encoders[TARGET] = le_target

# ── EDA ───────────────────────────────────────────────────────────────────────
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    # Age distribution by heart disease
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.histplot(data=df, x="Age", hue=TARGET, bins=20, kde=True, ax=ax1)
    ax1.set_title("Age Distribution by Heart Disease")
    st.pyplot(fig1)

    # Cholesterol vs Max HR
    hr_col = next((c for c in df.columns if "hr" in c.lower() or "heart rate" in c.lower() or "max" in c.lower()), None)
    chol_col = next((c for c in df.columns if "chol" in c.lower()), None)
    if hr_col and chol_col:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.scatterplot(x=chol_col, y=hr_col, hue=TARGET, data=df, ax=ax2, palette="coolwarm")
        ax2.set_title(f"{chol_col} vs {hr_col}")
        st.pyplot(fig2)

with col2:
    # Heart disease count
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    val_counts = df[TARGET].value_counts()
    ax3.pie(val_counts, labels=val_counts.index, autopct="%1.1f%%", startangle=90, colors=["#e74c3c", "#2ecc71"])
    ax3.set_title("Heart Disease Distribution")
    st.pyplot(fig3)

    # BP distribution
    bp_col = next((c for c in df.columns if "bp" in c.lower() or "blood" in c.lower()), None)
    if bp_col:
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        sns.boxplot(x=TARGET, y=bp_col, data=df, ax=ax4, palette="Set2")
        ax4.set_title(f"{bp_col} by Heart Disease")
        st.pyplot(fig4)

# Correlation heatmap
st.markdown("#### 🔥 Feature Correlation Heatmap")
fig_corr, ax_corr = plt.subplots(figsize=(10, 4))
corr = df_ml.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap="RdYlGn", linewidths=0.3, ax=ax_corr)
st.pyplot(fig_corr)

st.divider()

# ── Classification ────────────────────────────────────────────────────────────
st.subheader("🫀 Heart Disease Prediction Model")

X = df_ml.drop(columns=[TARGET])
y = df_ml[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

m1, m2, m3 = st.columns(3)
m1.metric("✅ Model Accuracy", f"{acc * 100:.1f}%")
m2.metric("🧪 Test Samples", len(X_test))
m3.metric("📋 Features Used", X.shape[1])

st.text("Classification Report")
st.text(classification_report(y_test, pred))

# Feature importance
st.markdown("#### 🏆 Top Feature Importances")
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
fig_fi, ax_fi = plt.subplots(figsize=(8, 3))
feat_imp.plot(kind="bar", ax=ax_fi, color="#e74c3c")
ax_fi.set_title("Top 10 Feature Importances")
ax_fi.set_ylabel("Importance")
st.pyplot(fig_fi)

st.divider()

# ── Clustering ────────────────────────────────────────────────────────────────
st.subheader("🧠 Heart Risk Lifestyle Clustering")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ml)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig5, ax5 = plt.subplots(figsize=(6, 4))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="Set1", s=40, ax=ax5)
ax5.set_title("Heart Risk Lifestyle Clusters (PCA)")
ax5.set_xlabel("Principal Component 1")
ax5.set_ylabel("Principal Component 2")
st.pyplot(fig5)


st.divider()
