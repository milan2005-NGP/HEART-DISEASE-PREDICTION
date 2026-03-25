
# ❤️ Heart Disease Prediction - Exploratory Data Analysis (EDA)

## 📌 Project Overview

This repository contains an in-depth Exploratory Data Analysis (EDA) for **Heart Disease Prediction**. The primary goal of this project is to analyze the dataset, uncover hidden patterns, visualize relationships between various health metrics, and identify the key risk factors that contribute to heart disease.

This EDA serves as the foundational step before building any predictive machine learning models.

## 📊 Dataset Information

The dataset used in this project is the **[Name of Dataset, e.g., UCI Heart Disease Dataset]**, sourced from **[Source, e.g., Kaggle]**.

**Key Features include:**

  * **age:** Age of the patient in years
  * **sex:** Gender (1 = male; 0 = female)
  * **cp:** Chest pain type (4 values)
  * **trestbps:** Resting blood pressure (in mm Hg)
  * **chol:** Serum cholesterol in mg/dl
  * **fbs:** Fasting blood sugar \> 120 mg/dl (1 = true; 0 = false)
  * **restecg:** Resting electrocardiographic results (values 0,1,2)
  * **thalach:** Maximum heart rate achieved
  * **exang:** Exercise induced angina (1 = yes; 0 = no)
  * **oldpeak:** ST depression induced by exercise relative to rest
  * **slope:** The slope of the peak exercise ST segment
  * **ca:** Number of major vessels (0-3) colored by fluoroscopy
  * **thal:** 0 = normal; 1 = fixed defect; 2 = reversable defect
  * **target:** Presence of heart disease (1 = yes, 0 = no)

## 🎯 Objectives

  * **Data Cleaning:** Handle missing values, duplicates, and correct data types.
  * **Univariate Analysis:** Understand the distribution of individual variables (e.g., age, cholesterol levels).
  * **Bivariate/Multivariate Analysis:** Explore relationships between variables (e.g., how does age and max heart rate correlate with heart disease?).
  * **Outlier Detection:** Identify anomalies in the medical data using boxplots and statistical methods.
  * **Correlation Analysis:** Create heatmaps to find highly correlated features.

## 🛠️ Technologies & Libraries Used

  * **Python 3**
  * **Jupyter Notebook**
  * **Pandas** (Data manipulation and analysis)
  * **NumPy** (Numerical computing)
  * **Matplotlib & Seaborn** (Data visualization)

## 💡 Key Insights & Findings

*(Update this section with your actual findings after running your notebook)*

  * **Insight 1:** E.g., Males in the dataset have a higher frequency of heart disease compared to females.
  * **Insight 2:** E.g., Patients with atypical angina (chest pain type 1) are highly likely to test positive for heart disease.
  * **Insight 3:** E.g., There is a negative correlation between maximum heart rate (thalach) and age.

## 🚀 How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/heart-disease-eda.git
    ```
2.  **Navigate to the directory:**
    ```bash
    cd heart-disease-eda
    ```
3.  **Install the required dependencies:**
    *(It's recommended to set up a virtual environment first)*
    ```bash
    pip install pandas numpy matplotlib seaborn jupyter
    ```
4.  **Open Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  **Run:** Open `Heart_Disease_EDA.ipynb` and run the cells to see the analysis.

## 📁 Repository Structure

```text
├── dataset/
│   └── heart_disease_data.csv    # The dataset used for analysis
├── images/
│   └── correlation_heatmap.png   # Exported plots/visualizations
├── Heart_Disease_EDA.ipynb       # Main Jupyter Notebook with code
├── README.md                     # Project documentation
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\! Feel free to check the issues page.



