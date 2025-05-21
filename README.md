# 💧 Water Potability Prediction

This project aims to determine whether water is safe for human consumption based on key physical and chemical properties. Using machine learning models like **Random Forest** and **Naive Bayes**, the project performs preprocessing, outlier handling, visualization, model tuning, and evaluation. The best-performing model is deployed using **Render**.

🔗 **Deployed App**: [Click Here to Try It](https://water-potability-yfqc.onrender.com/)  
📂 **Dataset Source**: Provided locally in `data/water_potability.csv`

---

## 📊 Problem Statement

Access to clean and safe drinking water is crucial for human health. This project classifies water samples as **potable (1)** or **not potable (0)** based on chemical indicators such as pH, hardness, sulfate content, and more.

---

## 🧪 Key Features & Workflow

### 1. Exploratory Data Analysis (EDA)
- Dataset inspection: shape, datatypes, and missing values.
- Visualized missing data and distributions.
- Created correlation matrix to understand feature relationships.

### 2. Data Cleaning
- Handled missing values using **median imputation** for features like `ph`, `Sulfate`, and `Trihalomethanes`.
- Created new features:
  - `TDS_to_Hardness = Solids / Hardness`
  - `Chloramine_to_Trihalomethanes = Chloramines / Trihalomethanes`

### 3. Outlier Detection & Removal
- Used **IQR method** to detect and remove outliers.
- Visualized with boxplots before and after removal.
- Performed outlier removal column-wise for cleaner inputs.

### 4. Feature Scaling
- Applied **StandardScaler** to normalize numerical values for model compatibility.

### 5. Modeling
- **Random Forest Classifier** with hyperparameter tuning using `RandomizedSearchCV`.
  - **Accuracy**: ~66.6%
- **Gaussian Naive Bayes** for baseline comparison.
  - **Accuracy**: ~42.2%
- Evaluated models using **precision, recall, and f1-score**.

### 6. Feature Importance
- Extracted and visualized top contributing features from the Random Forest model.

### 7. Deployment
- Saved the best model and scaler using `joblib`.
- Deployed the app on **Render** with a simple UI built using Flask/Streamlit/FastAPI.

---

## 🚀 Deployment

The model and preprocessing pipeline are hosted on **Render** with a user-friendly interface.

🔗 **Live App**: [https://water-potability-yfqc.onrender.com/](https://water-potability-yfqc.onrender.com/)

---

## 🧠 Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **Joblib**
- **Render** (for deployment)

---


## 📁 Project Structure

```bash
.
├── data/
│   └── water_potability.csv
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── analysis_and_modeling.ipynb
├── app/ 
│   └── main.py
├── static/
│   ├── index.html
│   └── results.html
├── templates/
│   └── style.css
├── requirements.txt
└── README.md
