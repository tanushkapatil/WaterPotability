💧 Water Potability Prediction
This project aims to determine whether water is safe for human consumption based on key physical and chemical properties. Using machine learning models like Random Forest and Naive Bayes, this project performs preprocessing, outlier handling, visualization, model tuning, and evaluation. The best-performing model is deployed using Render.

🔗 Deployed App: Visit the Water Potability Classifier
📂 Dataset Source: Provided locally in data/water_potability.csv

📊 Problem Statement
Access to clean and safe drinking water is crucial for human health. This project classifies water samples as potable (1) or not potable (0) based on chemical indicators such as pH, hardness, sulfate content, and more.

📁 Project Structure
bash
Copy
Edit
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
🧪 Key Features & Workflow
1. Exploratory Data Analysis (EDA)
Dataset inspection: shape, datatypes, missing values.

Visualized missing data and distributions.

Correlation matrix to understand feature relationships.

2. Data Cleaning
Handled missing values using median imputation for features like ph, Sulfate, and Trihalomethanes.

Created new features like:

TDS_to_Hardness = Solids / Hardness

Chloramine_to_Trihalomethanes

3. Outlier Detection & Removal
Applied IQR method.

Visualized boxplots before and after removal.

Removed outliers column-wise for reliable model input.

4. Feature Scaling
Applied StandardScaler to normalize numerical values for model compatibility.

5. Modeling
Random Forest Classifier with hyperparameter tuning using RandomizedSearchCV.

Accuracy: ~66.6%

Gaussian Naive Bayes for baseline comparison.

Accuracy: ~42.2%

Evaluated models with precision, recall, f1-score.

6. Feature Importance
Visualized using feature importance from the Random Forest model to identify impactful features.

7. Deployment
Saved the best model and scaler using joblib.

Deployed on Render for public access and usability.

🚀 Deployment
The model and preprocessing pipeline are deployed on Render with a user-friendly interface (if built using Flask/Streamlit/FastAPI).

🔗 Live App: Click Here to Try It

🧠 Tech Stack
Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Joblib

Render (for deployment)
