# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Generate Synthetic Dataset (You can replace this with a real dataset)
np.random.seed(42)
num_samples = 5000

data = pd.DataFrame({
    'Age': np.random.randint(18, 60, num_samples),
    'Gender': np.random.choice(['Male', 'Female'], num_samples),
    'Income': np.random.randint(30000, 150000, num_samples),
    'Occupation': np.random.choice(['Engineer', 'Doctor', 'Artist', 'Business', 'Student'], num_samples),
    'Website_Visits': np.random.randint(1, 20, num_samples),
    'Time_on_Site': np.random.uniform(1, 10, num_samples),
    'Email_Interactions': np.random.randint(0, 10, num_samples),
    'Ad_Clicks': np.random.randint(0, 5, num_samples),
    'Call_Response': np.random.choice(['Yes', 'No'], num_samples),
    'Past_Purchases': np.random.randint(0, 5, num_samples),
    'Interest_Score': np.random.uniform(0, 1, num_samples),
    'Converted': np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # 0 = No Purchase, 1 = Purchase
})

# Encode Categorical Features
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data['Occupation'] = LabelEncoder().fit_transform(data['Occupation'])
data['Call_Response'] = LabelEncoder().fit_transform(data['Call_Response'])

# Splitting Data into Train & Test
X = data.drop(columns=['Converted'])
y = data['Converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train & Evaluate Models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
    print(classification_report(y_test, y_pred))
    print("="*50)

# Feature Importance (for Random Forest)
rf_model = models["Random Forest"]
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', title="Feature Importance")
plt.show()
