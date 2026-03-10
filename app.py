# one_cell_credit_gradio.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import gradio as gr

# 1) Load dataset
df = pd.read_csv("german_credit.csv")

# 2) Pick only Age and Credit Amount columns
# Change these names if your CSV uses slightly different column names
age_col = "Age (years)"
credit_col = "Credit Amount"

# Auto-fix common alternative names
for c in df.columns:
    if c.lower().strip() in ["age", "age (years)", "age_years"]:
        age_col = c
    if c.lower().strip() in ["credit amount", "credit_amount", "creditamount"]:
        credit_col = c

# 3) Features and target
X = df[[age_col, credit_col]].copy()
y = df["Creditability"].astype(int)

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5) Build model
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, random_state=42))
])

pipe.fit(X_train, y_train)

# 6) Check accuracy
test_acc = pipe.score(X_test, y_test)
print("Using features:", [age_col, credit_col])
print("Test Accuracy:", round(test_acc, 4))

# 7) Save model
joblib.dump({
    "pipeline": pipe,
    "feature_cols": [age_col, credit_col]
}, "credit_age_amount_lr.joblib")

# 8) Prediction function
def predict_creditability(age, credit_amount):
    row = pd.DataFrame([[age, credit_amount]], columns=[age_col, credit_col])
    pred = pipe.predict(row)[0]
    prob = pipe.predict_proba(row)[0, 1]
    label = "Good" if pred == 1 else "Bad"
    return f"Prediction: {label} (1=Good, 0=Bad), P(Good) = {prob:.3f}"

# 9) Gradio interface
demo = gr.Interface(
    fn=predict_creditability,
    inputs=[
        gr.Number(label=age_col, value=35),
        gr.Number(label=credit_col, value=2000)
    ],
    outputs="text",
    title="German Credit Prediction using Age and Credit Amount",
    description="Enter Age and Credit Amount to predict creditability."
)

# 10) Launch
demo.launch()