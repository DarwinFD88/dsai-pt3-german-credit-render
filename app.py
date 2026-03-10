import os
import joblib
import pandas as pd
import gradio as gr

# Load model
artifact = joblib.load("credit_model.joblib")

model = artifact["model"]
age_col = artifact["age_col"]
credit_col = artifact["credit_col"]

# Prediction function
def predict_credit(age, credit_amount):
    data = pd.DataFrame([[age, credit_amount]], columns=[age_col, credit_col])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    label = "Good Credit" if pred == 1 else "Bad Credit"
    return f"{label} | Probability of Good Credit: {prob:.3f}"

# Gradio UI
demo = gr.Interface(
    fn=predict_credit,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Credit Amount")
    ],
    outputs="text",
    title="German Credit Prediction",
    description="Predict creditability using Age and Credit Amount"
)

# Launch for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
