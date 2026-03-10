import os
import joblib
import pandas as pd
import gradio as gr

artifact = joblib.load("credit_model.joblib")

model = artifact.get("model")
age_col = artifact.get("age_col")
credit_col = artifact.get("credit_col")

def predict_credit(age, credit_amount):

    if age is None or credit_amount is None:
        return "Please enter both Age and Credit Amount."

    age = float(age)
    credit_amount = float(credit_amount)

    data = pd.DataFrame([[age, credit_amount]], columns=[age_col, credit_col])

    pred = model.predict(data)[0]
    label = "Good Credit" if pred == 1 else "Bad Credit"

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(data)[0][1]
        return f"{label} | Probability of Good Credit: {prob:.3f}"

    return label

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
