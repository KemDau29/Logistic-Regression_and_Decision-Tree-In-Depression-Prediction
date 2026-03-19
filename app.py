import pickle as pk
import gradio as gr
import pandas as pd

# Load trained models
with open("logistic_regression.pkl", "rb") as f:
    artifact_lr = pk.load(f)
with open("decision_tree.pkl", "rb") as f:
    artifact_dt = pk.load(f)

models = {
    "Logistic Regression": {
        "model": artifact_lr["model_LR"],
        "feature_cols": artifact_lr["feature_cols"],
        "encoders": artifact_lr["encoders"]
    },
    "Decision Tree": {
        "model": artifact_dt["model_DT"],
        "feature_cols": artifact_dt["feature_cols"],
        "encoders": artifact_dt["encoders"]
    }
}

 
# Numeric ranges
numeric_ranges = {
    "Age": (15, 40),
    "Academic Pressure": (0, 5),
    "Work Pressure": (0, 5),
    "CGPA": (0.0, 10.0),
    "Study Satisfaction": (0, 5),
    "Job Satisfaction": (0, 5),
    "Sleep Duration": (0, 12),
    "Work/Study Hours": (0, 16),
    "Financial Stress": (0, 5),
}


# Prediction function
def predict(model_name, *user_inputs):
    model_info = models[model_name]
    model = model_info["model"]
    feature_cols = model_info["feature_cols"]
    encoders = model_info["encoders"]

    data = {}
    for col, val in zip(feature_cols, user_inputs):
        if col in encoders:
            data[col] = encoders[col][val]
        else:
            data[col] = val

    X = pd.DataFrame([data], columns=feature_cols)
    pred = model.predict(X.values)[0]
    return f"🧠 Model: {model_name}\n\n✅ Prediction: {'The student is likely experiencing depression' if pred == 1 else 'The student is not at risk of depression'}"


# Compact Gradio UI
feature_cols = artifact_lr["feature_cols"]
encoders = artifact_lr["encoders"]

with gr.Blocks(title="Depression Prediction") as demo:
    gr.Markdown("### 🧩 Student Depression Prediction ")

    model_dropdown = gr.Dropdown(
        choices=list(models.keys()),
        value="Logistic Regression",
        label="Select Model"
    )

    # 2-column layout for all features
    with gr.Row():
        with gr.Column():
            left_inputs = []
            for col in feature_cols[:len(feature_cols)//2]:
                if col in encoders:
                    left_inputs.append(gr.Dropdown(choices=list(encoders[col].keys()), label=col))
                elif col in numeric_ranges:
                    min_val, max_val = numeric_ranges[col]
                    step_val = 0.01 if col == "CGPA" else 1
                    left_inputs.append(gr.Slider(minimum=min_val, maximum=max_val, step=step_val, label=col))
                else:
                    left_inputs.append(gr.Number(label=col))

        with gr.Column():
            right_inputs = []
            for col in feature_cols[len(feature_cols)//2:]:
                if col in encoders:
                    right_inputs.append(gr.Dropdown(choices=list(encoders[col].keys()), label=col))
                elif col in numeric_ranges:
                    min_val, max_val = numeric_ranges[col]
                    step_val = 0.01 if col == "CGPA" else 1
                    right_inputs.append(gr.Slider(minimum=min_val, maximum=max_val, step=step_val, label=col))
                else:
                    right_inputs.append(gr.Number(label=col))

    with gr.Row():
        submit_btn = gr.Button("Predict", variant="primary")
        output_box = gr.Textbox(label="Result", lines=3)

    submit_btn.click(
        fn=predict,
        inputs=[model_dropdown] + left_inputs + right_inputs,
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch(share=True)
