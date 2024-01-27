import gradio as gr
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer


def predict_from_csv(file_path: str) -> pd.DataFrame:
    cv = joblib.load("models/vectorizer.joblib")
    model = joblib.load("models/classifier.joblib")
    df = pd.read_csv(file_path)
    X = df["message"]
    y = df["class_name"]
    features = cv.transform(X)
    output_df = pd.DataFrame(
        {
            "message": X,
            "class_name": y,
            "predicted_class": model.predict(features),
        }
    )
    return output_df


def predict_and_display_csv(file_path: str) -> str:
    df = predict_from_csv(file_path)
    csv_string = df.to_csv(index=False)
    return csv_string


predict_ui = gr.Interface(
    fn=predict_and_display_csv,
    inputs=gr.File(file_types=[".csv"], file_count="single"),
    outputs=gr.File(file_types=[".csv"], label="Download CSV", file_count="single"),
    title="Predict with Classification Model",
    allow_flagging="never",
)


predict_ui.launch()
