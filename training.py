import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import joblib


def train_classification_model(file_path: str):
    df = pd.read_csv(file_path)
    X = df["message"]
    y = df["class_name"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    cv = CountVectorizer()
    features = cv.fit_transform(X_train)
    model = SVC(probability=True)
    model.fit(features, y_train)
    joblib.dump(model, "models/classifier.joblib")
    joblib.dump(cv, "models/vectorizer.joblib")
    return f"Model trained with {len(X_train)} examples. Accuracy: \
            {accuracy_score(y_train, model.predict(features))}"


train_ui = gr.Interface(
    fn=train_classification_model,
    inputs=gr.File(file_types=[".csv"], file_count="single"),
    outputs="text",
    title="Train Classification Model",
    allow_flagging="never",
)


train_ui.launch()
