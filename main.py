import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
def train_linear_regression_model(file_path: str) -> str:
    df = pd.read_csv(file_path)
    X = df[['X']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    model_filename = "linear_regression_model.joblib"
    joblib.dump(model, model_filename)
    return f"Linear Regression Model trained and saved. Mean Squared Error: {mse:.4f}"
def predict_from_csv(file_path: str) -> pd.DataFrame:
    model = joblib.load("linear_regression_model.joblib")
    df = pd.read_csv(file_path)
    X_predict = df[['X']]
    predictions = model.predict(X_predict)
    df['Predicted_y'] = predictions
    return df
train_ui = gr.Interface(
    fn=train_linear_regression_model,
    inputs=gr.File(file_types=[".csv"], file_count='single'),
    outputs="text",
    title="Train Linear Regression Model",
    allow_flagging='never',
)
predict_ui = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(file_types=[".csv"], file_count='single'),
    outputs="table",
    title="Predict with Linear Regression Model",
    allow_flagging='never',
)
if _name_ == "_main_":
    # Launch the Gradio Interfaces
    train_ui.launch()
    predict_ui.launch()