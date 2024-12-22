import joblib
import gradio as gr
import numpy as np

# Tải mô hình từ file
random_forest = joblib.load("titanic.pkl")

# Define the prediction function
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    # Map inputs to match the model's feature set
    sex_male = 1 if Sex == "Male" else 0
    embarked_Q = 1 if Embarked == "Q" else 0
    embarked_S = 1 if Embarked == "S" else 0

    # Create a feature array
    features = np.array([[Pclass, Age, SibSp, Parch, Fare, sex_male, embarked_Q, embarked_S]])
    
    # Predict survival
    prediction = random_forest.predict(features)
    probability = random_forest.predict_proba(features)[0][1]
    
    return "Survived" if prediction[0] == 1 else "Did not survive", round(probability * 100, 2)

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown([1, 2, 3], label="Passenger Class (Pclass)"),
        gr.Radio(["Male", "Female"], label="Sex"),
        gr.Number(label="Age", value=30, interactive=True),
        gr.Slider(0, 10, step=1, label="Number of Siblings/Spouses Aboard (SibSp)"),
        gr.Slider(0, 10, step=1, label="Number of Parents/Children Aboard (Parch)"),
        gr.Number(label="Fare", value=30.0, interactive=True),
        gr.Dropdown(["C", "Q", "S"], label="Port of Embarkation (Embarked)")
    ],
    outputs=[
        gr.Text(label="Prediction"),
        gr.Text(label="Probability (%)")
    ],
    title="Titanic Survival Prediction",
    description="Enter the details of a Titanic passenger to predict if they survived."
)

# Launch the Gradio app
interface.launch()