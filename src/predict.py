import joblib
from src.preprocess import clean_text

model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_email(text):
    text = clean_text(text)
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        return "SPAM" 
    return"HAM" 



