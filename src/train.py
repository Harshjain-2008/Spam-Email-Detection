import pandas as pd 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.preprocess import clean_text

def train_model():

    df = pd.read_csv("data/spam.csv")

    df["message"] = df["message"].apply(clean_text)

    df["label"] = df["label"].map({
        "ham":0,
        "spam":1
    })

    X = df["message"]
    y = df["label"]

    vectorizer = TfidfVectorizer()

    x_vectorizer = vectorizer.fit_transform(X)

    X_train ,X_test ,y_train, y_test = train_test_split(x_vectorizer,y, test_size=0.2, random_state=42)

    models = {
        "Naive bayes": MultinomialNB(),
        "logistic regression" : LogisticRegression(max_iter=1000)
    }

    result = {}

    best_model = None
    best_accuracy = 0

    print("\n====MODEL COMPARISON====")

    for name, model in models.items():

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,pred)
        percision = precision_score(y_test,pred)
        f1 = f1_score(y_test, pred)
        recall = recall_score(y_test,pred)

        result[name] = accuracy

        print(f"{name}")
        print("-"*30)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Percision score : {percision:.4f}")
        print(f"F1 score : {f1:.4f}")
        print(f"Recall score : {recall:.4f}")
        print()

        if accuracy> best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("="*40)
    print(f"Best Accuracy : {best_accuracy:.4f}")
    print("="*40)

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\nBest Model saved Successfuly")

    return result    

