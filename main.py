from src.train import train_model
from src.predict import predict_email

def main():

    while True:

        print("\n ==== SPAM EMAIL DETECTION ====")

        print("1. Train Model ")
        print("2. Predict Email")
        print("3. Exit")

        choice = input("\n ENTER YOUR CHOICE: ")

        if choice== "1":

            train_model()

        elif choice == "2":

            email = input("\nENTER email\n")
            result = predict_email(email)
            print("\n Prediction :", result)

        elif choice=="3":
            print("Exit Model..")
            break

        else:
            return "INVALID INPUT" 

if __name__ == "__main__":
    main()       