import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(X_test,y_test,model):

    cm = confusion_matrix(y_test, model.predict(X_test))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix")
    plt.show()

    