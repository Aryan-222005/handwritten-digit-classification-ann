import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

from preprocess import load_data, preprocess_data, encode_labels


def evaluate_model():

    # 1. Load Data
    X_train, y_train, X_test, y_test = load_data()

    # 2. Preprocess
    X_train, X_test = preprocess_data(X_train, X_test)
    y_train, y_test = encode_labels(y_train, y_test)

    # 3. Load Model
    model = load_model("models/mnist_ann_v1.h5")

    # 4. Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 5. Classification Report
    print("\n Classification Report:\n")
    print(classification_report(y_true, y_pred_classes))

    # 6. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # 7. Wrong Predictions
    wrong = np.where(y_pred_classes != y_true)[0]

    plt.figure(figsize=(10, 5))
    for i in range(10):
        idx = wrong[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"P:{y_pred_classes[idx]} A:{y_true[idx]}")
        plt.axis('off')

    plt.savefig("outputs/wrong_predictions.png")
    plt.close()

    # 8. Accuracy & Loss Plot
    with open("outputs/training_history.pkl", "rb") as f:
        history = pickle.load(f)

    # Accuracy plot
    plt.figure()
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("outputs/accuracy_plot.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/loss_plot.png")
    plt.close()

    print("\n Evaluation Complete. Check outputs folder.\n")


if __name__ == "__main__":
    evaluate_model()