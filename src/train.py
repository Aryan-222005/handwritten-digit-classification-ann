from preprocess import load_data, preprocess_data, encode_labels
from model import build_model
import pickle

def train_model():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Preprocess
    X_train, X_test = preprocess_data(X_train, X_test)
    y_train, y_test = encode_labels(y_train, y_test)

    # Build model
    model = build_model()

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # Save model
    model.save("models/mnist_ann_v1.h5")

    # Save training history
    with open("outputs/training_history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    print("Training complete!")

if __name__ == "__main__":
    train_model()