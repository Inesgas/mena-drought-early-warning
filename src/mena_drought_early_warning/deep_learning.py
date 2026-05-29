from __future__ import annotations


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def lstm_architecture_note() -> dict:
    return {
        "model_type": "LSTM classifier",
        "sequence_length": 6,
        "layers": [
            "Input(sequence_length, n_features)",
            "LSTM(64)",
            "Dropout(0.2)",
            "Dense(32, activation='relu')",
            "Dense(n_classes, activation='softmax')",
        ],
        "loss": "sparse_categorical_crossentropy",
        "optimizer": "adam",
    }
