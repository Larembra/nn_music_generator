import numpy as np
import tensorflow as tf


def get_batch(vectorized_text: np.ndarray, sequence_length: int, batch_size: int):
    available = vectorized_text.size - sequence_length - 1
    if available <= 0:
        raise ValueError("Dataset is shorter than the requested sequence length")
    starts = np.random.randint(0, available, size=batch_size)
    inputs = np.stack([vectorized_text[i : i + sequence_length] for i in starts])
    targets = np.stack([vectorized_text[i + 1 : i + sequence_length + 1] for i in starts])
    return inputs, targets


def build_model(vocab_size: int, embedding_dim: int, rnn_units: int, batch_size: int):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(batch_shape=(batch_size, None), dtype=tf.int32),
            tf.keras.layers.Embedding(vocab_size, embedding_dim),
            tf.keras.layers.LSTM(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            ),
            tf.keras.layers.Dense(vocab_size),
        ]
    )


def reset_states(model) -> None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            layer.reset_states()
