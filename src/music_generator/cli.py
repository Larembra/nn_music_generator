import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
import typer

from music_generator.config import get_settings
from music_generator.data import extract_song_snippets, load_training_data, save_song
from music_generator.model import build_model, get_batch, reset_states

app = typer.Typer(no_args_is_help=True, help="Train and run the ABC music generator.")
logger = logging.getLogger("music_generator")


def configure_logging(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / "music-generator.log", encoding="utf-8"),
        ],
        force=True,
    )


def model_vocabulary(songs: list[str]) -> tuple[str, dict[str, int], np.ndarray]:
    joined = "\n\n".join(songs)
    vocabulary = "".join(sorted(set(joined)))
    char_to_index = {char: index for index, char in enumerate(vocabulary)}
    encoded = np.fromiter((char_to_index[char] for char in joined), dtype=np.int32)
    return vocabulary, char_to_index, encoded


@app.command("check-data")
def check_data() -> None:
    settings = get_settings()
    configure_logging(settings.logs_dir)
    songs = load_training_data(settings.data_path)
    typer.echo(f"Dataset: {settings.data_path}\nSongs: {len(songs)}")


@app.command()
def train(
    iterations: int | None = typer.Option(None, min=1),
    resume: bool = typer.Option(True, "--resume/--fresh"),
) -> None:
    settings = get_settings()
    configure_logging(settings.logs_dir)
    if settings.batch_size < 1 or settings.sequence_length < 1:
        raise typer.BadParameter("Batch size and sequence length must be positive")
    songs = load_training_data(settings.data_path)
    vocabulary, _, encoded = model_vocabulary(songs)
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = settings.checkpoint_path
    model = build_model(
        len(vocabulary), settings.embedding_dim, settings.rnn_units, settings.batch_size
    )
    if resume and checkpoint.is_file():
        model.load_weights(checkpoint)
        logger.info("Loaded checkpoint %s", checkpoint)
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings.learning_rate)

    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    targets, logits, from_logits=True
                )
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    total = iterations or settings.iterations
    logger.info("Training for %d iterations with %d songs", total, len(songs))
    for step in range(1, total + 1):
        inputs, targets = get_batch(encoded, settings.sequence_length, settings.batch_size)
        loss = train_step(inputs, targets)
        if step == 1 or step % 2 == 0:
            logger.info("step=%d loss=%.4f", step, float(loss.numpy()))
        if step % settings.checkpoint_every == 0:
            model.save_weights(checkpoint)
    model.save_weights(checkpoint)
    logger.info("Saved checkpoint to %s", checkpoint)


def sample_next(logits: tf.Tensor, temperature: float) -> int:
    if temperature <= 0:
        raise typer.BadParameter("Temperature must be greater than zero")
    return int(tf.random.categorical(logits / temperature, num_samples=1)[0, 0].numpy())


@app.command()
def generate(
    songs: int | None = typer.Option(None, min=1),
    output: Path = typer.Option(Path("artifacts/generated")),
    characters: int = typer.Option(6000, min=1),
    temperature: float | None = typer.Option(None, min=0.01),
) -> None:
    settings = get_settings()
    configure_logging(settings.logs_dir)
    if characters > settings.max_characters:
        raise typer.BadParameter(f"Maximum allowed generation length is {settings.max_characters}")
    corpus = load_training_data(settings.data_path)
    vocabulary, char_to_index, _ = model_vocabulary(corpus)
    checkpoint = settings.checkpoint_path
    if not checkpoint.is_file():
        raise typer.BadParameter("Checkpoint not found; run `music-generator train` first")
    model = build_model(len(vocabulary), settings.embedding_dim, settings.rnn_units, 1)
    model.load_weights(checkpoint)
    output_path = output.expanduser()
    output_dir = (Path.cwd() / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()
    if output_dir == Path(output_dir.anchor):
        raise typer.BadParameter("Output directory cannot be a filesystem root")
    if output_dir == Path.home() or Path.home() in output_dir.parents:
        raise typer.BadParameter("Output directory cannot be inside the user profile")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "X:"
    input_eval = tf.constant([[char_to_index[char] for char in prefix]], dtype=tf.int32)
    reset_states(model)
    generated: list[str] = []
    heat = temperature or settings.temperature
    for _ in range(characters):
        logits = tf.squeeze(model(input_eval, training=False), axis=0)[-1:]
        next_index = sample_next(logits, heat)
        generated.append(vocabulary[next_index])
        input_eval = tf.constant([[next_index]], dtype=tf.int32)
    text = prefix + "".join(generated)
    found = extract_song_snippets(text)
    target_count = songs or settings.default_songs
    selected = found[:target_count] or [text]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for index, song in enumerate(selected, start=1):
        destination = output_dir / f"song_{stamp}_{index:02d}.abc"
        save_song(song, destination)
        logger.info("Saved generated song to %s", destination)
    typer.echo(f"Generated {len(selected)} ABC file(s) in {output_dir}")


if __name__ == "__main__":
    app()
