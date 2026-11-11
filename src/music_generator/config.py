import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Configured path must stay inside the project: {value}")
    return resolved


def load_json(name: str) -> dict:
    path = ROOT / "configs" / name
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    data_path: Path
    artifacts_dir: Path
    logs_dir: Path
    checkpoint_dir: Path
    checkpoint_path: Path
    batch_size: int
    sequence_length: int
    embedding_dim: int
    rnn_units: int
    learning_rate: float
    iterations: int
    checkpoint_every: int
    temperature: float
    default_songs: int
    max_characters: int


def get_settings() -> Settings:
    training = load_json("training.json")
    inference = load_json("inference.json")
    return Settings(
        data_path=project_path(os.getenv("MUSIC_DATA_PATH", "data/irish.abc")),
        artifacts_dir=project_path(os.getenv("MUSIC_ARTIFACTS_DIR", "artifacts")),
        logs_dir=project_path(os.getenv("MUSIC_LOGS_DIR", "logs")),
        checkpoint_dir=project_path(os.getenv("MUSIC_CHECKPOINT_DIR", "artifacts/checkpoints")),
        checkpoint_path=project_path(
            os.getenv("MUSIC_CHECKPOINT_PATH", "artifacts/checkpoints/generator.weights.h5")
        ),
        batch_size=env_int("MUSIC_BATCH_SIZE", training["batch_size"]),
        sequence_length=env_int("MUSIC_SEQUENCE_LENGTH", training["sequence_length"]),
        embedding_dim=env_int("MUSIC_EMBEDDING_DIM", training["embedding_dim"]),
        rnn_units=env_int("MUSIC_RNN_UNITS", training["rnn_units"]),
        learning_rate=env_float("MUSIC_LEARNING_RATE", training["learning_rate"]),
        iterations=env_int("MUSIC_TRAINING_ITERATIONS", training["iterations"]),
        checkpoint_every=training["checkpoint_every"],
        temperature=env_float("MUSIC_TEMPERATURE", inference["temperature"]),
        default_songs=inference["default_songs"],
        max_characters=inference["max_characters"],
    )
