from pathlib import Path


def extract_song_snippets(text: str) -> list[str]:
    songs = []
    current = []
    for line in text.splitlines():
        if line.startswith("X:") and current:
            songs.append("\n".join(current).strip())
            current = []
        if line.strip():
            current.append(line)
    if current:
        songs.append("\n".join(current).strip())
    return songs


def load_training_data(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"ABC dataset not found: {path}")
    songs = extract_song_snippets(path.read_text(encoding="utf-8"))
    if not songs:
        raise ValueError(f"No ABC songs found in {path}")
    return songs


def save_song(song: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(song.rstrip() + "\n", encoding="utf-8")
