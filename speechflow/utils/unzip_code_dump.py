import pickle

from pathlib import Path

if __name__ == "__main__":
    dump_scripts_path = Path("scripts.pkl")
    out_path = dump_scripts_path.parent / "code"
    out_path.mkdir(exist_ok=True)

    code = pickle.loads(dump_scripts_path.read_bytes())
    for key, text in code.items():
        Path(out_path.as_posix() + key).parents[0].mkdir(parents=True, exist_ok=True)
        Path(out_path.as_posix() + key).write_bytes(text)
