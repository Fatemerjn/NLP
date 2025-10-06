from src.data import load_dataset
from pathlib import Path

def test_load_dataset(tmp_path: Path):
    p = tmp_path / "tiny.csv"
    p.write_text("text,label\nhi,one\nhello,one\nbye,two\nciao,two\n", encoding="utf-8")
    tr, te = load_dataset(str(p), lang="en", test_size=0.5, seed=1)
    assert not tr.empty and not te.empty