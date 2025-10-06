from src.data import load_dataset
import pandas as pd

def test_load_dataset(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text("text,label\nhello world,other\nsome plot,action\n", encoding="utf-8")
    tr, te = load_dataset(str(p), lang="en", test_size=0.5, seed=1)
    assert {"text","label"} <= set(tr.columns)
    assert not tr.empty and not te.empty