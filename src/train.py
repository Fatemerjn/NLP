from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.data import load_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to dataset CSV with text,label columns")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split fraction (0<test_size<1)")
    p.add_argument("--lang", default="fa", choices=["fa","en"])
    p.add_argument("--model", default="baseline", choices=["baseline","bert"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_dataset(args.csv, lang=args.lang, test_size=args.test_size)
    train_texts, train_labels = train_df["text"].tolist(), train_df["label"].tolist()
    test_texts,  test_labels  = test_df["text"].tolist(),  test_df["label"].tolist()

    if args.model == "baseline":
        from src.models.baseline import train_eval_baseline
        model, metrics = train_eval_baseline(train_texts, train_labels, test_texts, test_labels)
        (out_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("Baseline metrics:", metrics["accuracy"])
    else:
        from src.models.bert import train_eval_bert
        res = train_eval_bert(train_texts, train_labels, test_texts, test_labels,
                              epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        (out_dir / "bert_metrics.json").write_text(json.dumps(res["metrics"], indent=2), encoding="utf-8")
        print("BERT metrics:", res["metrics"])

if __name__ == "__main__":
    main()