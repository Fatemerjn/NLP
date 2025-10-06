from __future__ import annotations
import os, numpy as np, torch
from typing import Dict
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import evaluate

def _device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_eval_bert(
    train_texts, train_labels,
    test_texts,  test_labels,
    model_name="bert-base-multilingual-cased",
    epochs=3, lr=2e-5, batch_size=8, seed=42
) -> Dict:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = _device()

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test  = le.transform(test_labels)

    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_ds = Dataset.from_dict({"text": train_texts, "label": y_train}).map(tokenize, batched=True)
    test_ds  = Dataset.from_dict({"text": test_texts,  "label": y_test}).map(tokenize, batched=True)

    # Important bits for Apple Silicon:
    # - device_map="auto" lets Transformers place the model on MPS
    # - torch_dtype=torch.float32 avoids half-precision issues on MPS
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(le.classes_),
        torch_dtype=torch.float32,
        device_map="auto"
    ).to(device)

    acc = evaluate.load("accuracy")
    f1  = evaluate.load("f1")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": acc.compute(predictions=preds, references=p.label_ids)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
        }

    args = TrainingArguments(
        output_dir="checkpoints/bert",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        fp16=False,           # keep fp32 on MPS
        bf16=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    return {"label_encoder": le, "tokenizer": tok, "metrics": eval_metrics, "model": model}