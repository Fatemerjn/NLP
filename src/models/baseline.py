from __future__ import annotations
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def make_baseline_pipeline(ngram_range=(1,2), max_features=40000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None))
    ])

def train_eval_baseline(train_texts, train_labels, test_texts, test_labels) -> Tuple[Pipeline, dict]:
    pipe = make_baseline_pipeline()
    pipe.fit(train_texts, train_labels)
    preds = pipe.predict(test_texts)
    acc = accuracy_score(test_labels, preds)
    report = classification_report(test_labels, preds, digits=4, output_dict=True)
    return pipe, {"accuracy": acc, "report": report}