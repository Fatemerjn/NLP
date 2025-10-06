from src.models.baseline import train_eval_baseline

def test_baseline_tiny():
    X = ["a hero saves world", "a couple falls in love", "a spaceship battle", "a family drama"]
    y = ["action", "romance", "action", "drama"]
    model, metrics = train_eval_baseline(X, y, X, y)
    assert "accuracy" in metrics and 0 <= metrics["accuracy"] <= 1