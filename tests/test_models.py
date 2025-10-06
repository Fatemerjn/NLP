from src.models.baseline import train_eval_baseline

def test_baseline_tiny():
    X = ["a hero saves world", "a couple in love", "spaceship battle", "family drama"]
    y = ["action", "romance", "action", "drama"]
    _, metrics = train_eval_baseline(X, y, X, y)
    assert "accuracy" in metrics