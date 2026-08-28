import numpy as np

from src.utils.evaluation import evaluate_hierarchical_predictions


def test_trial_pooling_averages_probabilities_before_predicting():
    probabilities = np.array([
        [0.51, 0.49], [0.10, 0.90],  # s01, trial 0 -> class 1 after pooling
        [0.80, 0.20], [0.70, 0.30],  # s01, trial 1 -> class 0
        [0.20, 0.80], [0.30, 0.70],  # s02, trial 0 -> class 1
    ])
    labels = np.array([1, 1, 0, 0, 1, 1])
    subjects = np.array(["s01", "s01", "s01", "s01", "s02", "s02"])
    trials = np.array([0, 0, 1, 1, 0, 0])

    result = evaluate_hierarchical_predictions(probabilities, labels, subjects, trials)

    assert result["trial_predictions"].tolist() == [1, 0, 1]
    assert result["trial"]["accuracy"] == 1.0
    assert len(result["subject_metrics"]) == 2
    assert result["worst_subject_f1"] == 0.5
