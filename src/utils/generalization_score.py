"""
generalization_score.py

Σύνθετος δείκτης "Cross-Subject Generalization Score" (G), σχεδιασμένος
ειδικά για το πρόβλημα της διπλωματικής: δεν αρκεί υψηλό μέσο Accuracy/
F1 -- μας ενδιαφέρει να γενικεύει το μοντέλο ΕΞΙΣΟΥ καλά σε ΟΛΑ τα
subjects, όχι μόνο κατά μέσο όρο.

Χρησιμοποιείται ως το Optuna objective (βλ. optuna_search.py) αντί για
απλό validation accuracy/F1, ώστε η αναζήτηση υπερπαραμέτρων να
βελτιστοποιεί πραγματικά cross-subject γενίκευση.

Ορισμός
-------
    G = w_mean * mean_subject_f1
      + w_worst * worst_subject_f1
      - w_var * subject_f1_std

- mean_subject_f1: μέσος όρος του macro-F1 ανά subject (πόσο καλά
  δουλεύει το μοντέλο "σε γενικές γραμμές").
- worst_subject_f1: το macro-F1 του ΧΕΙΡΟΤΕΡΟΥ subject (πόσο άσχημα
  μπορεί να αποτύχει το μοντέλο σε κάποιον συγκεκριμένο άνθρωπο --
  κρίσιμο για μια πραγματική cross-subject εφαρμογή).
- subject_f1_std: τυπική απόκλιση του macro-F1 ανάμεσα στα subjects
  (πόσο ασταθές είναι το μοντέλο μεταξύ διαφορετικών ανθρώπων) --
  αφαιρείται (penalty) ώστε να προτιμώνται μοντέλα με πιο ομοιόμορφη
  απόδοση σε όλα τα subjects.

Τα default βάρη (0.5 / 0.3 / 0.2) δίνουν προτεραιότητα στον μέσο όρο,
αλλά ανταμείβουν σημαντικά και το χειρότερο subject και τιμωρούν τη
διακύμανση -- ακριβώς η φιλοσοφία που ζητήθηκε ρητά για τη διπλωματική
("Mean Subject F1, Worst Subject F1 και Subject Variance").
"""

from __future__ import annotations


def compute_generalization_score(
    metrics: dict,
    w_mean: float = 0.5,
    w_worst: float = 0.3,
    w_var: float = 0.2,
) -> float:
    """
    Υπολογίζει το Cross-Subject Generalization Score από ένα metrics
    dict όπως αυτό που επιστρέφει το `run_epoch()` (και των δύο
    training scripts) ή το `evaluate_hierarchical_predictions()`.

    Parameters
    ----------
    metrics : dict
        Πρέπει να περιέχει τα κλειδιά "subject_f1_mean",
        "worst_subject_f1", "subject_f1_std".
    w_mean, w_worst, w_var : float
        Βάρη του σύνθετου δείκτη (βλ. docstring module).
    """

    return (
        w_mean * metrics["subject_f1_mean"]
        + w_worst * metrics["worst_subject_f1"]
        - w_var * metrics["subject_f1_std"]
    )
