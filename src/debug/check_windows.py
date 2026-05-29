from data.deap_loader import load_all_subjects
from data.preprocessing import preprocess_subject
from data.dataset import segment_trials

DATA_DIR = "data/deap"

subjects = load_all_subjects(DATA_DIR, subject_ids=[1])

subject = preprocess_subject(subjects[1])

print("\n=== WINDOW CHECK ===")

eeg_w, eda_w, ppg_w, y = segment_trials(subject)

print("EEG windows:", eeg_w.shape)
print("EDA windows:", eda_w.shape)
print("PPG windows:", ppg_w.shape)
print("Labels:", y.shape)