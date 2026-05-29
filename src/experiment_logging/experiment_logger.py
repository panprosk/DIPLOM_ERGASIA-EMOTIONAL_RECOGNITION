"""
experiment_logger.py
Βήμα 16: Αποθήκευση αποτελεσμάτων σε CSV και SQLite.
"""

import os
import csv
import json
import sqlite3
import pandas as pd
from datetime import datetime


EXPERIMENTS_DIR = "experiments"
CSV_PATH        = os.path.join(EXPERIMENTS_DIR, "experiments.csv")
DB_PATH         = os.path.join(EXPERIMENTS_DIR, "experiments.db")


class ExperimentLogger:

    def __init__(self):
        os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Δημιουργεί τον πίνακα αν δεν υπάρχει."""
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id        TEXT PRIMARY KEY,
                timestamp            TEXT,
                model                TEXT,
                signals              TEXT,
                accuracy             REAL,
                f1                   REAL,
                worst_f1             REAL,
                variance             REAL,
                confidence           REAL,
                entropy              REAL,
                generalization_score REAL,
                best_val_f1          REAL,
                training_time        REAL,
                config_json          TEXT,
                subject_metrics_json TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log(self, results: dict):
        """Αποθηκεύει ένα experiment στο CSV και στο DB."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "experiment_id":        results["experiment_id"],
            "timestamp":            timestamp,
            "model":                results["model"],
            "signals":              str(results["signals"]),
            "accuracy":             results["accuracy"],
            "f1":                   results["f1"],
            "worst_f1":             results["worst_f1"],
            "variance":             results["variance"],
            "confidence":           results["confidence"],
            "entropy":              results["entropy"],
            "generalization_score": results["generalization_score"],
            "best_val_f1":          results["best_val_f1"],
            "training_time":        results["training_time"],
            "config_json":          json.dumps(results["config"]),
            "subject_metrics_json": json.dumps(results.get("subject_metrics", {})),
        }

        # CSV
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # SQLite
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO experiments VALUES
            (:experiment_id, :timestamp, :model, :signals,
             :accuracy, :f1, :worst_f1, :variance,
             :confidence, :entropy, :generalization_score,
             :best_val_f1, :training_time,
             :config_json, :subject_metrics_json)
        """, row)
        conn.commit()
        conn.close()

        print(f"  ✓ Logged → experiments.csv & experiments.db")

    def get_all(self) -> pd.DataFrame:
        """Επιστρέφει όλα τα experiments ως DataFrame."""
        if not os.path.exists(DB_PATH):
            return pd.DataFrame()
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql("SELECT * FROM experiments ORDER BY timestamp", conn)
        conn.close()
        return df

    def get_best(self, metric: str = "generalization_score") -> dict:
        """Επιστρέφει το best experiment βάσει metric."""
        df = self.get_all()
        if df.empty:
            return {}
        best_row = df.loc[df[metric].idxmax()]
        return best_row.to_dict()
