from typing import List, Dict, Any, Tuple
import numpy as np

from src.features.eeg_features import EEGFeatureExtractor
from src.features.eda_features import EDAFeatureExtractor
from src.features.ppg_features import PPGFeatureExtractor


class FeatureExtractor:
    """
    Εξάγει όλα τα handcrafted features είτε για ένα μεμονωμένο παράθυρο 
    είτε για ολόκληρο το σύνολο παραθύρων (dataset).
    """

    def __init__(self, sampling_rate: int = 128):
        self.eeg_extractor = EEGFeatureExtractor(sampling_rate=sampling_rate)
        self.eda_extractor = EDAFeatureExtractor(sampling_rate=sampling_rate)
        self.ppg_extractor = PPGFeatureExtractor(sampling_rate=sampling_rate)

    def extract_components(
        self,
        eeg: np.ndarray,
        eda: np.ndarray,
        ppg: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Υπολογίζει ξεχωριστά τα features ανά modality
        (χωρίς να τα συνενώνει).

        Χρήσιμο ώστε ο MLP Encoder του Hybrid CNN-MLP να
        χρησιμοποιεί αποκλειστικά τα physiological (EDA+PPG)
        features, ενώ τα EEG handcrafted features παραμένουν
        διαθέσιμα (π.χ. για ablation studies ή visualization).
        """

        eeg_feat = self.eeg_extractor.extract(eeg)
        eda_feat = self.eda_extractor.extract(eda)
        ppg_feat = self.ppg_extractor.extract(ppg)

        # Ασφάλεια: μερικά windows (π.χ. flat/κορεσμένο σήμα σε κάποιο
        # subject) μπορεί να παράγουν NaN/Inf σε skew/kurtosis ή σε
        # NeuroKit2 features. Ένα και μόνο NaN σε ΜΙΑ στήλη μολύνει το
        # mean/std του StandardScaler και άρα ΟΛΟ το dataset (όλα τα
        # samples γίνονται NaN μετά το scaling). Το καθαρίζουμε εδώ,
        # στην πηγή, πριν καν φτάσει στον scaler.
        eeg_feat = np.nan_to_num(eeg_feat, nan=0.0, posinf=0.0, neginf=0.0)
        eda_feat = np.nan_to_num(eda_feat, nan=0.0, posinf=0.0, neginf=0.0)
        ppg_feat = np.nan_to_num(ppg_feat, nan=0.0, posinf=0.0, neginf=0.0)

        return eeg_feat, eda_feat, ppg_feat

    def extract(self, eeg: np.ndarray, eda: np.ndarray, ppg: np.ndarray) -> np.ndarray:
        """
        Υπολογίζει και συνενώνει τα χαρακτηριστικά από όλα τα modalities για 1 παράθυρο.

        Parameters
        ----------
        eeg : np.ndarray
            Τα δεδομένα EEG του παράθυρου (32, window_samples).
        eda : np.ndarray
            Τα δεδομένα EDA του παράθυρου (1, window_samples).
        ppg : np.ndarray
            Τα δεδομένα PPG του παράθυρου (1, window_samples).

        Returns
        -------
        np.ndarray
            Ένα μονοδιάστατο array float32 με όλα τα συνενωμένα features.
        """
        eeg_feat, eda_feat, ppg_feat = self.extract_components(eeg, eda, ppg)

        features = np.concatenate([
            eeg_feat,
            eda_feat,
            ppg_feat
        ])

        # Ασφάλεια: μερικά windows (π.χ. flat/κορεσμένο σήμα σε κάποιο
        # subject) μπορεί να παράγουν NaN/Inf σε skew/kurtosis ή σε
        # NeuroKit2 features. Ένα και μόνο NaN σε ΜΙΑ στήλη μολύνει το
        # mean/std του StandardScaler και άρα ΟΛΟ το dataset (όλα τα
        # samples γίνονται NaN μετά το scaling). Το καθαρίζουμε εδώ,
        # στην πηγή, πριν καν φτάσει στον scaler.
        features = np.nan_to_num(
            features, nan=0.0, posinf=0.0, neginf=0.0
        )

        return features.astype(np.float32)

    def process_dataset(self, windows_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Διατρέχει όλα τα windows ενός dataset (train/val/test) και προσθέτει
        σε κάθε sample τα κλειδιά:

        - 'features'         : EEG + EDA + PPG handcrafted features (πλήρες vector)
        - 'physio_features'  : μόνο EDA + PPG handcrafted features
                                (είσοδος του MLP Encoder του Hybrid CNN-MLP)

        Parameters
        ----------
        windows_list : List[Dict[str, Any]]
            Η λίστα με τα παράθυρα που επιστρέφει ο WindowSegmenter.

        Returns
        -------
        List[Dict[str, Any]]
            Η ίδια λίστα παραθύρων εμπλουτισμένη με τα features.
        """
        for sample in windows_list:

            eeg_feat, eda_feat, ppg_feat = self.extract_components(
                eeg=sample["eeg"],
                eda=sample["eda"],
                ppg=sample["ppg"],
            )

            sample["features"] = np.concatenate(
                [eeg_feat, eda_feat, ppg_feat]
            ).astype(np.float32)

            sample["physio_features"] = np.concatenate(
                [eda_feat, ppg_feat]
            ).astype(np.float32)

        return windows_list