from typing import Dict, Any, List
from .window_segmenter import WindowSegmenter


class SegmentationPipeline:
    """
    Segments every subject independently and flattens all generated windows.
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        window_seconds: float = 4.0,
        overlap: float = 0.5
    ):
        self.segmenter = WindowSegmenter(
            sampling_rate=sampling_rate,
            window_seconds=window_seconds,
            overlap=overlap
        )

    def process_dataset(self, subjects: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Διατρέχει όλα τα υποκείμενα του dataset (π.χ. train_subjects) 
        και επιστρέφει μια ενιαία λίστα από όλα τα παρατηρούμενα παράθυρα (windows).
        """
        all_windows = []

        for subject_name, subject_data in subjects.items():
            subject_windows = self.segmenter.segment_subject(subject_data)
            all_windows.extend(subject_windows)

        return all_windows