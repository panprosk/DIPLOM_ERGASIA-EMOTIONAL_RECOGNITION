class SubjectDataset:

    """
    Οργανώνει το dataset
    ανά subject.
    """

    def __init__(self, dataset):

        self.dataset = dataset

    def subjects(self):

        return list(self.dataset.keys())

    def get_subject(self, subject_id):

        return self.dataset[subject_id]

    def number_of_subjects(self):

        return len(self.dataset)

    def trials(self, subject_id):

        return self.dataset[subject_id]["data"].shape[0]