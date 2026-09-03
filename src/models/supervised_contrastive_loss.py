import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon), Khosla et al. (2020),
    "Supervised Contrastive Learning".

    Κίνητρο (cross-subject generalization): σε αντίθεση με το
    Domain-Adversarial training (που αναγκάζει τα embeddings να ΜΗΝ
    περιέχουν καθόλου πληροφορία για την ταυτότητα του subject -- κάτι
    που αν εφαρμοστεί πολύ έντονα μπορεί να αφαιρέσει και χρήσιμη
    πληροφορία για το ίδιο το task, όπως παρατηρήθηκε πειραματικά εδώ),
    το Supervised Contrastive Learning δουλεύει θετικά: τραβάει κοντά
    embeddings δειγμάτων με το ΙΔΙΟ emotion label (ανεξαρτήτως από ποιο
    subject προέρχονται) και απωθεί embeddings με διαφορετικό label.
    Αυτό ενθαρρύνει άμεσα label-discriminative, subject-agnostic
    αναπαραστάσεις, χωρίς να τιμωρεί ρητά την υπόλοιπη subject-specific
    πληροφορία.

    Ιδέα από πρόσφατη cross-subject EEG emotion recognition βιβλιογραφία:
    "CA-DASCLNet: A deep learning framework for cross-subject EEG
    emotion recognition via joint domain-adversarial and contrastive
    learning" — Su et al. (2026), όπου το Supervised Contrastive
    Learning συνδυάζεται με Channel Attention και Domain-Adversarial
    Learning για βελτίωση της διαχωρισιμότητας emotion representations
    σε LOSO cross-subject πειράματα (DEAP κ.ά.).

    Υπολογίζεται πάνω σε ένα ξεχωριστό, L2-normalized projection του
    fused embedding (όχι απευθείας πάνω στο embedding που τροφοδοτεί
    τον classifier), ακολουθώντας την πρακτική SimCLR/SupCon όπου ο
    "projection head" αποσυνδέεται από το downstream task.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : torch.Tensor, shape (batch, proj_dim), ήδη L2-normalized.
        labels     : torch.Tensor, shape (batch,), integer class labels.

        Returns
        -------
        torch.Tensor, scalar loss. Επιστρέφει 0.0 αν το batch δεν έχει
        κανένα θετικό ζευγάρι (π.χ. batch size 1 ή όλα διαφορετικές
        κλάσεις μοναχικά).
        """

        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        labels = labels.view(-1, 1)

        # Μάσκα θετικών ζευγαριών (ίδιο label), εξαιρώντας τη διαγώνιο
        # (ένα δείγμα δεν είναι θετικό ζευγάρι με τον εαυτό του εδώ).
        positive_mask = torch.eq(labels, labels.T).float().to(device)

        self_mask = torch.eye(batch_size, device=device)

        positive_mask = positive_mask - self_mask

        # Αν κανένα δείγμα δεν έχει θετικό ζευγάρι στο batch, δεν έχει
        # νόημα ο υπολογισμός (θα δώσει NaN από διαίρεση με μηδέν).
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Numerical stability: αφαίρεση του max ανά γραμμή πριν το exp.
        similarity_max, _ = similarity.max(dim=1, keepdim=True)
        similarity = similarity - similarity_max.detach()

        exp_similarity = torch.exp(similarity) * (1.0 - self_mask)

        log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True) + 1e-12)

        # Μέσος log-likelihood πάνω στα θετικά ζευγάρια κάθε δείγματος.
        mean_log_prob_positive = (positive_mask * log_prob).sum(dim=1) / (
            positive_mask.sum(dim=1) + 1e-12
        )

        # Δείγματα χωρίς κανένα θετικό ζευγάρι δεν συνεισφέρουν στο loss.
        has_positive = positive_mask.sum(dim=1) > 0

        loss = -mean_log_prob_positive[has_positive].mean()

        return loss


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Βοηθητική συνάρτηση: L2-normalization, απαραίτητη πριν το SupCon."""
    return F.normalize(x, p=2, dim=dim, eps=eps)
