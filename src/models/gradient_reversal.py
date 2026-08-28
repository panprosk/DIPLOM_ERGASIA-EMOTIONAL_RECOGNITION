import torch
import torch.nn as nn


class _GradientReversalFunction(torch.autograd.Function):
    """
    Αντιστρέφει (και κλιμακώνει με -lambda) το gradient κατά το
    backward pass, χωρίς καμία επίδραση στο forward pass.

    Αυτό είναι το βασικό "κόλπο" του Domain-Adversarial Training
    (Ganin & Lempitsky, DANN 2015): ό,τι "μπροστά" στο layer αυτό
    εκπαιδεύεται να ΕΛΑΧΙΣΤΟΠΟΙΕΙ ένα loss (π.χ. ταξινόμηση subject),
    ενώ ό,τι είναι "πίσω" (ο encoder) εκπαιδεύεται να ΜΕΓΙΣΤΟΠΟΙΕΙ το
    ίδιο loss (δηλαδή να μπερδεύει τον subject classifier), παράγοντας
    έτσι subject-invariant embeddings.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    nn.Module wrapper γύρω από το _GradientReversalFunction, ώστε να
    μπορεί να μπει μέσα σε ένα nn.Sequential / forward pass σαν
    οποιοδήποτε άλλο layer.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)
