# predictor/base.py

from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    def __init__(self, name, n_classes):
        self.name = name
        self.n_classes = n_classes

    @abstractmethod
    def inputs(self):
        """
        Return required inputs as a set, e.g.:
        {"psd", "coh", "cont", "cat"}
        """
        pass

    @abstractmethod
    def predict_proba(self, **kwargs):
        """
        Return:
            proba: (n_epochs, n_classes)
            classes: list[str]
        """
        pass

    def hard_vote(self, proba):
        """
        epoch-level → subject-level majority vote + tie-break mean-prob
        """
        idx = np.argmax(proba, axis=1)
        vals, counts = np.unique(idx, return_counts=True)

        max_count = np.max(counts)
        candidates = vals[counts == max_count]

        if len(candidates) == 1:
            return candidates[0]

        # tie-break → mean probability
        mean = proba[:, candidates].mean(axis=0)
        return candidates[np.argmax(mean)]
