import logging
from typing import Any, List, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class MetricsHelper:
    def __init__(self):
        self.true_p = 0
        self.false_p = 0
        self.count = 0

    @property
    def accuracy(self):
        return self.true_p / self.count if self.count > 0 else 0

    def compare_true_pred(self, true, pred):
        if type(true) is list:
            for t, p in zip(true, pred):
                self.compare_true_pred(t, p)

        if type(true) is dict:
            for key in true:
                if key not in pred:
                    logger.warning(f"key {key} not found in predictions")
                    continue

                self.compare_true_pred(true[key], pred[key])

        if type(true) is str or type(true) is int or type(true) is float or type(true) is bool:
            if true is None:
                return

            self.count += 1

            logger.debug('True: %s, Pred: %s, TP: %s', true, pred, true == pred)

            if pred == true:
                self.true_p += 1
            else:
                self.false_p += 1
