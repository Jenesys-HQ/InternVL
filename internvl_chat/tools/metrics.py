import logging
from typing import Any, List, Dict, Optional, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class MetricsHelper:
    def __init__(self):
        self.true_p: int = 0
        self.false_p: int = 0
        self.count: int = 0

    @property
    def accuracy(self) -> float:
        return self.true_p / self.count if self.count > 0 else 0

    def compare_true_pred(
            self,
            true: Union[List, Dict, str, int, float, bool],
            pred: Union[List, Dict, str, int, float, bool]
    ):
        if type(true) is list:
            for i, t_el in enumerate(true):
                try:
                    p_el = pred[i]
                except IndexError:
                    p_el = {}

                self.compare_true_pred(t_el, p_el)

        if type(true) is dict:
            for key, t_value in true.items():
                if pred is None:
                    p_value = None
                else:
                    p_value = pred.get(key, None)

                self.compare_true_pred(t_value, p_value)

        if true is None or true == '':
            return

        if type(true) is str or type(true) is int or type(true) is float or type(true) is bool:
            self.count += 1

            logger.debug('True: %s, Pred: %s, TP: %s', true, pred, true == pred)

            if pred == true:
                self.true_p += 1
            else:
                self.false_p += 1
