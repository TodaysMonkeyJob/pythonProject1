"""
Runtime gas inference: curve → gas type, confidence and concentration. No plotting imports.
"""
import math
from typing import Any, Dict, Sequence

import joblib
import numpy as np
import sklearn

from system_setup.logger_setup import init_logger

logger = init_logger()

DEFAULT_CLASSIFIER_PATH = "models/gas_classifier.joblib"
DEFAULT_REGRESSOR_PATH = "models/gas_concentration.joblib"
CLEAN_AIR = "clean_air"


class GasPredictor:
    """
    Loads the artifacts written by modules.gas.train.
    """

    def __init__(self, classifier_path: str = DEFAULT_CLASSIFIER_PATH,
                 regressor_path: str = DEFAULT_REGRESSOR_PATH):
        """
        :param classifier_path: models/gas_classifier.joblib
        :param regressor_path: models/gas_concentration.joblib
        """
        classifier = joblib.load(classifier_path)
        regressor = joblib.load(regressor_path)
        for name, artifact in (("classifier", classifier), ("regressor", regressor)):
            if artifact["sklearn_version"] != sklearn.__version__:
                logger.warning(f"Gas {name} was trained with scikit-learn {artifact['sklearn_version']}, "
                               f"running {sklearn.__version__}; retrain if predictions look wrong")
        self.model = classifier["model"]
        self.class_names = list(classifier["class_names"])
        self.voltages = list(classifier["voltages"])
        self.regressors = regressor["models"]
        self.n_features = len(self.voltages)

    def validate(self, curve: Sequence[float]) -> np.ndarray:
        """
        :param curve: Resistance values at self.voltages
        :return: Curve as a (1, n_features) array
        """
        if len(curve) != self.n_features:
            raise ValueError(f"Expected {self.n_features} values (Ug {self.voltages[0]}..{self.voltages[-1]} V), "
                             f"got {len(curve)}")
        values = [float(v) for v in curve]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Curve contains NaN or infinite values")
        return np.array(values, dtype=float).reshape(1, -1)

    def predict(self, curve: Sequence[float]) -> Dict[str, Any]:
        """
        :param curve: 16 resistance values for Ug = 0.0..1.5 V
        :return: {"gas": str, "gas_confidence": float, "concentration_pct": float}
        """
        X = self.validate(curve)
        # Label from predict() so runtime matches the reported test metrics (SVC's Platt-scaled
        # probabilities can disagree with its decision function); confidence from predict_proba()
        label = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        confidence = proba[list(self.model.classes_).index(label)]
        gas = self.class_names[label]
        if gas == CLEAN_AIR:
            concentration = 0.0
        else:
            concentration = max(0.0, float(self.regressors[gas].predict(X)[0]))
        return {"gas": gas, "gas_confidence": round(float(confidence), 4),
                "concentration_pct": round(concentration, 3)}
