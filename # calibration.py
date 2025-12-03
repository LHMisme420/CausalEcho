

import numpy as np

class TemperatureScaler:
    """
    Post-hoc confidence calibration.
    """

    def __init__(self, temperature: float = 1.5):
        self.temperature = temperature

    def calibrate(self, prob: float) -> float:
        # Prevent division edge cases
        prob = min(max(prob, 1e-6), 1 - 1e-6)

        logit = np.log(prob / (1 - prob))
        calibrated = 1 / (1 + np.exp(-logit / self.temperature))
        return float(round(calibrated, 5))
