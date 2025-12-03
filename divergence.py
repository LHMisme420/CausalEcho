from models import HospitalState

def compute_cdi(predicted: HospitalState, actual: HospitalState) -> float:
    total = 0
    keys = vars(predicted).keys()

    for k in keys:
        total += abs(getattr(predicted, k) - getattr(actual, k))

    return round(total / (len(keys) * 100), 4)
