from models import HospitalState

def compute_ethics_score(state: HospitalState) -> float:
    w_patient = 0.45
    w_legal = 0.30
    w_trust = 0.25

    harm = (
        w_patient * state.patient_risk +
        w_legal * state.legal_risk +
        w_trust * state.public_trust_risk
    )

    ethics = 1.0 - (harm / 100.0)
    return round(max(0.0, min(1.0, ethics)), 3)
