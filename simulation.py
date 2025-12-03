from models import HospitalState

INITIAL_STATE = HospitalState(
    ransomware_spread=18.0,
    ehr_operational=92.0,
    icu_operational=98.0,
    med_dispense_operational=90.0,
    lab_operational=95.0,
    patient_risk=22.0,
    legal_risk=15.0,
    public_trust_risk=10.0
)

def apply_decision(state: HospitalState, decision: str) -> HospitalState:
    s = HospitalState(**vars(state))

    if decision == "ISOLATE_CORE":
        s.ransomware_spread *= 0.45
        s.ehr_operational -= 18
        s.med_dispense_operational -= 15
        s.patient_risk += 12
        s.legal_risk -= 8
        s.public_trust_risk += 5

    elif decision == "MONITOR_ONLY":
        s.ransomware_spread *= 1.65
        s.ehr_operational -= 5
        s.med_dispense_operational -= 4
        s.patient_risk += 18
        s.legal_risk += 22
        s.public_trust_risk += 12

    elif decision == "FULL_LOCKDOWN":
        s.ransomware_spread *= 0.25
        s.ehr_operational -= 35
        s.med_dispense_operational -= 40
        s.lab_operational -= 30
        s.patient_risk += 38
        s.legal_risk -= 15
        s.public_trust_risk += 20

    s.patient_risk += (100 - s.icu_operational) * 0.05
    s.legal_risk += s.ransomware_spread * 0.04

    for field in vars(s):
        setattr(s, field, max(0, min(100, getattr(s, field))))

    return s
