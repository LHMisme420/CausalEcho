from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

@dataclass
class HospitalState:
    ransomware_spread: float
    ehr_operational: float
    icu_operational: float
    med_dispense_operational: float
    lab_operational: float
    patient_risk: float
    legal_risk: float
    public_trust_risk: float


@dataclass
class DecisionRecord:
    actor_id: str
    timestamp: datetime
    round_id: int
    option_selected: str
    rationale: str
    stress_level: int


@dataclass
class PredictionRecord:
    predicted_outcomes: Dict[str, float]
    ethics_score_pred: float


@dataclass
class OutcomeRecord:
    actual_outcomes: Dict[str, float]


@dataclass
class EthicsRecord:
    ethics_score_actual: float
    decision_justified: bool
    primary_responsible_actor: str


@dataclass
class DivergenceRecord:
    cdi: float


@dataclass
class ForensicRecord:
    decision: DecisionRecord
    prediction: PredictionRecord
    outcome: OutcomeRecord
    ethics: EthicsRecord
    divergence: DivergenceRecord
    record_hash: str
    previous_hash: Optional[str] = None
