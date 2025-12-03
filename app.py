import streamlit as st
from datetime import datetime
from models import *
from simulation import INITIAL_STATE, apply_decision
from ethics import compute_ethics_score
from divergence import compute_cdi
from forensics import seal_forensic_record

st.set_page_config(layout="wide")
st.title("🛡️ CausalEcho — Hospital Cyber Defense Command")

if "state" not in st.session_state:
    st.session_state.state = INITIAL_STATE
    st.session_state.prev_hash = ""

col1, col2 = st.columns(2)

with col1:
    st.subheader("System Telemetry")
    st.metric("Ransomware Spread", st.session_state.state.ransomware_spread)
    st.metric("EHR Operational", st.session_state.state.ehr_operational)
    st.metric("ICU Operational", st.session_state.state.icu_operational)
    st.metric("Patient Risk", st.session_state.state.patient_risk)

with col2:
    st.subheader("Commander Decision")
    option = st.radio("Select Action:", ["ISOLATE_CORE", "MONITOR_ONLY", "FULL_LOCKDOWN"])
    rationale = st.text_area("Decision Rationale")
    stress = st.slider("Stress Level", 0, 100, 50)
    execute = st.button("EXECUTE")

if execute:
    predicted_state = apply_decision(st.session_state.state, option)
    actual_state = apply_decision(st.session_state.state, option)

    ethics_pred = compute_ethics_score(predicted_state)
    ethics_actual = compute_ethics_score(actual_state)
    cdi = compute_cdi(predicted_state, actual_state)

    decision = DecisionRecord("Cmd_01", datetime.utcnow(), 1, option, rationale, stress)
    prediction = PredictionRecord(vars(predicted_state), ethics_pred)
    outcome = OutcomeRecord(vars(actual_state))
    ethics = EthicsRecord(ethics_actual, ethics_actual >= 0.6, "Cmd_01")
    divergence = DivergenceRecord(cdi)

    forensic_data = {
        "decision": vars(decision),
        "prediction": prediction.predicted_outcomes,
        "outcome": outcome.actual_outcomes,
        "ethics": vars(ethics),
        "divergence": vars(divergence)
    }

    record_hash = seal_forensic_record(forensic_data, st.session_state.prev_hash)
    st.session_state.prev_hash = record_hash
    st.session_state.state = actual_state

    st.success("Decision Executed & Forensic Record Sealed")
    st.code(record_hash)
