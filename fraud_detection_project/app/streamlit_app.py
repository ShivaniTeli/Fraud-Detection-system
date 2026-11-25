import os, requests, streamlit as st
API = os.getenv("API_URL", "http://127.0.0.1:8000")
RAW = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
st.set_page_config(page_title="Fraud Risk Console", page_icon="🛡️", layout="wide")
st.title("🛡️ Financial Fraud Detection")
st.caption("XGBoost • Risk scoring • SHAP explainability")
try:
    status = requests.get(f"{API}/health", timeout=4).json()
    st.sidebar.success("API connected")
    st.sidebar.write("Model ready:", status["model_ready"])
except requests.RequestException:
    st.sidebar.error("API unavailable")
with st.form("transaction"):
    txn = st.text_input("Transaction ID", "DEMO-TXN-001")
    st.info("The default values are illustrative only, not a real transaction.")
    vals = {}
    cols = st.columns(3)
    for i, name in enumerate(RAW):
        default = 1200.0 if name == "Time" else 149.5 if name == "Amount" else 0.0
        with cols[i % 3]:
            vals[name] = st.number_input(name, value=default, format="%.6f")
    submit = st.form_submit_button("Analyze transaction", type="primary")
if submit:
    try:
        r = requests.post(f"{API}/predict", json={"transaction_id": txn, "features": vals}, timeout=60)
        if r.status_code != 200:
            st.error(r.text)
        else:
            result = r.json()
            a, b, c = st.columns(3)
            a.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
            b.metric("Risk score", f"{result['risk_score']:.1f}/100")
            c.metric("Decision", result["decision"])
            st.subheader("SHAP explanation")
            if result["explanation"]:
                st.dataframe(result["explanation"], use_container_width=True)
            else:
                st.info("No explanation returned.")
            with st.expander("Full API response"):
                st.json(result)
    except requests.RequestException as e:
        st.error(str(e))
