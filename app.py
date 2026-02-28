import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/rf_best_mode_core.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def main():
    st.set_page_config(page_title="Smart Adaptive Suspension", layout="centered")
    st.title("Smart Adaptive Suspension (Local Demo)")
    st.write("Enter road & speed values → get predicted suspension mode + confidence.")

    model = load_model()

    road_event = st.selectbox(
        "road_event",
        ["smooth", "pothole_moderate", "pothole_deep", "rough", "moderate", "unknown"],
        index=0
    )

    depth_cm = st.slider("Pothole depth (cm)", 0.0, 15.0, 3.5, 0.1)
    width_cm = st.slider("Pothole width (cm)", 0.0, 150.0, 40.0, 1.0)
    speed_kmh = st.slider("Vehicle speed (km/h)", 0.0, 120.0, 50.0, 1.0)
    severity = st.slider("Severity score (0..1)", 0.0, 1.0, 0.45, 0.01)

    X = pd.DataFrame([{
        "depth_m_est": depth_cm / 100.0,
        "width_m_est": width_cm / 100.0,
        "speed_mps": speed_kmh / 3.6,
        "severity_score": severity,
        "road_event": road_event
    }])

    if st.button("Predict mode"):
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        conf = float(proba.max())

        st.subheader("Result")
        st.write(f"Predicted mode: **{pred.upper()}**")
        st.write(f"Confidence: {conf*100:.1f}%")

        st.subheader("Probabilities")
        prob_df = pd.DataFrame({"mode": classes, "probability": proba}).sort_values("probability", ascending=False)
        st.dataframe(prob_df, hide_index=True)

if __name__ == "__main__":
    main()

