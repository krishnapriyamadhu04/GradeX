# app.py
# ─────────────────────────────────────────────────────────────────
# GradeX — Performance & CGPA Predictor
# Streamlit web application
#
# Run with:  streamlit run app.py
#
# Navigation is handled entirely via st.session_state["page"]:
#   "input"   → Page 1 — Student Academic Details form
#   "results" → Page 2 — Prediction results
# ─────────────────────────────────────────────────────────────────

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import FEATURE_COLS

# ── Page configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="GradeX",
    page_icon="🎓",
    layout="centered",
)

# ── Load models & scaler (cached — loaded only once) ─────────────
@st.cache_resource
def load_artifacts():
    rf     = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return rf, scaler

rf_model, scaler = load_artifacts()

# ── Initialise session state ──────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "input"

# ── Theme palette ─────────────────────────────────────────────────
C_BG        = "#F8F7F4"   # page background
C_CARD      = "#FFFFFF"   # card/container fill
C_ACCENT    = "#6C63A8"   # primary / lavender
C_ACCENT_LT = "#EAE7F5"   # soft accent background
C_TEXT      = "#292735"   # dark charcoal text
C_MUTED     = "#777481"   # secondary / muted text
C_BORDER    = "#DDD9EE"   # subtle border
C_SUCCESS   = "#6F9277"   # green
C_WARN      = "#B08A55"   # amber
C_DANGER    = "#B05555"   # red (muted)

# ── Global CSS ────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
        /* ── Page background & base text ── */
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {C_BG} !important;
            color: {C_TEXT};
        }}
        [data-testid="stMain"] {{
            background-color: {C_BG} !important;
        }}
        /* ── Block container spacing ── */
        .block-container {{
            padding-top: 1.8rem !important;
            padding-bottom: 1rem !important;
        }}
        /* ── Remove excess gaps between Streamlit rows ── */
        div[data-testid="stVerticalBlock"] > div {{ margin-bottom: 0 !important; }}
        div[data-testid="stNumberInput"]   {{ margin-bottom: 0.15rem !important; }}
        h3 {{ margin-top: 0.4rem !important; margin-bottom: 0.3rem !important; }}
        /* ── Headings ── */
        h1, h2, h3, h4 {{ color: {C_TEXT} !important; }}
        /* ── Primary button → accent lavender ── */
        div[data-testid="stButton"] button[kind="primary"] {{
            background-color: {C_ACCENT} !important;
            border: none !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            padding: 0.55rem 1rem !important;
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover {{
            background-color: #5a5295 !important;
        }}
        /* ── Secondary button (back) ── */
        div[data-testid="stButton"] button[kind="secondary"] {{
            background-color: {C_CARD} !important;
            border: 1.5px solid {C_BORDER} !important;
            color: {C_TEXT} !important;
            border-radius: 8px !important;
        }}
        div[data-testid="stButton"] button[kind="secondary"]:hover {{
            border-color: {C_ACCENT} !important;
            color: {C_ACCENT} !important;
        }}
        /* ── Streamlit divider ── */
        hr {{ border-color: {C_BORDER} !important; opacity: 0.6 !important; }}
        /* ── Number input styling ── */
        div[data-testid="stNumberInput"] input {{
            background-color: {C_CARD} !important;
            border: 1.5px solid {C_BORDER} !important;
            border-radius: 8px !important;
            color: {C_TEXT} !important;
        }}
        div[data-testid="stNumberInput"] label {{
            color: {C_TEXT} !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
        }}
        /* ── Metric widget ── */
        [data-testid="stMetric"] {{
            background: {C_CARD};
            border: 1px solid {C_BORDER};
            border-radius: 10px;
            padding: 0.6rem 0.75rem !important;
        }}
        [data-testid="stMetricLabel"] {{ color: {C_MUTED} !important; font-size: 0.78rem !important; }}
        [data-testid="stMetricValue"] {{ color: {C_TEXT} !important; font-size: 1.15rem !important; font-weight: 700 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════
#  SHARED HEADER  (shown on both pages)
# ════════════════════════════════════════════════════════════════
def render_header(subtitle: str = "Performance &amp; CGPA Predictor") -> None:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 0.1rem 0 0.2rem 0;">
            <h1 style="font-size:2.4rem; font-weight:800; margin:0 0 0.05rem 0;
                       color:{C_ACCENT}; letter-spacing:-0.5px;">GradeX</h1>
            <p style="font-size:1rem; color:{C_MUTED}; margin:0;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


# ════════════════════════════════════════════════════════════════
#  PAGE 1 — INPUT PAGE
# ════════════════════════════════════════════════════════════════
def page_input() -> None:
    render_header()

    st.markdown(
        f"<p style='text-align:center;color:{C_MUTED};font-size:0.9rem;margin:0 0 0.5rem 0;'>"
        "Enter your academic details to estimate your current semester CGPA."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(":material/school: **Student Academic Details**")

    col_a, col_b = st.columns(2)

    with col_a:
        internal_exam = st.number_input(
            "Internal Exam (out of 50)",
            min_value=0.0, max_value=50.0,
            value=st.session_state.get("internal_exam", 35.0),
            step=0.5,
            help="Internal examination score — maximum 50 marks",
        )
        lab_practical = st.number_input(
            "Lab / Practical (out of 100)",
            min_value=0.0, max_value=100.0,
            value=st.session_state.get("lab_practical", 72.0),
            step=1.0,
            help="Lab or practical examination score — maximum 100 marks",
        )
        previous_semester_cgpa = st.number_input(
            "Previous Semester CGPA (out of 10)",
            min_value=0.0, max_value=10.0,
            value=st.session_state.get("previous_semester_cgpa", 6.5),
            step=0.1,
            help="CGPA secured in the previous semester — 0 to 10",
        )

    with col_b:
        assignment = st.number_input(
            "Assignment (out of 15)",
            min_value=0.0, max_value=15.0,
            value=st.session_state.get("assignment", 10.0),
            step=0.5,
            help="Assignment score — maximum 15 marks",
        )
        attendance = st.number_input(
            "Attendance (out of 10)",
            min_value=0.0, max_value=10.0,
            value=st.session_state.get("attendance", 7.5),
            step=0.5,
            help="Attendance marks — maximum 10",
        )

    if st.button(":material/auto_graph: Predict CGPA", type="primary", use_container_width=True):
        # Persist inputs and run prediction before switching page
        st.session_state["internal_exam"]          = internal_exam
        st.session_state["assignment"]             = assignment
        st.session_state["lab_practical"]          = lab_practical
        st.session_state["attendance"]             = attendance
        st.session_state["previous_semester_cgpa"] = previous_semester_cgpa

        # Run prediction and cache the result
        input_data = pd.DataFrame(
            [[internal_exam, assignment, lab_practical,
              attendance, previous_semester_cgpa]],
            columns=FEATURE_COLS,
        )
        input_scaled = scaler.transform(input_data)
        rf_pred = float(np.clip(rf_model.predict(input_scaled)[0], 0.0, 10.0))
        st.session_state["rf_pred"] = rf_pred

        st.session_state["page"] = "results"
        st.rerun()


# ════════════════════════════════════════════════════════════════
#  PAGE 2 — RESULTS PAGE
# ════════════════════════════════════════════════════════════════
def page_results() -> None:
    render_header()

    rf_pred = st.session_state["rf_pred"]

    # Recover inputs from session state
    internal_exam          = st.session_state["internal_exam"]
    assignment             = st.session_state["assignment"]
    lab_practical          = st.session_state["lab_practical"]
    attendance             = st.session_state["attendance"]
    previous_semester_cgpa = st.session_state["previous_semester_cgpa"]

    # ── Grade label ──────────────────────────────────────────────
    def grade_label(cgpa: float) -> str:
        if cgpa >= 9.0: return "Outstanding"
        if cgpa >= 8.0: return "Excellent"
        if cgpa >= 7.0: return "Good"
        if cgpa >= 6.0: return "Average"
        if cgpa >= 5.0: return "Below Average"
        return "Poor"

    # ── 1. Predicted CGPA card ───────────────────────────────────
    st.markdown(":material/auto_graph: **Your Prediction**")
    st.markdown(
        f"""
        <div style="background:{C_CARD};border:1.5px solid {C_BORDER};
                    border-radius:16px;padding:2rem 1.5rem;text-align:center;
                    margin-bottom:1.2rem;
                    box-shadow:0 2px 12px rgba(108,99,168,0.08);">
            <div style="font-size:0.85rem;color:{C_MUTED};margin-bottom:0.35rem;
                        font-weight:500;letter-spacing:0.03em;text-transform:uppercase;">
                Predicted Current Semester CGPA
            </div>
            <div style="font-size:3.8rem;font-weight:800;color:{C_ACCENT};line-height:1.1;">
                {rf_pred:.2f}
                <span style="font-size:1.5rem;color:{C_MUTED};font-weight:400;"> / 10</span>
            </div>
            <div style="display:inline-block;margin-top:0.6rem;
                        background:{C_ACCENT_LT};color:{C_ACCENT};
                        font-size:0.9rem;font-weight:600;
                        padding:0.25rem 1rem;border-radius:20px;">
                {grade_label(rf_pred)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Input Summary ─────────────────────────────────────────
    st.markdown(":material/assignment: **Input Summary**")
    s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
    for col, label, value, max_marks in zip(
        [s_col1, s_col2, s_col3, s_col4, s_col5],
        ["Internal Exam", "Assignment", "Lab / Practical", "Attendance", "Prev. CGPA"],
        [internal_exam, assignment, lab_practical, attendance, previous_semester_cgpa],
        [50, 15, 100, 10, 10],
    ):
        with col:
            st.metric(label=f"{label} / {max_marks}", value=f"{value:.1f}")
    st.divider()

    # ── 3. What Your Prediction Means ────────────────────────────
    if rf_pred >= 8.0:
        insight_bg     = "#F0FAF2"
        insight_border = "#B6DBBE"
        insight_title_color = C_SUCCESS
        insight_msg    = (
            "You're on a strong track! Your predicted CGPA indicates good overall "
            "academic performance. Keep maintaining your consistency across internals, "
            "assignments, practicals, and attendance."
        )
    elif rf_pred >= 6.0:
        insight_bg     = "#FDF8EE"
        insight_border = "#E8D5A8"
        insight_title_color = C_WARN
        insight_msg    = (
            "You're doing well — keep pushing! Your prediction shows a solid academic "
            "performance. Improving your internal marks and practical performance could "
            "help you move toward a higher CGPA."
        )
    else:
        insight_bg     = "#FDF2F2"
        insight_border = "#E8B8B8"
        insight_title_color = C_DANGER
        insight_msg    = (
            "There's room to improve! Focusing on internals, assignments, practicals, "
            "and consistent attendance could help improve your academic performance."
        )

    st.markdown(
        f"""
        <div style="background:{insight_bg};border:1.5px solid {insight_border};
                    border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:1rem;">
            <div style="font-size:0.95rem;font-weight:700;color:{insight_title_color};
                        margin-bottom:0.45rem;display:flex;align-items:center;gap:0.4rem;">
                &#128161; What Your Prediction Means
            </div>
            <div style="font-size:0.92rem;color:{C_TEXT};line-height:1.65;">
                {insight_msg}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 4. Focus Areas ───────────────────────────────────────────
    def focus_status(value: float, maximum: float) -> tuple[str, str]:
        pct = (value / maximum) * 100
        if pct >= 90:   return "Strong",               C_SUCCESS
        elif pct >= 75: return "Good",                 C_ACCENT
        elif pct >= 60: return "Fair",                 C_WARN
        else:           return "Room for improvement", C_DANGER

    focus_items = [
        ("Internal Exam",          internal_exam,          50,  f"{internal_exam:.1f}"),
        ("Assignment",             assignment,             15,  f"{assignment:.1f}"),
        ("Lab / Practical",        lab_practical,          100, f"{lab_practical:.1f}"),
        ("Attendance",             attendance,             10,  f"{attendance:.1f}"),
        ("Previous Semester CGPA", previous_semester_cgpa, 10,  f"{previous_semester_cgpa:.2f}"),
    ]

    rows_html = ""
    for label, value, maximum, display_val in focus_items:
        status, colour = focus_status(value, maximum)
        rows_html += f"""
        <div style="display:grid;grid-template-columns:1fr auto auto;
                    align-items:center;gap:0 1rem;
                    padding:0.55rem 0;border-bottom:1px solid {C_BORDER};">
            <span style="font-size:0.9rem;color:{C_TEXT};font-weight:500;
                         white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</span>
            <span style="font-size:0.9rem;color:{C_MUTED};
                         white-space:nowrap;text-align:right;">{display_val}&thinsp;/&thinsp;{maximum}</span>
            <span style="font-size:0.83rem;font-weight:600;color:{colour};
                         white-space:nowrap;width:10rem;text-align:right;">{status}</span>
        </div>"""

    st.markdown(
        f"""
        <div style="background:{C_CARD};border:1.5px solid {C_BORDER};
                    border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;
                    box-shadow:0 1px 6px rgba(108,99,168,0.06);">
            <div style="font-size:0.95rem;font-weight:700;color:{C_TEXT};
                        margin-bottom:0.75rem;display:flex;align-items:center;gap:0.4rem;">
                &#127919; Focus Areas
            </div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Disclaimer ───────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="font-size:0.78rem;color:{C_MUTED};margin-top:0.4rem;
                    background:{C_ACCENT_LT};border:1px solid {C_BORDER};
                    border-radius:8px;padding:0.7rem 1rem;">
            <em>Note: This is an ML-based prediction. Actual CGPA may vary based on your final semester performance and university evaluation.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Back button ──────────────────────────────────────────────
    if st.button("← Edit Details", use_container_width=True):
        st.session_state["page"] = "input"
        st.rerun()


# ════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════
if st.session_state["page"] == "results" and "rf_pred" in st.session_state:
    page_results()
else:
    page_input()
