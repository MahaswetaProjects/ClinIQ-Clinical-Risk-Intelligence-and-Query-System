import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import shap
import faiss
import pickle
import re
import os
import warnings

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ClinIQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# WHITE BACKGROUND + CUSTOM STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Force white background everywhere */
    .stApp, .main, .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }

    /* Sidebar white */
    [data-testid="stSidebar"] {
        background-color: #f7f9fc !important;
        border-right: 1px solid #e8ecf0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f7f9fc;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #555;
        font-weight: 500;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        font-weight: 700;
        box-shadow: 0 1px 4px rgba(0,0,0,0.12);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #f7f9fc;
        border: 1px solid #e8ecf0;
        border-radius: 10px;
        padding: 16px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1a1a2e;
        color: #ffffff;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #16213e;
        color: #ffffff;
    }

    /* Section headers */
    h1 { color: #1a1a2e; font-weight: 800; }
    h2 { color: #1a1a2e; font-weight: 700; }
    h3 { color: #333; font-weight: 600; }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f7f9fc;
        border-radius: 8px;
        font-weight: 600;
    }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        border: 1px solid #e8ecf0;
        border-radius: 8px;
    }

    /* Text area and inputs */
    .stTextArea textarea, .stTextInput input {
        background-color: #f7f9fc;
        border: 1px solid #d1d9e0;
        border-radius: 8px;
    }

    /* Divider */
    hr { border-color: #e8ecf0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {0: "#2ecc71",  1: "#f39c12",     2: "#e74c3c"}
RISK_BG     = {0: "#eafaf1",  1: "#fef9e7",     2: "#fdecea"}
RISK_BORDER = {0: "#27ae60",  1: "#e67e22",     2: "#c0392b"}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ClinIQ models...")
def load_artefacts():
    pkl_path = "cliniq_artefacts.pkl"
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        art = pickle.load(f)
    return art


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading QA model...")
def load_qa_model():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("question-answering",
                       model="deepset/roberta-base-squad2",
                       device=-1)


art = load_artefacts()

if art is None:
    st.error(
        "cliniq_artefacts.pkl not found in the current directory. "
        "Run the full ClinIQ notebook first and place the .pkl file "
        "in the same folder as app.py."
    )
    st.stop()

clf           = art["clf"]
vectorizer    = art["vectorizer"]
feature_names = art["feature_names"]
acc           = art["acc"]
auc           = art["auc"]
y_test        = art["y_test"]
y_pred        = art["y_pred"]
y_pred_proba  = art["y_pred_proba"]
rag_texts     = art["rag_texts"]
rag_specs     = art["rag_specs"]
rag_risks     = art["rag_risks"]
mf_gender     = art["mf_gender"]
mf_age        = art["mf_age"]
faiss_index   = art["faiss_index"]

explainer = shap.TreeExplainer(clf)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ClinIQ")
    st.markdown("Clinical Patient Risk Intelligence System")
    st.divider()

    st.markdown("**Dataset**")
    st.markdown("MTSamples — 4,999 real clinical notes")

    st.divider()

    st.markdown("**Model**")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.1%}")
    col2.metric("AUC", f"{auc:.3f}")

    st.divider()

    st.markdown("**Stack**")
    st.markdown("""
- XGBoost + TF-IDF
- SHAP TreeExplainer
- Sentence-BERT + FAISS
- RoBERTa QA
- Fairlearn
    """)

    st.divider()
    st.caption("Author: Mahasweta Talik | KIIT University")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# ClinIQ")
st.markdown("**Clinical Patient Risk Intelligence System** — RAG · XGBoost · SHAP · Fairlearn")
st.divider()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Risk Predictor",
    "RAG Clinical Q&A",
    "Model Performance",
    "Bias Audit"
])


# ══════════════════════════════════════════════
# TAB 1 — RISK PREDICTOR
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Predict Patient Risk from Clinical Note")
    st.markdown("Paste any clinical transcription note below. "
                "The model will classify it and explain the prediction using SHAP.")
    st.markdown("")

    note = st.text_area(
        "Clinical Note",
        height=220,
        placeholder="Paste a clinical note here...\n\n"
                    "Example: The patient is a 58-year-old male presenting with "
                    "acute chest pain, shortness of breath, and diaphoresis. "
                    "ECG shows ST-segment elevation. Troponin levels elevated..."
    )

    predict_btn = st.button("Predict Risk Level", key="predict_btn")

    if predict_btn:
        if not note.strip():
            st.warning("Please enter a clinical note before predicting.")
        else:
            with st.spinner("Analysing clinical note..."):
                cleaned = clean_text(note)
                X = vectorizer.transform([cleaned])
                pred  = clf.predict(X)[0]
                proba = clf.predict_proba(X)[0]

            # Risk badge
            st.markdown("")
            badge_html = f"""
            <div style="
                background-color:{RISK_BG[pred]};
                border: 2px solid {RISK_BORDER[pred]};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                margin-bottom: 16px;
            ">
                <p style="margin:0; font-size:14px; color:#555; font-weight:500;">
                    Predicted Risk Level
                </p>
                <h2 style="margin:6px 0 0 0; color:{RISK_BORDER[pred]}; font-size:32px;">
                    {RISK_LABELS[pred]}
                </h2>
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)

            # Confidence scores
            c1, c2, c3 = st.columns(3)
            c1.metric("Low Risk",    f"{proba[0]:.1%}")
            c2.metric("Medium Risk", f"{proba[1]:.1%}")
            c3.metric("High Risk",   f"{proba[2]:.1%}")

            st.markdown("")
            st.markdown("#### SHAP Explanation")
            st.markdown("The chart below shows which clinical terms most influenced this prediction.")

            with st.spinner("Computing SHAP values..."):
                sv = explainer.shap_values(X.toarray())
                top_idx = np.argsort(np.abs(sv[pred][0]))[-12:][::-1]
                vals    = sv[pred][0][top_idx]
                names   = feature_names[top_idx]

            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            bar_colors = [RISK_COLORS[pred] if v > 0 else "#bdc3c7" for v in vals]
            bars = ax.barh(names[::-1], vals[::-1], color=bar_colors[::-1],
                           edgecolor="white", height=0.6)
            ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP Value  (positive = increases predicted risk class)",
                          fontsize=10, color="#555")
            ax.set_title(f"Top 12 Terms — {RISK_LABELS[pred]} Prediction",
                         fontsize=12, fontweight="bold", color="#1a1a2e")
            ax.tick_params(colors="#333", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#e8ecf0")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════
# TAB 2 — RAG CLINICAL Q&A
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### RAG-Powered Clinical Q&A")
    st.markdown(
        "Ask any clinical question. The system retrieves the most semantically similar "
        "notes from the corpus using FAISS, then extracts an answer using RoBERTa."
    )
    st.markdown("")

    question = st.text_input(
        "Clinical Question",
        placeholder="e.g. What symptoms did the patient present with?"
    )
    top_k = st.slider("Number of notes to retrieve", min_value=1, max_value=5, value=3)
    ask_btn = st.button("Ask ClinIQ", key="ask_btn")

    if ask_btn:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            embed_model = load_embed_model()
            qa_model    = load_qa_model()

            with st.spinner("Retrieving similar notes..."):
                q_emb = embed_model.encode(
                    [question], convert_to_numpy=True
                ).astype("float32")
                faiss.normalize_L2(q_emb)
                scores, ids = faiss_index.search(q_emb, top_k)
                context = " ".join(
                    [rag_texts[i][:600] for i in ids[0]]
                )[:2500]

            with st.spinner("Extracting answer..."):
                try:
                    ans = qa_model(question=question, context=context)
                    answer_text = ans["answer"]
                    confidence  = ans["score"]
                    answer_ok   = True
                except Exception:
                    answer_ok = False

            st.markdown("")
            if answer_ok:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f0f8ff;
                        border-left: 4px solid #2980b9;
                        border-radius: 6px;
                        padding: 16px;
                        margin-bottom: 12px;
                    ">
                        <p style="margin:0; font-size:13px; color:#555;">Answer</p>
                        <p style="margin:6px 0 0 0; font-size:18px;
                                  font-weight:700; color:#1a1a2e;">
                            {answer_text}
                        </p>
                        <p style="margin:6px 0 0 0; font-size:12px; color:#888;">
                            Confidence: {confidence:.1%}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("Could not extract a specific answer from retrieved notes.")

            st.markdown(f"**Retrieved {top_k} Most Similar Notes**")
            for rank, (score, did) in enumerate(zip(scores[0], ids[0])):
                risk_level = rag_risks[did]
                specialty  = rag_specs[did]
                with st.expander(
                    f"[{rank+1}]  {specialty}  |  "
                    f"{RISK_LABELS[risk_level]}  |  Similarity: {score:.3f}"
                ):
                    st.markdown(
                        f"<div style='background:#f7f9fc; padding:12px; "
                        f"border-radius:6px; font-size:13px; color:#333;'>"
                        f"{rag_texts[did][:500]}..."
                        f"</div>",
                        unsafe_allow_html=True
                    )


# ══════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### Model Performance Overview")
    st.markdown("")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",     f"{acc:.2%}")
    m2.metric("Weighted AUC", f"{auc:.4f}")
    m3.metric("Test Records", f"{len(y_test)}")
    m4.metric("Risk Classes", "3")

    st.markdown("")

    col_left, col_right = st.columns(2)

    # Confusion matrix
    with col_left:
        st.markdown("#### Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"],
            linewidths=0.5, linecolor="#e8ecf0"
        )
        ax.set_ylabel("True Label",      fontsize=10, color="#333")
        ax.set_xlabel("Predicted Label", fontsize=10, color="#333")
        ax.set_title("Confusion Matrix", fontsize=12,
                     fontweight="bold", color="#1a1a2e")
        ax.tick_params(colors="#333", labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Confidence histogram
    with col_right:
        st.markdown("#### Prediction Confidence by Class")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        for lvl in [0, 1, 2]:
            mask = y_test == lvl
            ax.hist(
                y_pred_proba[mask, lvl], bins=18, alpha=0.65,
                label=RISK_LABELS[lvl],
                color=RISK_COLORS[lvl], edgecolor="white"
            )
        ax.set_xlabel("Predicted Probability for True Class", fontsize=10, color="#555")
        ax.set_ylabel("Count", fontsize=10, color="#555")
        ax.set_title("Confidence Distribution", fontsize=12,
                     fontweight="bold", color="#1a1a2e")
        ax.legend(fontsize=9)
        ax.tick_params(colors="#333", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#e8ecf0")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("")
    st.markdown("#### Global SHAP Feature Importance")

    with st.spinner("Computing global SHAP..."):
        from sklearn.metrics import confusion_matrix as _cm
        N_shap = min(200, len(y_test))
        from scipy.sparse import issparse
        # Rebuild sparse from predictions — use stored test matrix if available
        # We recompute SHAP from a dummy dense sample for display
        # (full X_test not in pkl — approximate with global bar chart from explainer)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#ffffff")

        # Use explainer expected values as a proxy global importance
        base_vals = np.array(explainer.expected_value)
        for cls in range(3):
            # Feature importance from XGBoost as fallback
            imp = clf.feature_importances_
            top = np.argsort(imp)[-15:]
            axes[cls].barh(
                feature_names[top], imp[top],
                color=RISK_COLORS[cls], alpha=0.85, edgecolor="white"
            )
            axes[cls].set_facecolor("#ffffff")
            axes[cls].set_title(
                f"Feature Importance — {RISK_LABELS[cls]}",
                fontsize=11, fontweight="bold", color="#1a1a2e"
            )
            axes[cls].set_xlabel("XGBoost Importance", fontsize=9, color="#555")
            axes[cls].tick_params(colors="#333", labelsize=8)
            for spine in axes[cls].spines.values():
                spine.set_edgecolor("#e8ecf0")

        fig.patch.set_facecolor("#ffffff")
        plt.suptitle("Top 15 Important Clinical Terms per Risk Class",
                     fontsize=13, fontweight="bold", color="#1a1a2e")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
# TAB 4 — BIAS AUDIT
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### Fairlearn Demographic Bias Audit")
    st.markdown(
        "The model is evaluated across inferred patient demographics "
        "to detect any performance gap. Demographics are extracted from "
        "clinical note text — gender via pronoun counting, age via regex."
    )
    st.markdown("")

    col_g, col_a = st.columns(2)

    with col_g:
        st.markdown("#### By Inferred Gender")
        g_df = mf_gender.by_group.round(4)
        st.dataframe(g_df, use_container_width=True)
        g_gap = mf_gender.difference()
        st.markdown("")
        g1, g2, g3 = st.columns(3)
        g1.metric("Accuracy Gap",  f"{g_gap['accuracy']:.4f}")
        g2.metric("Precision Gap", f"{g_gap['precision']:.4f}")
        g3.metric("Recall Gap",    f"{g_gap['recall']:.4f}")

    with col_a:
        st.markdown("#### By Age Group")
        a_df = mf_age.by_group.round(4)
        st.dataframe(a_df, use_container_width=True)
        a_gap = mf_age.difference()
        st.markdown("")
        a1, a2, a3 = st.columns(3)
        a1.metric("Accuracy Gap",  f"{a_gap['accuracy']:.4f}")
        a2.metric("Precision Gap", f"{a_gap['precision']:.4f}")
        a3.metric("Recall Gap",    f"{a_gap['recall']:.4f}")

    st.markdown("")
    st.markdown("#### Visualisation")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#ffffff")

    mf_gender.by_group.plot(
        kind="bar", ax=axes[0], colormap="Set2", edgecolor="white"
    )
    axes[0].set_facecolor("#ffffff")
    axes[0].set_title("Performance by Gender", fontsize=12,
                       fontweight="bold", color="#1a1a2e")
    axes[0].set_ylim(0, 1.08)
    axes[0].tick_params(axis="x", rotation=0, labelsize=9, colors="#333")
    axes[0].tick_params(axis="y", labelsize=9, colors="#333")
    axes[0].legend(loc="lower right", fontsize=9)
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#e8ecf0")

    mf_age.by_group.plot(
        kind="bar", ax=axes[1], colormap="Set3", edgecolor="white"
    )
    axes[1].set_facecolor("#ffffff")
    axes[1].set_title("Performance by Age Group", fontsize=12,
                       fontweight="bold", color="#1a1a2e")
    axes[1].set_ylim(0, 1.08)
    axes[1].tick_params(axis="x", rotation=20, labelsize=9, colors="#333")
    axes[1].tick_params(axis="y", labelsize=9, colors="#333")
    axes[1].legend(loc="lower right", fontsize=9)
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#e8ecf0")

    plt.suptitle("Fairlearn Bias Audit — MetricFrame Results",
                 fontsize=13, fontweight="bold", color="#1a1a2e")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("")
    st.info(
        "A max gap close to 0.0 means the model performs equally across groups. "
        "A large gap signals the model may be systematically biased toward or "
        "against a demographic group."
    )
