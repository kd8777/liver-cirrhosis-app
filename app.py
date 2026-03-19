"""
LiverGuard AI – Liver Cirrhosis Prediction System
Run: streamlit run app.py
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')

# ── Path setup – must be FIRST ─────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st

st.set_page_config(
    page_title="LiverGuard AI – Cirrhosis Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (after path setup) ─────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import joblib
    from utils.model_utils import (
        load_data, preprocess, train_model, load_model, model_exists,
        predict_single, STAGE_LABELS, STAGE_COLORS, RISK_LEVEL,
        DATA_PATH, MODEL_PATH
    )
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  { font-family:'DM Sans',sans-serif; }
.main                        { background:#0a0f1e; color:#e2e8f0; }
.block-container             { padding:1.5rem 2rem; max-width:1400px; }

section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0d1b2a,#0f2438 60%,#0a1628);
    border-right:1px solid rgba(56,189,248,.15);
}

/* ── Hero ── */
.hero {
    background:linear-gradient(135deg,#0d1b2a,#0f2d44 50%,#0d2235);
    border:1px solid rgba(56,189,248,.2); border-radius:20px;
    padding:2.5rem 3rem; margin-bottom:1.8rem; position:relative; overflow:hidden;
}
.hero::before {
    content:''; position:absolute; top:-50%; right:-10%;
    width:400px; height:400px;
    background:radial-gradient(circle,rgba(56,189,248,.08),transparent 70%);
    border-radius:50%;
}
.hero-badge {
    display:inline-block;
    background:rgba(56,189,248,.12); border:1px solid rgba(56,189,248,.3);
    color:#38bdf8; padding:.3rem 1rem; border-radius:100px;
    font-size:.78rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; margin-bottom:.8rem;
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
    background:linear-gradient(135deg,#38bdf8,#818cf8 50%,#c084fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0; line-height:1.1;
}
.hero-sub { font-size:1rem; color:#94a3b8; margin-top:.5rem; font-weight:300; }

/* ── Metric cards ── */
.mcard {
    background:linear-gradient(135deg,#0f1f35,#0d1a2d);
    border:1px solid rgba(56,189,248,.15); border-radius:16px;
    padding:1.3rem 1.5rem; text-align:center;
    transition:transform .2s,border-color .2s;
}
.mcard:hover { transform:translateY(-3px); border-color:rgba(56,189,248,.35); }
.mval { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#38bdf8; }
.mlbl { font-size:.75rem; color:#64748b; text-transform:uppercase;
        letter-spacing:.08em; margin-top:.3rem; font-weight:600; }

/* ── Section title ── */
.stitle {
    font-family:'Syne',sans-serif; font-size:1.35rem; font-weight:700;
    color:#e2e8f0; margin:1rem 0 .7rem; display:flex; align-items:center; gap:.5rem;
}
.stitle span { width:4px; height:22px;
    background:linear-gradient(180deg,#38bdf8,#818cf8); border-radius:4px; }

/* ── Input box ── */
.ibox {
    background:linear-gradient(135deg,#0f1f35,#0d1a2d);
    border:1px solid rgba(56,189,248,.12); border-radius:14px;
    padding:1.5rem 1.6rem; margin-bottom:.8rem;
}
.ititle {
    font-family:'Syne',sans-serif; font-size:.82rem; font-weight:700;
    color:#38bdf8; text-transform:uppercase; letter-spacing:.1em;
    margin-bottom:.8rem; padding-bottom:.5rem;
    border-bottom:1px solid rgba(56,189,248,.1);
}

/* ── Predict button ── */
.stButton>button {
    background:linear-gradient(135deg,#0284c7,#6366f1) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:1rem !important; padding:.8rem 2rem !important; width:100% !important;
    letter-spacing:.04em !important; transition:all .3s !important;
}
.stButton>button:hover { transform:translateY(-2px) !important;
    box-shadow:0 8px 25px rgba(56,189,248,.3) !important; }

/* ── Result card ── */
.rcard {
    border-radius:18px; padding:2rem; text-align:center;
    position:relative; overflow:hidden;
}
.rstage { font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:800; margin-top:.4rem; }
.rrisk  { font-size:.8rem; font-weight:700; letter-spacing:.12em;
          text-transform:uppercase; padding:.3rem 1rem; border-radius:100px;
          display:inline-block; margin-top:.6rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(15,31,53,.8) !important; border-radius:10px !important;
    padding:3px !important; gap:3px !important;
    border:1px solid rgba(56,189,248,.1) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
    color:#64748b !important; border-radius:7px !important; padding:.45rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,rgba(56,189,248,.15),rgba(129,140,248,.15)) !important;
    color:#38bdf8 !important;
}

/* ── Sidebar ── */
.sb-logo { text-align:center; padding:.8rem 0 1.2rem;
           border-bottom:1px solid rgba(56,189,248,.1); margin-bottom:1.2rem; }
.sb-name { font-family:'Syne',sans-serif; font-size:1.35rem; font-weight:800;
           background:linear-gradient(135deg,#38bdf8,#818cf8);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:#0a0f1e}
::-webkit-scrollbar-thumb{background:rgba(56,189,248,.3);border-radius:3px}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GUARD: imports failed?
# ══════════════════════════════════════════════════════════════════════════════
if not IMPORTS_OK:
    st.error(f"❌ Import error: {IMPORT_ERROR}")
    st.info("Run: `pip install -r requirements.txt` then restart the app.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def ensure_model():
    if model_exists():
        try:
            return load_model()
        except Exception:
            pass   # corrupt file – retrain
    with st.spinner("⚙️ Training Random Forest model… please wait ~15 seconds"):
        df = load_data()
        bundle = train_model(df)
    st.success(f"✅ Model trained! Accuracy: {bundle['metrics']['accuracy']}%")
    return bundle


@st.cache_resource(show_spinner=False)
def get_model():
    return ensure_model()


def hex_rgb(h):
    h = h.lstrip('#')
    return ','.join(str(int(h[i:i+2],16)) for i in (0,2,4))


def gauge(prob, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob*100,1),
        title={'text':title,'font':{'size':11,'color':'#94a3b8','family':'DM Sans'}},
        number={'suffix':'%','font':{'size':18,'color':color,'family':'Syne'}},
        gauge={
            'axis':{'range':[0,100],'tickcolor':'#334155','tickfont':{'size':8,'color':'#64748b'}},
            'bar':{'color':color,'thickness':.25},
            'bgcolor':'rgba(15,31,53,.8)','borderwidth':0,
            'steps':[{'range':[0,100],'color':'rgba(56,189,248,.04)'}],
        }
    ))
    fig.update_layout(height=170,margin=dict(l=20,r=20,t=38,b=8),
                      paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    return fig


def proba_bar(pd_):
    stages=sorted(pd_.keys()); vals=[pd_[s]*100 for s in stages]
    labels=[f"Stage {s}" for s in stages]; clrs=[STAGE_COLORS[s] for s in stages]
    fig=go.Figure(go.Bar(x=labels,y=vals,marker_color=clrs,
        text=[f"{v:.1f}%" for v in vals],textposition='outside',
        textfont=dict(color='#e2e8f0',size=12,family='Syne')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font={'color':'#e2e8f0','family':'DM Sans'},height=260,
        margin=dict(l=10,r=10,t=15,b=15),
        xaxis=dict(showgrid=False,color='#334155'),
        yaxis=dict(showgrid=True,gridcolor='rgba(56,189,248,.07)',color='#64748b',range=[0,115]),
        bargap=.35,showlegend=False)
    return fig


def radar_chart(inp):
    cats=['Bilirubin','Albumin','Prothrombin','Copper','SGOT','Platelets']
    nmax={'Bilirubin':1.2,'Albumin':5.0,'Prothrombin':13,'Copper':140,'SGOT':40,'Platelets':400}
    vals=[min(inp.get(c,0)/nmax.get(c,1),2.5) for c in cats]+[min(inp.get(cats[0],0)/nmax[cats[0]],2.5)]
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],fill='toself',
        fillcolor='rgba(56,189,248,.1)',line=dict(color='#38bdf8',width=2),
        marker=dict(color='#38bdf8',size=5)))
    fig.add_trace(go.Scatterpolar(r=[1]*len(cats+[cats[0]]),theta=cats+[cats[0]],
        line=dict(color='rgba(100,116,139,.4)',width=1,dash='dot'),showlegend=False,mode='lines'))
    fig.update_layout(
        polar=dict(bgcolor='rgba(15,31,53,.5)',
            radialaxis=dict(visible=True,range=[0,2.5],showticklabels=False,
                            gridcolor='rgba(56,189,248,.1)'),
            angularaxis=dict(gridcolor='rgba(56,189,248,.1)',color='#94a3b8',tickfont=dict(size=10))),
        paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=40,r=40,t=35,b=35),height=270,showlegend=False)
    return fig


def cm_fig(cm, labels):
    fig=px.imshow(cm,labels=dict(x="Predicted",y="Actual",color="Count"),
        x=labels,y=labels,
        color_continuous_scale=[[0,'#0f1f35'],[.5,'#0284c7'],[1,'#38bdf8']],
        text_auto=True)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font={'color':'#e2e8f0','family':'DM Sans'},margin=dict(l=20,r=20,t=25,b=15),
        height=340,coloraxis_showscale=False)
    fig.update_traces(textfont_size=13,textfont_color='white')
    return fig


def fi_fig(fi, n=15):
    top=fi.head(n)
    clrs=px.colors.sequential.Blues[2:]
    bc=[clrs[int(i*(len(clrs)-1)/(len(top)-1))] for i in range(len(top))]
    fig=go.Figure(go.Bar(x=top['importance'],y=top['feature'],orientation='h',
        marker=dict(color=bc),
        text=[f"{v*100:.1f}%" for v in top['importance']],
        textposition='outside',textfont=dict(color='#94a3b8',size=10)))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font={'color':'#e2e8f0','family':'DM Sans'},height=400,
        margin=dict(l=15,r=55,t=15,b=15),
        xaxis=dict(showgrid=False,color='#334155',zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,color='#94a3b8'),bargap=.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
bundle  = get_model()
metrics = bundle['metrics']

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div style="font-size:2.3rem">🫀</div>
        <div class="sb-name">LiverGuard AI</div>
        <div style="color:#475569;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;margin-top:.2rem">
            Cirrhosis Prediction System
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Home & Predict",
        "📊  Model Analytics",
        "📁  Dataset Explorer",
        "ℹ️   About",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="color:#475569;font-size:.78rem;line-height:1.7">
        <b style="color:#64748b">Dataset</b><br>
        Kaggle Cirrhosis Prediction<br>
        <span style="color:#38bdf8">fedesoriano</span><br><br>
        <b style="color:#64748b">Algorithm</b><br>
        Random Forest + Gradient Boost<br><br>
        <b style="color:#64748b">Target</b><br>
        Cirrhosis Stage 1–4
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🔄  Retrain Model"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        st.cache_resource.clear()
        st.rerun()

    st.markdown("""
    <div style="color:#334155;font-size:.68rem;text-align:center;margin-top:1rem;padding:0 .5rem">
        ⚕️ Educational use only<br>Not a substitute for medical advice
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME & PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "Home" in page:

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🧬 AI-Powered Medical Diagnostics</div>
        <h1 class="hero-title">Liver Cirrhosis<br>Prediction System</h1>
        <p class="hero-sub">
            Random Forest ensemble trained on Mayo Clinic PBC trial data.<br>
            Enter patient biomarkers below to predict cirrhosis stage.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metric row
    c1,c2,c3,c4 = st.columns(4)
    for col,(val,lbl) in zip([c1,c2,c3,c4],[
        (f"{metrics['accuracy']}%","Accuracy"),
        (f"{metrics['auc_roc']}%","AUC-ROC"),
        (f"{metrics['f1_score']}%","F1 Score"),
        (f"{metrics['oob_score']}%","OOB Score"),
    ]):
        col.markdown(f'<div class="mcard"><div class="mval">{val}</div>'
                     f'<div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="stitle"><span></span>Patient Data Input</div>', unsafe_allow_html=True)

    # ── FORM ──────────────────────────────────────────────────────────────────
    with st.form("pred_form"):

        st.markdown('<div class="ibox"><div class="ititle">👤 Demographics</div>', unsafe_allow_html=True)
        d1,d2,d3,d4 = st.columns(4)
        age    = d1.number_input("Age (years)", 18, 90, 52, 1)
        sex    = d2.selectbox("Sex", ["F","M"])
        drug   = d3.selectbox("Drug Treatment", ["D-penicillamine","Placebo"])
        status = d4.selectbox("Patient Status", ["C – Censored","CL – Liver Tx","D – Death"])
        n_days = st.slider("Follow-up Days", 41, 5000, 1500)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ibox"><div class="ititle">🩺 Clinical Symptoms</div>', unsafe_allow_html=True)
        s1,s2,s3,s4 = st.columns(4)
        ascites   = s1.selectbox("Ascites",         ["N","Y"])
        hepatomeg = s2.selectbox("Hepatomegaly",    ["N","Y"])
        spiders   = s3.selectbox("Spider Angioma",  ["N","Y"])
        edema     = s4.selectbox("Edema",           ["N","S","Y"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ibox"><div class="ititle">🔬 Laboratory Values</div>', unsafe_allow_html=True)
        l1,l2,l3 = st.columns(3)
        bilirubin   = l1.number_input("Bilirubin (mg/dl)",       0.1, 30.0,  1.2, 0.1)
        cholesterol = l1.number_input("Cholesterol (mg/dl)",     50.0,1500.0,300.0,10.0)
        albumin     = l1.number_input("Albumin (gm/dl)",         1.0,  6.0,  3.5, 0.1)
        copper      = l2.number_input("Urine Copper (ug/day)",   0.0, 600.0, 73.0, 1.0)
        alk_phos    = l2.number_input("Alkaline Phosphatase (U/L)",100.0,15000.0,1000.0,10.0)
        sgot        = l2.number_input("SGOT (U/ml)",             20.0, 500.0,120.0, 1.0)
        tryglice    = l3.number_input("Triglycerides (mg/dl)",   30.0, 600.0,124.0, 1.0)
        platelets   = l3.number_input("Platelets (ml/1000)",     50.0, 700.0,257.0, 1.0)
        prothrombin = l3.number_input("Prothrombin Time (s)",     8.0,  20.0, 10.7, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍  Predict Cirrhosis Stage")

    # ── RESULT ────────────────────────────────────────────────────────────────
    if submitted:
        sm = {"C – Censored":"C","CL – Liver Tx":"CL","D – Death":"D"}
        inp = {
            'N_Days':n_days,'Age':age,'Drug':drug,'Sex':sex,
            'Ascites':ascites,'Hepatomegaly':hepatomeg,'Spiders':spiders,'Edema':edema,
            'Status':sm[status],
            'Bilirubin':bilirubin,'Cholesterol':cholesterol,'Albumin':albumin,
            'Copper':copper,'Alk_Phos':alk_phos,'SGOT':sgot,
            'Tryglicerides':tryglice,'Platelets':platelets,'Prothrombin':prothrombin,
        }

        with st.spinner("🧠 Analysing biomarkers…"):
            time.sleep(0.5)
            pred, proba = predict_single(bundle, inp)

        sc   = STAGE_COLORS[pred]
        risk = RISK_LEVEL[pred]
        conf = proba[pred]*100
        risk_cls_map={1:"#22c55e",2:"#eab308",3:"#f97316",4:"#ef4444"}
        rc   = risk_cls_map[pred]
        icon = {1:"🟢",2:"🟡",3:"🟠",4:"🔴"}[pred]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="stitle"><span></span>Prediction Results</div>', unsafe_allow_html=True)

        r1,r2 = st.columns([1,1.4])
        with r1:
            st.markdown(f"""
            <div class="rcard" style="background:linear-gradient(135deg,
                rgba({hex_rgb(sc)},.12),rgba(15,31,53,.95));
                border:2px solid {sc}44;">
                <div style="font-size:3.5rem">{icon}</div>
                <div class="rstage" style="color:{sc}">{STAGE_LABELS[pred]}</div>
                <div class="rrisk" style="background:rgba({hex_rgb(rc)},.12);
                    color:{rc};border:1px solid {rc}44">{risk}</div>
                <div style="margin-top:1rem;color:#64748b;font-size:.82rem">Model Confidence</div>
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{sc}">
                    {conf:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(radar_chart(inp), use_container_width=True, config={'displayModeBar':False})

        with r2:
            st.markdown("**Stage Probability Distribution**")
            st.plotly_chart(proba_bar(proba), use_container_width=True, config={'displayModeBar':False})
            g1,g2 = st.columns(2)
            for i,(s,p) in enumerate(sorted(proba.items())):
                (g1 if i%2==0 else g2).plotly_chart(
                    gauge(p, f"Stage {s}", STAGE_COLORS[s]),
                    use_container_width=True, config={'displayModeBar':False})

        notes={
            1:"Early-stage cirrhosis. Regular monitoring, lifestyle modifications and medication adherence are key.",
            2:"Moderate cirrhosis. Close monitoring, dietary adjustments and specialist follow-up are advised.",
            3:"Severe cirrhosis. Immediate specialist consultation required. Evaluate for advanced therapies.",
            4:"End-stage cirrhosis. Urgent medical attention needed. Liver transplantation evaluation recommended.",
        }
        st.markdown(f"""
        <div style="background:rgba({hex_rgb(rc)},.08);border-left:4px solid {rc};
             border-radius:0 12px 12px 0;padding:1rem 1.5rem;margin-top:1rem">
            <b style="color:{rc}">⚕️ Clinical Note</b><br>
            <span style="color:#cbd5e1;font-size:.9rem">{notes[pred]}</span><br>
            <span style="color:#475569;font-size:.75rem;margin-top:.4rem;display:block">
                ⚠️ AI prediction for educational use only. Consult a qualified hepatologist.
            </span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown("""
    <div class="hero" style="padding:2rem 3rem">
        <div class="hero-badge">📊 Model Performance</div>
        <h1 class="hero-title" style="font-size:2rem">Model Analytics Dashboard</h1>
        <p class="hero-sub">Comprehensive evaluation of the Random Forest ensemble.</p>
    </div>
    """, unsafe_allow_html=True)

    m = metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(val,lbl) in zip([c1,c2,c3,c4,c5],[
        (f"{m['accuracy']}%","Test Accuracy"),
        (f"{m['f1_score']}%","F1 Score"),
        (f"{m['precision']}%","Precision"),
        (f"{m['recall']}%","Recall"),
        (f"{m['auc_roc']}%","AUC-ROC"),
    ]):
        col.markdown(f'<div class="mcard"><div class="mval" style="font-size:1.7rem">{val}</div>'
                     f'<div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    t1,t2,t3 = st.tabs(["🎯 Confusion Matrix","📈 Feature Importance","📋 Classification Report"])

    with t1:
        cm = bundle['confusion_matrix']
        yt = bundle['y_test']
        stages = sorted(yt.unique())
        labels = [f"Stage {s}" for s in stages]
        c1,c2 = st.columns([1.2,1])
        with c1:
            st.markdown('<div class="stitle"><span></span>Confusion Matrix</div>', unsafe_allow_html=True)
            st.plotly_chart(cm_fig(cm, labels), use_container_width=True)
        with c2:
            st.markdown('<div class="stitle"><span></span>Performance Summary</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Metric":["Train Samples","Test Samples","CV Accuracy","CV Std","OOB Score"],
                "Value":[str(m['train_size']),str(m['test_size']),
                         f"{m['cv_mean']}%",f"±{m['cv_std']}%",f"{m['oob_score']}%"],
            }), hide_index=True, use_container_width=True)
            for i,s in enumerate(stages):
                total=cm[i].sum(); correct=cm[i][i]
                pct=correct/total*100 if total>0 else 0
                sc=STAGE_COLORS[s]
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                     padding:.45rem 0;border-bottom:1px solid rgba(56,189,248,.07)">
                    <span style="color:#94a3b8;font-size:.84rem">Stage {s}</span>
                    <div style="background:{sc}22;border:1px solid {sc}44;color:{sc};
                         padding:.18rem .65rem;border-radius:100px;font-size:.8rem;font-weight:700">
                        {pct:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)

    with t2:
        fi = bundle['feature_importance']
        c1,c2 = st.columns([1.5,1])
        with c1:
            st.markdown('<div class="stitle"><span></span>Top Feature Importances</div>', unsafe_allow_html=True)
            st.plotly_chart(fi_fig(fi), use_container_width=True)
        with c2:
            st.markdown('<div class="stitle"><span></span>Ranked Features</div>', unsafe_allow_html=True)
            st.dataframe(fi.head(15), hide_index=True, use_container_width=True, height=400)

    with t3:
        st.markdown('<div class="stitle"><span></span>Classification Report</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0f1f35;border:1px solid rgba(56,189,248,.12);
             border-radius:12px;padding:1.5rem;font-family:monospace;font-size:.84rem;
             color:#94a3b8;white-space:pre;overflow-x:auto">
{bundle['classification_report']}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown("""
    <div class="hero" style="padding:2rem 3rem">
        <div class="hero-badge">📁 Data Insights</div>
        <h1 class="hero-title" style="font-size:2rem">Dataset Explorer</h1>
        <p class="hero-sub">Explore the Cirrhosis Prediction dataset used for training.</p>
    </div>
    """, unsafe_allow_html=True)

    df_raw  = load_data()
    df_proc = preprocess(df_raw.copy())

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,lbl) in zip([c1,c2,c3,c4],[
        (len(df_raw),"Total Patients"),
        (df_raw.shape[1],"Features"),
        (df_raw.isnull().sum().sum(),"Missing Values"),
        (4,"Target Classes"),
    ]):
        col.markdown(f'<div class="mcard"><div class="mval">{val}</div>'
                     f'<div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    t1,t2,t3 = st.tabs(["📋 Data Preview","📊 Distributions","🔗 Correlations"])

    with t1:
        n = st.slider("Rows to show", 10, min(200, len(df_raw)), 20)
        st.dataframe(df_raw.head(n), use_container_width=True, height=380)
        c1,c2 = st.columns(2)
        with c1:
            info = pd.DataFrame({'Column':df_raw.columns,'Type':df_raw.dtypes.values,
                'Non-Null':df_raw.count().values,'Null':df_raw.isnull().sum().values,
                'Unique':df_raw.nunique().values})
            st.dataframe(info, hide_index=True, use_container_width=True)
        with c2:
            if 'Stage' in df_raw.columns:
                sc_ = df_raw['Stage'].value_counts().sort_index()
                fig = go.Figure(go.Bar(x=[f"Stage {i}" for i in sc_.index],y=sc_.values,
                    marker_color=[STAGE_COLORS.get(i,'#38bdf8') for i in sc_.index],
                    text=sc_.values,textposition='outside',textfont=dict(color='#e2e8f0')))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                    font={'color':'#e2e8f0'},height=280,margin=dict(l=10,r=10,t=15,b=15),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True,gridcolor='rgba(56,189,248,.07)'))
                st.plotly_chart(fig, use_container_width=True)

    with t2:
        num_cols = [c for c in df_raw.select_dtypes(include=np.number).columns
                    if c not in ['ID','Stage']]
        sel = st.selectbox("Feature to visualise", num_cols)
        c1,c2 = st.columns(2)
        with c1:
            fig=px.histogram(df_raw,x=sel,color='Stage',
                color_discrete_map={1:'#22c55e',2:'#eab308',3:'#f97316',4:'#ef4444'},
                nbins=30,barmode='overlay',opacity=.7)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                font={'color':'#e2e8f0'},height=290,margin=dict(l=10,r=10,t=25,b=15),
                legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig=px.box(df_raw,x='Stage',y=sel,color='Stage',
                color_discrete_map={1:'#22c55e',2:'#eab308',3:'#f97316',4:'#ef4444'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                font={'color':'#e2e8f0'},height=290,margin=dict(l=10,r=10,t=25,b=15),showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with t3:
        num_df = df_proc.select_dtypes(include=np.number).dropna(axis=1)
        corr   = num_df.corr()
        fig=px.imshow(corr,color_continuous_scale='RdBu_r',zmin=-1,zmax=1,aspect='auto')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',font={'color':'#e2e8f0'},
            height=540,margin=dict(l=10,r=10,t=25,b=15))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">ℹ️ Project Information</div>
        <h1 class="hero-title">About LiverGuard AI</h1>
        <p class="hero-sub">College project – ML applied to hepatology diagnostics.</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2 = st.columns([1.2,1])
    with c1:
        st.markdown("""
        <div class="ibox">
            <div class="ititle">🎯 Project Overview</div>
            <p style="color:#94a3b8;line-height:1.8;font-size:.9rem">
                LiverGuard AI predicts liver cirrhosis stage (1–4) using clinical biomarkers
                from the Mayo Clinic PBC trial. It uses a Random Forest + Gradient Boosting
                ensemble for maximum accuracy, with MELD score proxy and engineered features
                to boost predictive power.
            </p>
            <div class="ititle" style="margin-top:1.2rem">🔬 Dataset</div>
            <p style="color:#94a3b8;line-height:1.8;font-size:.9rem">
                <b style="color:#38bdf8">Source:</b> Kaggle – Cirrhosis Prediction (fedesoriano)<br>
                <b style="color:#38bdf8">Records:</b> 418 patients (Mayo Clinic 1974–1984)<br>
                <b style="color:#38bdf8">Target:</b> Liver fibrosis stage (1 mild → 4 end-stage)
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ibox"><div class="ititle">⚙️ Tech Stack</div>', unsafe_allow_html=True)
        for tech,desc in [
            ("🐍 Python 3.10+","Core language"),
            ("🌊 Streamlit","Web application"),
            ("🌲 Scikit-learn","Random Forest / ML"),
            ("📊 Plotly","Interactive charts"),
            ("🐼 Pandas / NumPy","Data processing"),
            ("💾 Joblib","Model persistence"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                 padding:.5rem 0;border-bottom:1px solid rgba(56,189,248,.07)">
                <span style="color:#e2e8f0;font-size:.86rem;font-weight:600">{tech}</span>
                <span style="color:#64748b;font-size:.8rem">{desc}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:2rem;padding:1.2rem;
         background:rgba(239,68,68,.05);border:1px solid rgba(239,68,68,.15);border-radius:12px;
         color:#94a3b8;font-size:.84rem">
        ⚠️ <b style="color:#ef4444">Medical Disclaimer:</b>
        This system is for educational and academic purposes only.
        Always consult a qualified hepatologist for medical decisions.
    </div>
    """, unsafe_allow_html=True)
