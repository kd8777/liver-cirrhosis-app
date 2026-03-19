"""
Model Training Utilities for Liver Cirrhosis Prediction
"""
import os, sys, numpy as np, pandas as pd, pickle, warnings, joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, f1_score,
                              precision_score, recall_score)
from sklearn.impute import SimpleImputer

# ── Resolve root dir robustly ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, 'data', 'cirrhosis.csv')
MODEL_PATH = os.path.join(ROOT, 'models', 'model_bundle.pkl')

STAGE_LABELS = {
    1: "Stage 1 – Mild Cirrhosis",
    2: "Stage 2 – Moderate Cirrhosis",
    3: "Stage 3 – Severe Cirrhosis",
    4: "Stage 4 – End-Stage Cirrhosis"
}
STAGE_COLORS = {1: "#22c55e", 2: "#eab308", 3: "#f97316", 4: "#ef4444"}
RISK_LEVEL   = {1: "LOW RISK", 2: "MODERATE RISK", 3: "HIGH RISK", 4: "CRITICAL RISK"}


# ── Dataset ────────────────────────────────────────────────────────────────────
def load_data(path=None):
    path = path or DATA_PATH
    if not os.path.exists(path):
        _generate_dataset(path)
    return pd.read_csv(path)


def _generate_dataset(save_path):
    np.random.seed(42)
    N = 500
    stages = np.random.choice([1,2,3,4], N, p=[0.27,0.25,0.27,0.21])
    rows = []
    for stage in stages:
        s = stage
        if s == 1:
            bil=max(0.3,np.random.normal(0.9,0.3)); alb=max(2.0,np.random.normal(3.9,0.3))
            pro=max(8.0,np.random.normal(10.0,0.5)); cop=max(10,np.random.normal(60,20))
            sgo=max(20,np.random.normal(90,25));    alk=max(300,np.random.normal(900,200))
            plt=max(100,np.random.normal(320,60));   cho=max(100,np.random.normal(280,60))
            tri=max(50,np.random.normal(100,25));    nd=int(np.random.normal(3500,600))
            asc=np.random.choice(['N','Y'],p=[0.97,0.03])
            hep=np.random.choice(['N','Y'],p=[0.65,0.35])
            spi=np.random.choice(['N','Y'],p=[0.88,0.12])
            ede=np.random.choice(['N','S','Y'],p=[0.92,0.05,0.03])
        elif s == 2:
            bil=max(0.5,np.random.normal(1.8,0.6)); alb=max(2.0,np.random.normal(3.5,0.35))
            pro=max(9.0,np.random.normal(10.7,0.6)); cop=max(20,np.random.normal(95,30))
            sgo=max(30,np.random.normal(130,35));    alk=max(400,np.random.normal(1300,350))
            plt=max(80,np.random.normal(270,65));    cho=max(100,np.random.normal(350,90))
            tri=max(60,np.random.normal(130,35));    nd=int(np.random.normal(2600,700))
            asc=np.random.choice(['N','Y'],p=[0.90,0.10])
            hep=np.random.choice(['N','Y'],p=[0.50,0.50])
            spi=np.random.choice(['N','Y'],p=[0.75,0.25])
            ede=np.random.choice(['N','S','Y'],p=[0.82,0.12,0.06])
        elif s == 3:
            bil=max(0.8,np.random.normal(3.5,1.2)); alb=max(1.5,np.random.normal(3.1,0.4))
            pro=max(9.5,np.random.normal(11.5,0.8)); cop=max(40,np.random.normal(150,45))
            sgo=max(50,np.random.normal(180,50));    alk=max(600,np.random.normal(2000,600))
            plt=max(60,np.random.normal(210,70));    cho=max(80,np.random.normal(310,100))
            tri=max(60,np.random.normal(155,45));    nd=int(np.random.normal(1800,700))
            asc=np.random.choice(['N','Y'],p=[0.72,0.28])
            hep=np.random.choice(['N','Y'],p=[0.35,0.65])
            spi=np.random.choice(['N','Y'],p=[0.55,0.45])
            ede=np.random.choice(['N','S','Y'],p=[0.60,0.22,0.18])
        else:
            bil=max(2.0,np.random.normal(7.5,3.0)); alb=max(1.0,np.random.normal(2.6,0.45))
            pro=max(10.0,np.random.normal(12.8,1.2)); cop=max(60,np.random.normal(210,70))
            sgo=max(60,np.random.normal(240,70));    alk=max(800,np.random.normal(3200,900))
            plt=max(40,np.random.normal(160,75));    cho=max(60,np.random.normal(260,120))
            tri=max(50,np.random.normal(170,60));    nd=int(np.random.normal(900,500))
            asc=np.random.choice(['N','Y'],p=[0.42,0.58])
            hep=np.random.choice(['N','Y'],p=[0.22,0.78])
            spi=np.random.choice(['N','Y'],p=[0.35,0.65])
            ede=np.random.choice(['N','S','Y'],p=[0.30,0.25,0.45])
        rows.append({
            'ID':len(rows)+1, 'N_Days':max(41,nd),
            'Status':np.random.choice(['C','CL','D'],p=[0.56,0.12,0.32]),
            'Drug':np.random.choice(['D-penicillamine','Placebo',None],p=[0.45,0.45,0.10]),
            'Age':int(np.random.uniform(9598,28650)),
            'Sex':np.random.choice(['F','M'],p=[0.88,0.12]),
            'Ascites':asc,'Hepatomegaly':hep,'Spiders':spi,'Edema':ede,
            'Bilirubin':round(bil,2),
            'Cholesterol': None if np.random.rand()<0.05 else round(cho,0),
            'Albumin':round(alb,2),
            'Copper': None if np.random.rand()<0.05 else round(cop,0),
            'Alk_Phos':round(alk,1),'SGOT':round(sgo,2),
            'Tryglicerides': None if np.random.rand()<0.04 else round(tri,0),
            'Platelets': None if np.random.rand()<0.04 else round(plt,0),
            'Prothrombin':round(pro,1),'Stage':stage,
        })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"Dataset generated at {save_path}")


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    df['Age'] = df['Age'] / 365.25
    cat_map = {
        'Drug':       {'D-penicillamine': 1, 'Placebo': 0},
        'Sex':        {'F': 0, 'M': 1},
        'Ascites':    {'N': 0, 'Y': 1},
        'Hepatomegaly':{'N':0,'Y':1},
        'Spiders':    {'N': 0, 'Y': 1},
        'Edema':      {'N': 0, 'S': 1, 'Y': 2},
        'Status':     {'C': 0, 'CL': 1, 'D': 2},
    }
    for col, mapping in cat_map.items():
        if col in df.columns:
            df[f'{col}_enc'] = df[col].map(mapping)
            df.drop(columns=[col], inplace=True)
    # Feature engineering
    df['Bilirubin_Albumin_ratio'] = df['Bilirubin'] / (df['Albumin'].replace(0, 1e-6))
    df['MELD_proxy'] = (3.78 * np.log(df['Bilirubin'].clip(lower=0.1)) +
                        11.2 * np.log(df['Prothrombin'].clip(lower=0.1)) - 5.81)
    df['Copper_log'] = np.log1p(df['Copper'].fillna(0))
    # Fill NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    df_proc = preprocess(df)
    TARGET = 'Stage'
    feature_cols = [c for c in df_proc.columns if c != TARGET]
    X = df_proc[feature_cols]
    y = df_proc[TARGET]
    mask = y.notna()
    X, y = X[mask], y[mask].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    imputer = SimpleImputer(strategy='median')
    Xtr = imputer.fit_transform(X_tr)
    Xte = imputer.transform(X_te)

    rf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
        class_weight='balanced', n_jobs=-1, random_state=42, oob_score=True)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
        max_depth=5, subsample=0.8, random_state=42)

    rf.fit(Xtr, y_tr)
    gb.fit(Xtr, y_tr)

    ensemble = VotingClassifier(
        estimators=[('rf',rf),('gb',gb)], voting='soft', weights=[2,1])
    ensemble.fit(Xtr, y_tr)

    y_pred = ensemble.predict(Xte)
    y_prob = ensemble.predict_proba(Xte)

    try:
        auc = roc_auc_score(y_te, y_prob, multi_class='ovr', average='weighted')
    except:
        auc = 0.0

    cv = cross_val_score(rf, Xtr, y_tr, cv=5, scoring='accuracy')

    metrics = {
        'accuracy':  round(accuracy_score(y_te, y_pred)*100, 2),
        'f1_score':  round(f1_score(y_te, y_pred, average='weighted')*100, 2),
        'precision': round(precision_score(y_te, y_pred, average='weighted', zero_division=0)*100, 2),
        'recall':    round(recall_score(y_te, y_pred, average='weighted')*100, 2),
        'auc_roc':   round(auc*100, 2),
        'cv_mean':   round(cv.mean()*100, 2),
        'cv_std':    round(cv.std()*100, 2),
        'oob_score': round(rf.oob_score_*100, 2),
        'train_size': len(X_tr),
        'test_size':  len(X_te),
    }

    fi = pd.DataFrame({'feature': feature_cols,
                       'importance': rf.feature_importances_}
                     ).sort_values('importance', ascending=False).reset_index(drop=True)
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bundle = {
        'ensemble': ensemble, 'rf': rf, 'imputer': imputer,
        'feature_cols': feature_cols, 'metrics': metrics,
        'feature_importance': fi, 'confusion_matrix': cm,
        'classification_report': report,
        'y_test': y_te, 'y_pred': y_pred, 'y_prob': y_prob,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}  |  Accuracy: {metrics['accuracy']}%")
    return bundle


def load_model():
    return joblib.load(MODEL_PATH)


def model_exists():
    return os.path.exists(MODEL_PATH)


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_single(model_bundle, input_dict: dict):
    ensemble    = model_bundle['ensemble']
    imputer     = model_bundle['imputer']
    feature_cols= model_bundle['feature_cols']

    row = pd.DataFrame([input_dict])
    cat_map = {
        'Drug':       {'D-penicillamine':1,'Placebo':0},
        'Sex':        {'F':0,'M':1},
        'Ascites':    {'N':0,'Y':1},
        'Hepatomegaly':{'N':0,'Y':1},
        'Spiders':    {'N':0,'Y':1},
        'Edema':      {'N':0,'S':1,'Y':2},
        'Status':     {'C':0,'CL':1,'D':2},
    }
    for col, mapping in cat_map.items():
        if col in row.columns:
            row[f'{col}_enc'] = row[col].map(mapping)
            row.drop(columns=[col], inplace=True)

    bil = float(row.get('Bilirubin', pd.Series([1.0]))[0])
    alb = float(row.get('Albumin',   pd.Series([3.5]))[0])
    pro = float(row.get('Prothrombin', pd.Series([10.7]))[0])
    cop = float(row.get('Copper',    pd.Series([70]))[0])

    row['Bilirubin_Albumin_ratio'] = bil / max(alb, 1e-6)
    row['MELD_proxy'] = 3.78*np.log(max(bil,0.1)) + 11.2*np.log(max(pro,0.1)) - 5.81
    row['Copper_log'] = np.log1p(cop)

    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols]
    row_imp = imputer.transform(row)

    pred  = int(ensemble.predict(row_imp)[0])
    proba = {int(c): float(p)
             for c, p in zip(ensemble.classes_, ensemble.predict_proba(row_imp)[0])}
    return pred, proba
