"""
churn_model.py
---------------
Müşteri Churn (Kayıp) Tahmin Modeli — uçtan uca pipeline.

Adımlar:
  1. Veri yükleme ve temizleme
  2. Keşifsel veri analizi (EDA) görselleri
  3. Feature engineering + preprocessing (ColumnTransformer)
  4. 3 model: Logistic Regression, Random Forest, Gradient Boosting
  5. Değerlendirme: ROC-AUC, Gini, PSI, classification report, confusion matrix
  6. Feature importance ve ROC eğrisi görselleri
  7. En iyi modelin diske kaydı (joblib)

Çalıştırma:
    python src/churn_model.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "telco_churn.csv"
OUT = "."
RANDOM_STATE = 42
os.makedirs(OUT, exist_ok=True)


# ───────────────────────── 1. VERİ YÜKLEME & TEMİZLEME ─────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges sayısala çevrilir, boşlar tenure*MonthlyCharges ile doldurulur
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = (df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"])

    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


# ───────────────────────── 2. EDA GÖRSELLERİ ──────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    df["Churn"].map({0: "Kaldi", 1: "Churn"}).value_counts().plot(
        kind="bar", ax=axes[0], color=["#2563EB", "#DC2626"])
    axes[0].set_title("Churn Dagilimi")
    axes[0].set_ylabel("Musteri")
    axes[0].tick_params(axis="x", rotation=0)

    df.groupby("Contract")["Churn"].mean().sort_values().plot(
        kind="barh", ax=axes[1], color="#2563EB")
    axes[1].set_title("Sozlesme Tipine Gore Churn Orani")
    axes[1].set_xlabel("Churn Orani")

    for label, color in [(0, "#2563EB"), (1, "#DC2626")]:
        df[df["Churn"] == label]["tenure"].plot(
            kind="hist", bins=30, alpha=0.6, ax=axes[2], color=color,
            label="Churn" if label else "Kaldi")
    axes[2].set_title("Tenure (Ay) Dagilimi")
    axes[2].set_xlabel("Tenure")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{OUT}/eda.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [+] EDA gorseli -> {OUT}/eda.png")


# ───────────────────────── 3. PREPROCESSING ───────────────────────────────
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


# ───────────────────────── 4. METRİK YARDIMCILARI ─────────────────────────
def gini(auc: float) -> float:
    """Gini = 2*AUC - 1 (bankacilikta yaygin kullanilan ayirici guc metrigi)."""
    return 2 * auc - 1


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — skor dagilim kaymasini olcer (<0.1 stabil)."""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e = np.histogram(expected, breakpoints)[0] / len(expected)
    a = np.histogram(actual, breakpoints)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ───────────────────────── 5. EĞİTİM & DEĞERLENDİRME ──────────────────────
def main() -> None:
    print("[1/5] Veri yukleniyor ve temizleniyor...")
    df = load_and_clean(DATA_PATH)
    print(f"      Satir: {len(df)} | Churn orani: {df['Churn'].mean():.1%}")

    print("[2/5] EDA gorselleri uretiliyor...")
    run_eda(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    pre = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    print("[3/5] Modeller egitiliyor...")
    results, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_te, proba)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "F1": f1_score(y_te, pred),
            "ROC-AUC": auc,
            "Gini": gini(auc),
            "PSI": psi(pipe.predict_proba(X_tr)[:, 1], proba),
        })
        fitted[name] = (pipe, proba)
        print(f"      {name:22s} AUC={auc:.3f}  Gini={gini(auc):.3f}")

    res_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    res_df.to_csv(f"{OUT}/metrics.csv", index=False)

    print("\n[4/5] Model karsilastirma tablosu:")
    print(res_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best_name = res_df.iloc[0]["Model"]
    best_pipe, best_proba = fitted[best_name]
    best_pred = (best_proba >= 0.5).astype(int)

    print(f"\n      En iyi model: {best_name}")
    print("\n", classification_report(y_te, best_pred, target_names=["Kaldi", "Churn"]))

    # ROC eğrileri
    plt.figure(figsize=(7, 6))
    for name, (_, proba) in fitted.items():
        fpr, tpr, _ = roc_curve(y_te, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Egrileri"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(f"{OUT}/roc_curves.png", dpi=120); plt.close()
    print(f"  [+] ROC egrileri -> {OUT}/roc_curves.png")

    # Confusion matrix (en iyi model)
    cm = confusion_matrix(y_te, best_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black", fontsize=14)
    plt.xticks([0, 1], ["Kaldi", "Churn"]); plt.yticks([0, 1], ["Kaldi", "Churn"])
    plt.xlabel("Tahmin"); plt.ylabel("Gercek")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrix.png", dpi=120); plt.close()
    print(f"  [+] Confusion matrix -> {OUT}/confusion_matrix.png")

    # Feature importance (tree tabanli modeller icin)
    try:
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            names = best_pipe.named_steps["pre"].get_feature_names_out()
            imp = pd.Series(clf.feature_importances_, index=names).sort_values()[-15:]
            plt.figure(figsize=(8, 6))
            imp.plot(kind="barh", color="#2563EB")
            plt.title(f"En Onemli 15 Degisken — {best_name}")
            plt.tight_layout(); plt.savefig(f"{OUT}/feature_importance.png", dpi=120); plt.close()
            print(f"  [+] Feature importance -> {OUT}/feature_importance.png")
    except Exception as e:
        print("  [!] Feature importance atlandi:", e)

    print("[5/5] En iyi model kaydediliyor...")
    joblib.dump(best_pipe, f"{OUT}/best_model.joblib")
    print(f"  [+] Model -> {OUT}/best_model.joblib")
    print("\nTAMAMLANDI. Tum ciktilar ana dizinde olusturuldu.")


if __name__ == "__main__":
    main()
