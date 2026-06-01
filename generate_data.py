"""
generate_data.py
-----------------
Gerçek Telco Customer Churn veri setinin dinamiklerini taklit eden,
tekrarlanabilir (seed'li) sentetik bir müşteri churn veri seti üretir.

Not: Orijinal IBM Telco veri seti internet erişimi olan ortamlarda
Kaggle'dan indirilebilir. Bu script, ağ erişimi olmadan da projenin
uçtan uca çalışabilmesi için gerçekçi bir veri seti üretir. Değişkenler
ve churn ilişkileri, literatürdeki Telco churn davranışına göre kurgulanmıştır.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 7043  # orijinal Telco veri seti ile aynı satır sayısı


def generate(n: int = N) -> pd.DataFrame:
    gender = RNG.choice(["Male", "Female"], n)
    senior = RNG.choice([0, 1], n, p=[0.84, 0.16])
    partner = RNG.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = RNG.choice(["Yes", "No"], n, p=[0.30, 0.70])

    tenure = RNG.integers(0, 73, n)  # ay

    contract = RNG.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    paperless = RNG.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = RNG.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21],
    )

    phone = RNG.choice(["Yes", "No"], n, p=[0.90, 0.10])
    internet = RNG.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    online_security = RNG.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22])
    tech_support = RNG.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22])
    streaming_tv = RNG.choice(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.22])

    # Aylık ücret: internet tipine bağlı
    base = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 50, 20))
    monthly = base + RNG.normal(0, 12, n)
    monthly = np.clip(monthly, 18, 120).round(2)

    total = (monthly * np.maximum(tenure, 0.5) * RNG.uniform(0.9, 1.05, n)).round(2)

    # ---- Churn olasılığı: gerçekçi risk faktörleri ----
    logit = (
        -1.6
        + 1.4 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        + 0.8 * (internet == "Fiber optic")
        + 0.7 * (payment == "Electronic check")
        - 0.025 * tenure                      # uzun süre kalan az churn eder
        + 0.5 * (online_security == "No")
        + 0.5 * (tech_support == "No")
        + 0.4 * senior
        - 0.3 * (partner == "Yes")
        + 0.010 * (monthly - 65)              # pahalı paket -> daha çok churn
        + RNG.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (RNG.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "customerID": [f"{RNG.integers(1000,9999)}-{''.join(RNG.choice(list('ABCDEFGHIJ'),4))}" for _ in range(n)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return df


if __name__ == "__main__":
    df = generate()
    out = "telco_churn.csv"
    df.to_csv(out, index=False)
    rate = (df["Churn"] == "Yes").mean()
    print(f"Veri seti olusturuldu: {out}")
    print(f"Satir: {len(df)} | Sutun: {df.shape[1]} | Churn orani: {rate:.1%}")
