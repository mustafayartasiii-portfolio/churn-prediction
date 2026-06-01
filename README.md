# 📊 Müşteri Churn (Kayıp) Tahmin Modeli

Telekom müşterilerinin **churn (hizmeti bırakma)** olasılığını tahmin eden uçtan uca bir makine öğrenmesi projesi. Veri temizleme, keşifsel analiz (EDA), feature engineering, model karşılaştırması ve bankacılık/finans sektöründe kullanılan ayırıcı güç metrikleriyle (**ROC-AUC, Gini, PSI**) değerlendirme içerir.

---

## 🎯 Problem
Bir telekom şirketinde müşteri kaybı (churn), gelir kaybının en büyük nedenlerinden biridir. Risk altındaki müşterileri önceden tespit etmek, hedefli tutundurma (retention) kampanyalarıyla maliyeti düşürür. Bu proje, müşteri özelliklerinden churn olasılığını tahmin eder.

## 📁 Proje Yapısı
```
churn-prediction/
├── data/
│   └── telco_churn.csv          # Veri seti (7.043 müşteri, 17 değişken)
├── src/
│   ├── generate_data.py         # Veri üretimi (tekrarlanabilir, seed=42)
│   └── churn_model.py           # Ana pipeline: temizleme → EDA → model → değerlendirme
├── outputs/
│   ├── eda.png                  # Keşifsel analiz görselleri
│   ├── roc_curves.png           # 3 modelin ROC eğrileri
│   ├── confusion_matrix.png     # En iyi modelin hata matrisi
│   ├── feature_importance.png   # En önemli 15 değişken
│   ├── metrics.csv              # Model karşılaştırma tablosu
│   └── best_model.joblib        # Kaydedilmiş en iyi model
├── requirements.txt
└── README.md
```

## ⚙️ Kurulum & Çalıştırma
```bash
pip install -r requirements.txt
python src/generate_data.py    # veri setini üretir
python src/churn_model.py       # modeli eğitir ve çıktıları üretir
```

## 🔬 Yöntem
1. **Temizleme:** `TotalCharges` sayısala çevrilir, eksik değerler `tenure × MonthlyCharges` ile doldurulur.
2. **EDA:** Churn dağılımı, sözleşme tipine göre churn oranı, tenure analizi.
3. **Preprocessing:** `ColumnTransformer` ile sayısal değişkenlerde `StandardScaler`, kategoriklerde `OneHotEncoder`.
4. **Modeller:** Logistic Regression, Random Forest, Gradient Boosting (dengesiz sınıf için `class_weight="balanced"`).
5. **Değerlendirme:** ROC-AUC, **Gini = 2×AUC−1**, **PSI** (skor stabilitesi), F1, confusion matrix.

## 📈 Sonuçlar

| Model | Accuracy | F1 | ROC-AUC | Gini | PSI |
|---|---|---|---|---|---|
| **Gradient Boosting** ⭐ | 0.739 | 0.540 | **0.783** | **0.566** | 0.005 |
| Logistic Regression | 0.698 | 0.614 | 0.782 | 0.563 | 0.005 |
| Random Forest | 0.729 | 0.572 | 0.772 | 0.543 | 0.285 |

**En iyi model: Gradient Boosting** — ROC-AUC 0.783, Gini 0.566. PSI değerinin 0.1'in altında olması skor dağılımının stabil olduğunu gösterir.

En önemli churn faktörleri: sözleşme tipi (month-to-month), tenure, internet servisi (fiber optik), ödeme yöntemi ve aylık ücret.

### Görseller
![ROC](outputs/roc_curves.png)
![EDA](outputs/eda.png)

## 🛠 Teknolojiler
`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `joblib`

## 📌 Not
Veri seti, ağ erişimi gerektirmeden projenin uçtan uca çalışabilmesi için gerçek Telco churn dinamikleri taklit edilerek üretilmiştir. Orijinal IBM Telco Customer Churn veri seti ile aynı şema (7.043 satır, 17 değişken) ve benzer churn davranışını taşır.

---
**Mustafa Yartaş** · [mustafaayartasii@gmail.com](mailto:mustafaayartasii@gmail.com)
