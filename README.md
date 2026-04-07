#  Diabetic Foot Ulcer Detection & Severity Assessment
> An end-to-end deep learning pipeline for DFU detection, severity classification, and clinical assessment.

---

##  Overview
This project builds a **3-stage pipeline** that:
1. Detects whether an image contains a diabetic foot ulcer
2. Classifies severity as **Mild / Moderate / Severe**
3. Provides **clinical curability guidance**

Three models are trained and compared — **EfficientNetB0**, **MobileNetV2**, and **ResNet50**.

---

##  Pipeline Flow
```
Input Image
     │
     ▼
[Stage 1] Binary Detection (EfficientNetB0)
     │
     ├── No Ulcer →  Pipeline Stops
     │
     └── Ulcer Detected
          │
          ▼
     [Stage 2] Severity Classification
     ┌──────────────┬──────────────┬──────────────┐
     │EfficientNetB0│  MobileNetV2 │   ResNet50   │
     └──────────────┴──────────────┴──────────────┘
          │ Best model prediction used
          ▼
     [Stage 3] Curability Assessment (Rule-based)
     🟢 Mild → Routine Care
     🟡 Moderate → See Doctor 48hrs
     🔴 Severe → Emergency NOW
          │
          ▼
     Final Report + Probability Chart
```

---

##  Severity Labeling (No Ground-Truth Labels Available)

Since EfficientDFU has no severity labels, they were generated using:
```
512 Ulcer Images
      │
      ▼
EfficientNetB0 Feature Extraction (1280-dim vectors)
      │
      ▼
KMeans Clustering (3 clusters)
      │
      ▼
Brightness Anchoring:
  Brightest → 🟢 Mild
  Middle    → 🟡 Moderate
  Darkest   → 🔴 Severe
```
> Medical reasoning: severe ulcers (necrotic tissue) appear darker; mild ulcers (surface redness) appear brighter.

---

##  Model Architecture

All models share the same custom head:
```
Frozen Pretrained Backbone
         │
GlobalAveragePooling2D
         │
     Dropout(0.4)
         │
    Dense(128, ReLU)
         │
     Dropout(0.2)
         │
Dense(3, Softmax) or Dense(1, Sigmoid)
```

| Model | Strength | Preprocessing |
|---|---|---|
| EfficientNetB0 | High accuracy | `eff_preprocess` → `[-1,1]` |
| MobileNetV2 | Lightweight | `mob_preprocess` → `[-1,1]` |
| ResNet50 | Deep features | `res_preprocess` → channel mean subtraction |

>  Each model uses its **own** preprocessing function — not generic `rescale=1/255`.

---

## ⚙️ Training Config

| Parameter | Value |
|---|---|
| Input Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | Adam (lr=1e-4) |
| Max Epochs | 20 + EarlyStopping |
| Val Split | 20% |
| Augmentation | Flip, Rotation±15°, Zoom±10%, Brightness±20% |

---

##  Dataset

**EfficientDFU** (Kaggle)

| Folder | Description | Count |
|---|---|---|
| `Patches/Abnormal(Ulcer)` | Ulcer patches | 512 |
| `Patches/Normal(Healthy skin)` | Healthy skin | Varies |
| `TestSet` | Held-out test images | Varies |

---

##  How to Run

1. Open in **Google Colab** with GPU enabled
2. Upload `dfu_dataset.zip` when prompted
3. Run all sections top to bottom
4. Upload your own test image in Section 15

---

## 🏥 Curability Assessment

| Severity | Action | Timeline |
|---|---|---|
|  Mild | Standard wound care + offloading | 4–8 weeks |
|  Moderate | Medical evaluation, possible debridement | 8–16 weeks |
|  Severe | Emergency — vascular/surgical review | Immediate |

---

##  Output Files

| File | Description |
|---|---|
| `binary_ulcer_model.h5` | Stage 1 detection model |
| `severity_*.h5` | Stage 2 severity models (×3) |
| `severity_samples.png` | Sample images per class |
| `training_curves.png` | Accuracy & loss plots |
| `confusion_*.png` | Confusion matrices |
| `model_comparison.png` | Accuracy bar chart |
| `prediction_result.png` | Last inference output |

---

##  Limitations

- Severity labels are **unsupervised proxies**, not clinician-verified
- Small dataset (512 ulcer patches)
- Not externally validated
- Not a medical device — always consult a qualified professional

---

##  References

- Popa et al. (2023) — Survival Prediction in DFUs using ML. *J. Clin. Med.*
- aan de Stegge et al. (2021) — Foot Ulcer Recurrence Prediction. *BMJ Open*
- Spinazzola et al. (2025) — Chronic Ulcer Healing Prediction via ML. *J. Clin. Med.*
