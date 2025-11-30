# MTGA Draft Helper — Model Training & Evaluation Report Specification

This document defines the required metrics, plots, and diagnostics for evaluating
the Skill Model (M1), Joint Model (M2), and Deck Bump Decomposition (M3).
Use this as the authoritative reference when generating production‑grade evaluation
reports in VS Code or CI.

---

## 1. Core Metrics

### 1.1 Skill‑Only Model (M1)
* **R²_skill**
* **RMSE_skill**
* **MAE_skill**
* **Residual distribution summary**
* **Cross‑validated R²_skill** (5‑fold recommended)

---

### 1.2 Joint Deck + Skill Model (M2)
* **R²_joint**
* **RMSE_joint**
* **MAE_joint**
* **Cross‑validated R²_joint**
* **ΔR²_deck = R²_joint − R²_skill**

---

### 1.3 Deck Boost Decomposition (M3)
For each deck:
* `skill_pred`
* `joint_pred`
* `deck_boost = joint_pred − skill_pred`

Report:
* Mean, variance, and distribution histogram of deck_boost  
* Percentage of decks with positive deck_boost  

---

## 2. Calibration Plots (Required)

### 2.1 Net Effect Calibration
**Predicted run win rate (joint_pred) vs observed run WR**.

Plot must include:
* 20 equal‑sized bins of predicted WR  
* Bin means: predicted vs observed  
* 1:1 reference line  
* Optional isotonic regression calibration line  
* RMSE of calibration residuals  

---

### 2.2 Deck Bump Calibration
**Predicted deck_boost vs observed Δp**  
(where Δp = wins/(wins+losses) − base_p, or posterior mean p − base_p).

Plot must include:
* 20 bins by deck_boost  
* Predicted Δp vs observed Δp  
* 1:1 line  
* Weighted linear fit with slope, intercept, R², RMSE  
* Weights = number of games per run  

---

## 3. Error Analysis

### 3.1 Residual Diagnostics
Plot residuals of the joint model vs:
* skill_bucket  
* main_colors  
* splash_colors  
* number of games  
* deck_size  
* build_index  
* run date or draft_id (optional temporal check)

### 3.2 Residual Distribution
* Histogram  
* QQ‑plot vs normal  
* Boxplots per skill bucket  

---

## 4. Model Explainability

### 4.1 Feature Importance (Joint Model)
Report:
* XGBoost gain importance  
* Optional permutation importance  
* Top 30 deck_* features  
* Top skill features  

---

### 4.2 SHAP (Optional)
Produce:
* SHAP summary plot (top 20 features)  
* Separate deck_*‑only SHAP importance list  

---

## 5. Train/Validation Diagnostics

### 5.1 Train/Valid Metrics
* Train vs validation R²  
* Overfitting gap  
* Learning curve: R² vs #training samples  

---

## 6. Cross‑Validation Results

Perform 5‑fold CV:  
* R²_skill across folds  
* R²_joint across folds  
* Mean ± SD  
* Boxplot of fold‑level R²  

---

## 7. Deck Boost Distribution

Report distribution of deck_boost:
* Histogram  
* Empirical CDF  
* Summary: median, IQR, 90th, 95th percentile  

---

## 8. Correlation Structure (Optional)

Provide:
* Correlation matrix of top 50 features  
* Correlation between skill_pred and deck_boost  
* Correlation between joint_pred and skill_pred  

---

## 9. Deliverables for the Final Report

A full model‑evaluation notebook or script must output:

1. Metrics table (M1, M2, M3)
2. All calibration plots
3. Residual diagnostics
4. Feature importance plots
5. SHAP summary (optional)
6. Learning curves
7. Cross‑validation R² figures
8. Deck boost distribution summary & plots

This document serves as the official reference for generating production‑quality model evaluation reports in the MTGA Draft Helper project.
