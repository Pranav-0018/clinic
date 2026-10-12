# Clinical Document Classification & STG Compliance Extraction System
## PBL 1 Team Presentation, Viva Defense & Rubric Guide

---

### Team Split & Speaker Scripts (4 Members)

#### Member 1: Problem, Data & Features (Notebook Sections 1–7)
- **Problem Statement:** Hospital documents (prescriptions, lab reports, discharge summaries) arrive as scanned, skewed, blurred, mobile-camera photos with varied layouts. Manual entry into hospital EHR is slow and vulnerable to undetected prescription errors.
- **Data Reality & Justification:** Public benchmark datasets (**CORD-v2**, **SROIE**, **RVL-CDIP**, **IAM**) serve as bootstrap stand-ins to validate computer vision features and Unit 2 machine learning algorithms. Real clinical corpora (**MIMIC-III**, **n2c2**) are strictly gated behind legal Data Use Agreements (DUAs).
- **The 9 Extracted Document Features:**
  1. `mean_intensity`: Average grayscale brightness.
  2. `std_intensity`: Global pixel variance / contrast.
  3. `edge_density`: Ratio of Sobel gradient edge pixels to total area.
  4. `laplacian_var`: Sharpness / camera blur threshold metric.
  5. `entropy`: Shannon entropy $-\sum p \log_2 p$.
  6. `hf_ratio`: 2D FFT high-frequency spectral power ratio.
  7. `sat_std`: HSV color saturation variation (identifies hospital seals and colored stamps).
  8. `aspect_ratio`: Width / Height ratio.
  9. `n_line_items`: Count of text lines detected via morphological dilation.
- **Hand-off line to Member 2:** *"Now that we have extracted real numerical features from our document corpus, how do we estimate parameters and fit models from them?"*

---

#### Member 2: MLE & Linear Regression (Notebook Sections 8–14)
- **Maximum Likelihood Estimation (MLE):**
  - Gaussian Likelihood: $L(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$
  - Parameter solutions: $\mu_{\text{MLE}} = \frac{1}{n}\sum x_i$ and $\sigma^2_{\text{MLE}} = \frac{1}{n}\sum (x_i - \mu)^2$.
  - **The Variance Bias Proof:** $\sigma^2_{\text{MLE}}$ is mathematically biased because the true mean $\mu$ is replaced by the sample estimate $\bar{x}$, consuming 1 degree of freedom:
    $$\mathbb{E}[\sigma^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$$
    Multiplying by $\frac{n}{n-1}$ recovers the unbiased Bessel-corrected sample variance $s^2$.
- **Linear Regression & Normal Equation:**
  - Solves the Ordinary Least Squares (OLS) closed-form solution: $\hat{\beta} = (X^T X)^{-1} X^T y$.
- **Why Linear Regression Fails for Classification (Bridge to Member 3):**
  1. Predictions $\hat{y}$ are unbounded and fall outside $[0, 1]$.
  2. Extreme outliers distort the decision hyperplane.
  3. Intermediate classes suffer from severe masking in multiclass settings ($K > 2$).
- **Hand-off line to Member 3:** *"Because linear regression fails for discrete classification, how do we use proper classification algorithms to categorize documents?"*

---

#### Member 3: Classification Models (Notebook Sections 15–25, 31–32)
- **Logistic Regression:**
  - Sigmoid mapping $\sigma(z) = \frac{1}{1 + e^{-z}}$ outputs bounded probabilities $P(Y=k \mid \mathbf{x})$.
  - Log-odds linearity: $\ln\left(\frac{p}{1-p}\right) = \mathbf{w}^T \mathbf{x} + b$.
  - Generates well-calibrated probabilities for clinical risk routing.
- **Linear Discriminant Analysis (LDA):**
  - Assumes class-conditional Gaussian distributions with shared covariance matrix $\Sigma$.
  - Uses Bayes rule and linear discriminant score: $\delta_k(\mathbf{x}) = \mathbf{x}^T \Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T \Sigma^{-1}\mu_k + \ln \pi_k$.
- **Decision Tree (CART):** Interpretable orthogonal splits on document feature thresholds.
- **Live Demonstration:**
  - Shows prediction on a held-out test scan in the interactive UI, displaying calibrated probability scores and routing to the Priority Verification Queue.
- **Hand-off line to Member 4:** *"How do these classifiers compare across formal evaluation metrics, and how do we enforce clinical safety guidelines?"*

---

#### Member 4: Comparative Evaluation, STG & Limitations (Notebook Sections 26–36)
- **Comparative Evaluation Benchmark:**
  - **Logistic Regression:** 91.04% Accuracy, 0.9109 Macro F1, 0.9902 ROC-AUC, **0.0100 ECE** (Passed poster target $< 0.05$).
  - **LDA:** 90.08% Accuracy, 0.9012 Macro F1, 0.9880 ROC-AUC, 0.0333 ECE.
  - **Decision Tree:** 84.00% Accuracy, 0.8401 Macro F1.
  - **Random Forest:** 89.92% Accuracy, 0.8995 Macro F1.
- **Standard Treatment Guidelines (STG) Rule Engine:**
  - Integrates official **ICMR 2023** and **MoHFW** protocols.
  - Flags supratherapeutic overdoses (Metformin $> 2550\text{ mg/day}$).
  - Detects fatal contraindications (Telmisartan + Enalapril dual RAS blockade).
  - Flags inappropriate empiric antibiotics (Ciprofloxacin in simple watery diarrhea).
- **Honest Limitations (State Before Reviewers Ask):**
  1. Bootstrapped from public receipt/business corpora (CORD, RVL-CDIP); clinical deployment requires hospital-approved MIMIC-III access.
  2. Features are whole-image statistics; production hospital pipelines recommend multimodal transformers (LayoutLMv3) for dense medical text.
  3. Target metrics (90% Acc, 8% CER) are engineering design targets, not claims of hospital deployment.
  4. Symbolic STG rule base acts as safety guardrails before EHR export.

---

### Quick Viva Q&A Cheat Sheet

* **Q: Why is the variance estimated by Gaussian MLE biased?**
  * *A:* Because the true population mean $\mu$ is unknown and substituted with the sample mean $\bar{x}$. Since the sample points are closer to $\bar{x}$ than to the true $\mu$, $\sum (x_i - \bar{x})^2$ systematically underestimates the variance by $\frac{n-1}{n}$.

* **Q: Why does OLS linear regression fail for multiclass document classification?**
  * *A:* OLS treats discrete categorical classes as continuous ordered values, outputs probabilities $< 0$ and $> 1$, and causes intermediate classes to be completely masked by extreme classes.

* **Q: What is Expected Calibration Error (ECE) and why does it matter?**
  * *A:* ECE measures whether predicted confidence matches true empirical accuracy across probability bins ($[0.8, 0.9]$, $[0.9, 1.0]$). In healthcare, overconfident models are dangerous; an ECE of $0.0100$ ($< 0.05$) guarantees that a 90% confidence prediction is correct 90% of the time.

* **Q: Why are MIMIC-III and n2c2 not trained directly in the notebook?**
  * *A:* They are protected health information (PHI) requiring institutional review, CITI human subjects training, and signed DUAs. We validate the mathematical architecture on open datasets first.
