# Research Questions and Theoretical Gaps

## 1. Introduction

This document outlines the motivation for the project and the gap between empirical success and theoretical understanding. The system achieves reasonable directional accuracy and backtest metrics (e.g. ~73% directional accuracy, Sharpe ~1.5–2.0) using an over-parameterized LSTM trained with SGD on a relatively small financial time series. Why such a setup generalizes at all, and how to improve it theoretically, are open research questions that connect optimization theory, learning theory, and financial econometrics.

**Why this project exists.** Machine learning is increasingly used in quantitative finance for return prediction, risk modeling, and execution. In practice, practitioners tune architectures and hyperparameters empirically. There is limited theory that explains when and why these models generalize, how to choose loss functions and regularizers, and how to account for non-stationarity and regime change. This project implements a working ML trading pipeline and explicitly highlights where theory is missing or incomplete.

**Gap between empirical success and theory.** We observe that (1) SGD with a fixed learning rate and momentum often converges to a good solution; (2) an LSTM with hundreds of thousands of parameters trained on thousands of samples does not always overfit badly; (3) backtest performance is sensitive to transaction costs and data period. These observations raise questions about optimization landscape, implicit regularization, sample complexity, and the appropriateness of standard loss functions (e.g. MSE or BCE) for financial outcomes. The following sections formalize these into research questions and point to relevant literature and potential directions for professors (e.g. Zou, Li, Cao) working in optimization, econometrics, and learning theory.

---

## 2. Optimization Theory Questions

### 2.1 Why does SGD converge with the chosen hyperparameters?

We use SGD with momentum (e.g. 0.9), learning rate 0.001, and cosine annealing. Empirically, training loss decreases and validation loss reaches a reasonable minimum. However:

- **Convergence guarantees:** Most classical results assume convex objectives or specific smoothness conditions. The LSTM objective is non-convex and non-smooth (e.g. due to ReLU and sigmoid). Under what conditions can we prove convergence (e.g. to a stationary point) for this architecture?

- **Learning rate:** The learning rate is chosen by grid search. Can optimization theory provide guidance—e.g. in terms of Lipschitz constants or Hessian spectrum—for a theoretically motivated learning rate schedule? This connects to recent work on learning rate schedules and warm restarts.

- **Momentum:** How does momentum change the effective dynamics and the set of minima that the optimizer can reach? Is there a “better” momentum schedule for RNNs/LSTMs?

**Relevant directions:** Analysis of SGD in non-convex settings; role of learning rate and batch size; implicit bias of gradient-based methods. References: Li et al. on convergence of SGD for over-parameterized networks; Zou and related work on optimization landscape.

### 2.2 What is the optimization landscape?

- **Loss surface:** Is the loss surface relatively benign (e.g. many good minima) or are we likely in a sharp minimum? How does the choice of architecture (depth, width, dropout) affect the geometry?

- **Implicit regularization:** SGD is known to impose an implicit regularization (e.g. toward solutions with certain norm or rank). For sequence models trained on financial data, what is this implicit bias and does it align with “simple” or “robust” predictors?

**Research proposal (high level):** Characterize the loss landscape of the LSTM used in this project (e.g. via Hessian spectrum, loss curvature along directions of interest) and relate it to generalization and stability of backtest performance.

---

## 3. Generalization Theory Questions

### 3.1 Why does an over-parameterized model generalize?

The LSTM has on the order of 500k parameters and is trained on sequences derived from roughly 1k–10k effective samples (after sequence construction and train/val split). Classical VC-style bounds would suggest high overfitting risk. Yet we observe non-trivial validation accuracy.

- **Benign overfitting:** In linear and kernel settings, “benign overfitting” describes regimes where models interpolate the training data but still generalize. Is there an analogous phenomenon for RNNs/LSTMs on time series? Under what data distributions and architectures does overfitting remain benign?

- **Sample complexity:** Can we derive sample complexity bounds (e.g. number of sequences needed for a given excess risk) that reflect the temporal structure and the architecture? Such bounds would inform how much data is “enough” for a given market regime.

- **Role of architecture:** How do depth, width, and recurrence interact with generalization? For example, does dropout or weight decay provide a provable regularizing effect in this setting?

**Relevant directions:** Modern theory of generalization in deep learning; Rademacher/generative complexity for RNNs; connection to stability and flat minima. References: Cao and collaborators on learning theory and generalization.

### 3.2 Temporal and distribution shift

Financial data are non-stationary. Train/val/test splits are temporal, so test data are “future” relative to training. This raises:

- **Distribution shift:** Can we quantify or bound the shift between train and test distributions (e.g. in terms of returns or volatility)? How should we adapt the model or the evaluation protocol when shift is large?

- **Regime change:** Should the model explicitly account for regimes (e.g. high vs low volatility) and, if so, how can we learn or detect regimes with theoretical guarantees?

---

## 4. Financial Econometrics Questions

### 4.1 Why BCE loss for financial prediction?

We predict direction (up/down) and use binary cross-entropy. Alternatives include:

- **MSE on returns:** Direct prediction of next-period return. This penalizes large errors more and may be more aligned with portfolio construction (e.g. mean–variance).

- **Quantile regression:** For tail risk and VaR, quantile loss might be more appropriate. Can we combine direction and quantile objectives in a principled way?

- **Economic loss:** A loss that directly reflects transaction costs, drawdown, or Sharpe ratio would be closer to the actual objective. How to optimize such objectives (often non-differentiable or non-smooth) and how do they relate to BCE/MSE?

**Research proposal:** Compare BCE, MSE, and a simple economic loss (e.g. negative Sharpe on a validation backtest) in terms of out-of-sample backtest metrics and stability. Derive or approximate gradients for the economic loss and train the LSTM with it.

### 4.2 Volatility and GARCH

We use historical volatility and ATR as features. Volatility is time-varying and often modeled with GARCH-type processes.

- **GARCH vs ML:** Can we integrate a GARCH (or similar) component into the feature set or the model (e.g. as a latent state) to capture volatility clustering? What is the gain in prediction or risk control?

- **Uncertainty quantification:** Can we attach confidence intervals or predictive distributions to the LSTM output? This would require either Bayesian methods, dropout at test time, or calibration techniques.

### 4.3 Feature selection in high dimensions

We use 45 hand-crafted features. In higher dimensions (e.g. hundreds of features or alternative data), feature selection becomes critical.

- **Sparsity:** Can we encourage sparse feature use (e.g. L1 or group L1 on the first layer) and derive conditions under which the true predictive features are recovered?

- **Causal vs predictive:** Some features may be predictive only due to correlation, not causation. How can we combine causal inference (e.g. from econometrics) with ML to select robust features?

**Relevant directions:** Financial econometrics; volatility modeling; causal inference in time series. References: Li and collaborators on econometric methods and high-dimensional statistics.

---

## 5. Specific Research Proposals

### Proposal 1: Optimization landscape and learning rate

- **Motivation:** Our learning rate and schedule are chosen empirically. A theoretical or semi-theoretical guide would improve reproducibility and robustness.
- **Approach:** (1) Estimate local curvature (e.g. largest eigenvalues of the Hessian or Gauss–Newton matrix) on a subset of parameters. (2) Relate curvature to maximal stable learning rate and to generalization. (3) Propose a learning rate schedule that adapts to estimated curvature.
- **Expected outcome:** A practical rule or schedule for LSTM training that is motivated by optimization theory and validated on our (and possibly other) financial datasets. Connection to work by Zou and others on optimization.

### Proposal 2: Generalization bounds for LSTM on time series

- **Motivation:** We lack sample complexity bounds that account for both the LSTM architecture and the temporal nature of financial data.
- **Approach:** (1) Define a notion of “effective complexity” for the LSTM (e.g. path norm or Rademacher complexity). (2) Assume a mixing or stationarity condition on the data generating process. (3) Derive excess risk bounds in terms of sample size and complexity. (4) Check consistency with empirical train/val/test gaps.
- **Expected outcome:** A bound that explains, at least qualitatively, why our model does not overfit as much as naive parameter count would suggest. Connection to Cao and learning theory.

### Proposal 3: Economic loss and differentiable backtest

- **Motivation:** BCE is a proxy for “correct direction”; the real goal is risk-adjusted return or drawdown.
- **Approach:** (1) Implement a differentiable approximation of backtest PnL or Sharpe (e.g. soft thresholds, continuous position). (2) Train the LSTM with gradient ascent on this objective (or a combined loss). (3) Compare with BCE-trained model on out-of-sample backtest.
- **Expected outcome:** Evidence that optimizing an economic objective can improve backtest metrics and robustness. Connection to Li and financial econometrics (objective functions in portfolio choice).

### Proposal 4: Regime detection and model adaptation

- **Motivation:** Performance may degrade when the market regime shifts (e.g. from low to high volatility).
- **Approach:** (1) Use a simple regime indicator (e.g. rolling volatility or VIX proxy) to split data into regimes. (2) Train either regime-specific LSTMs or a single model with regime as input. (3) Evaluate on periods that include regime changes.
- **Expected outcome:** A method that improves robustness across regimes and a clearer understanding of when the model fails.

### Proposal 5: Uncertainty quantification and calibration

- **Motivation:** A probability output (e.g. 0.7 for “up”) is more useful if it is calibrated (e.g. 70% of such predictions are actually up).
- **Approach:** (1) Apply temperature scaling or Platt scaling to the LSTM output. (2) Report calibration curves and Brier score. (3) Optionally, use dropout at test time or ensembles to produce predictive intervals.
- **Expected outcome:** Calibrated probabilities that can be used for position sizing or risk management, with documented calibration quality.

---

## 6. References

### Optimization and deep learning

- Li, Q., et al. (various). Convergence of SGD and variants for over-parameterized neural networks.
- Zou, D., et al. (various). Optimization landscape and implicit regularization in deep learning.

### Generalization and learning theory

- Cao, Y., and related. Generalization bounds, Rademacher complexity, and benign overfitting.

### Financial econometrics and ML

- Campbell, J. Y., Lo, A. W., MacKinlay, A. C. The Econometrics of Financial Markets.
- Engle, R. GARCH and volatility modeling.
- High-dimensional and sparse methods in finance (e.g. factor models, variable selection).

### Practical ML for finance

- López de Prado, M. Advances in Financial Machine Learning.
- Papers on LSTM/RNN for asset returns and volatility (applied literature).

---

*This document is intended to demonstrate research-mindedness and to connect the implemented system to open theoretical questions. It can be tailored further to align with specific professors’ interests (e.g. Zou for optimization, Li for econometrics, Cao for learning theory).*
