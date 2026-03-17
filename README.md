# 📉 Understanding Double Descent in Machine Learning

### Simulation-based exploration of model complexity in high-dimensional regression

This project implements a **simulation framework** to study the **double descent phenomenon** in linear regression.

It demonstrates how test error behaves as model complexity increases, especially in **high-dimensional (p ≥ n) settings**.

---

## 🎯 Problem Statement

Classical learning theory suggests that increasing model complexity leads to overfitting.

However, modern machine learning models often exhibit:

* Rising test error near interpolation
* Followed by improved performance in highly overparameterized regimes

This behavior is known as **double descent**.

The goal of this project is to:

👉 Simulate and analyze this phenomenon
👉 Understand instability near the interpolation threshold
👉 Study the effect of regularization and noise

---

## 🧠 Project Overview

The project builds an end-to-end simulation pipeline:

```text
Synthetic Data Generation
        ⬇️
Feature-to-Sample Ratio (p/n)
        ⬇️
OLS (Minimum-Norm Solution)
        ⬇️
Ridge Regression
        ⬇️
Test Error Computation
        ⬇️
Condition Number Analysis
        ⬇️
Visualization
```

---

## 📊 Simulation Setup

Data is generated using a synthetic linear model:

[
X \sim \mathcal{N}(0, \Sigma), \quad y = X\beta + \epsilon
]

Where:

* ( \Sigma ) controls feature correlation
* ( \epsilon \sim \mathcal{N}(0, \sigma^2) ) is noise

Key parameter:

* **p/n ratio** — controls model complexity

---

## 🛠️ Key Components

### 1️⃣ Data Generation

* Gaussian feature generation
* Optional feature correlation (Toeplitz covariance)
* Randomized trials for robustness

---

### 2️⃣ Model Complexity Control

* Vary number of features ( p ) relative to samples ( n )
* Explore:

  * Underparameterized regime (p/n < 1)
  * Interpolation threshold (p/n ≈ 1)
  * Overparameterized regime (p/n > 1)

---

### 3️⃣ Modeling Approaches

#### OLS (Minimum-Norm Solution)

* Uses pseudoinverse:
  [
  w = X^{+} y
  ]
* Works in both under- and overparameterized regimes
* Highly unstable near p/n ≈ 1

---

#### Ridge Regression

* Regularized solution:
  [
  w = (X^T X + \lambda I)^{-1} X^T y
  ]
* Improves numerical stability
* Reduces variance and peak error

---

### 4️⃣ Model Evaluation

Metric: **Mean Squared Error (MSE)**

* Evaluated on independent test data
* Averaged across multiple trials
* Variability also measured

---

### 5️⃣ Numerical Stability Analysis

* Computes condition number of ( X^T X )
* Shows instability near interpolation threshold
* Connects linear algebra to generalization behavior

---

## 📈 Key Observations

* 📉 Test error peaks near p/n ≈ 1
* ⚠️ Ill-conditioned matrices amplify noise
* 🔁 Error decreases again in overparameterized regime
* 🛡️ Ridge regularization smooths the curve

---

## 🧰 Tools & Technologies

* Python
* NumPy
* Matplotlib


---

## 💡 What This Project Demonstrates

* Simulation of modern ML phenomena
* Understanding beyond bias–variance tradeoff
* Numerical linear algebra in machine learning
* Effect of regularization in high dimensions

---

## 🎯 Why This Matters

Double descent explains behavior in:

* Deep learning models
* Overparameterized systems
* Modern ML architectures

This project provides a **controlled experimental setup** to study these effects.

---

## 📌 Project Highlights

* Built a simulation framework for high-dimensional regression
* Implemented minimum-norm OLS and ridge regression
* Analyzed model behavior across regimes
* Connected theory with empirical results

---



## ⭐ If you found this useful

Feel free to explore and build on this work!
