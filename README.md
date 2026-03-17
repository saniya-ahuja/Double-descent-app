# 📉 The Double Descent Phenomenon in High-Dimensional Linear Regression

### A Theoretical and Empirical Study

This project investigates the **Double Descent phenomenon** in modern machine learning through a combination of **theoretical insights and simulation-based experiments**.

It bridges the gap between **classical statistical learning theory** and the behavior of **overparameterized models**.

---

## 🎯 Research Motivation

Classical learning theory predicts a **U-shaped curve** between model complexity and test error (bias–variance tradeoff).

However, modern models often show:

* A second decrease in test error
* Even after perfectly fitting training data

This leads to the **Double Descent phenomenon**, where:

👉 Test error increases near interpolation
👉 Then decreases again in highly overparameterized regimes

Understanding this behavior is essential for explaining the success of:

* Deep learning models
* High-dimensional systems



---

## 🎯 Problem Statement

The project focuses on understanding:

* Why test error peaks near the interpolation threshold (p ≈ n)
* What causes the second descent in overparameterized regimes
* How factors like:

  * Noise variance
  * Feature correlation
  * Dimensional scaling
    affect generalization performance



---

## 🧠 Project Overview

The study combines **theoretical analysis + simulations**:

```text
Synthetic Data Generation
        ⬇️
High-Dimensional Scaling (p/n)
        ⬇️
OLS (Minimum-Norm Solution)
        ⬇️
Ridge Regression
        ⬇️
Test Error Analysis
        ⬇️
Spectral & Conditioning Analysis
```

---

## 📐 Mathematical Framework

The project uses:

* Bias–Variance decomposition
* Gaussian feature assumptions
* Prediction risk as a function of p/n

Goal:

👉 Derive insights into how dimensionality impacts generalization



---

## 🔍 Spectral Analysis & Matrix Conditioning

A key focus is on:

* Singular Value Decomposition (SVD)
* Eigenvalue distribution of covariance matrices

Findings:

* Conditioning worsens near p ≈ n
* Leads to instability and error peak
* Stabilizes again in overparameterized regime

This explains the **mechanism behind double descent**



---

## 🧪 Experimental Strategy

Theoretical insights are validated using simulations:

* Synthetic datasets with controllable:

  * Feature correlation
  * Noise
  * Dimensionality

* Test error tracked across p/n

* Ridge regularization added to study smoothing effects



---

## 📊 Key Observations

* 📉 Error spikes near interpolation threshold
* ⚠️ Ill-conditioning amplifies noise
* 🔁 Error decreases again in high dimensions
* 🛡️ Ridge regularization stabilizes predictions

---

## 🧰 Tools & Technologies

* Python
* NumPy
* SciPy
* Matplotlib

---

## 💡 What This Project Demonstrates

* Understanding of modern ML theory
* High-dimensional statistics
* Numerical linear algebra in ML
* Simulation-based validation of theory
* Bridging theory and practice

---

## 🎯 Why This Matters

This phenomenon explains why:

* Deep neural networks generalize well
* Overparameterized models outperform classical expectations

This project provides a **controlled framework** to study these effects.

---

## 📌 Project Highlights

* Developed a simulation framework for double descent
* Analyzed interpolation threshold behavior
* Connected spectral properties with generalization
* Demonstrated impact of ridge regularization


---

## ⭐ If you found this useful

Feel free to explore and build on this work!
