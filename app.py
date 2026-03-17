import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Double Descent Explorer",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Double Descent Explorer")
st.markdown(
    """
This app visualizes the **double descent phenomenon** in high-dimensional linear regression.

It shows how **test error** changes as the feature-to-sample ratio \\( p/n \\) increases,
and how this behavior is affected by:

- **noise**
- **ridge regularization**
- **feature correlation**
"""
)


# -----------------------------
# Helper functions
# -----------------------------
def make_covariance_matrix(p: int, rho: float) -> np.ndarray:
    """
    Toeplitz-style covariance matrix:
    Sigma[i, j] = rho^|i-j|
    """
    idx = np.arange(p)
    return rho ** np.abs(np.subtract.outer(idx, idx))


def generate_data(
    n_train: int,
    n_test: int,
    p: int,
    noise_std: float,
    rho: float,
    rng: np.random.Generator,
):
    """
    Generate synthetic regression data:
        X ~ N(0, Sigma)
        y = X beta + epsilon
    """
    if rho > 0:
        Sigma = make_covariance_matrix(p, rho)
        X_train = rng.multivariate_normal(
            mean=np.zeros(p), cov=Sigma, size=n_train
        )
        X_test = rng.multivariate_normal(
            mean=np.zeros(p), cov=Sigma, size=n_test
        )
    else:
        X_train = rng.normal(size=(n_train, p))
        X_test = rng.normal(size=(n_test, p))

    # True coefficient vector
    beta_true = rng.normal(size=p) / np.sqrt(max(p, 1))

    # Targets
    y_train = X_train @ beta_true + rng.normal(scale=noise_std, size=n_train)
    y_test = X_test @ beta_true + rng.normal(scale=noise_std, size=n_test)

    return X_train, y_train, X_test, y_test


def fit_ols_min_norm(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Minimum-norm least squares solution using pseudoinverse.
    Works in both underparameterized and overparameterized regimes.
    """
    return np.linalg.pinv(X) @ y


def fit_ridge(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """
    Ridge regression closed form:
        w = (X^T X + lambda I)^(-1) X^T y
    """
    p = X.shape[1]
    A = X.T @ X + ridge_lambda * np.eye(p)
    b = X.T @ y
    return np.linalg.solve(A, b)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def condition_number_xtx(X: np.ndarray) -> float:
    """
    Compute condition number of X^T X.
    Uses SVD for numerical stability.
    """
    s = np.linalg.svd(X, compute_uv=False)
    if s.size == 0:
        return np.nan

    s_max = np.max(s)
    s_min = np.min(s)

    # Condition number of X^T X is (s_max / s_min)^2
    if s_min < 1e-12:
        return np.inf
    return float((s_max / s_min) ** 2)


@st.cache_data(show_spinner=False)
def run_simulation(
    n_train: int,
    n_test: int,
    p_min: int,
    p_max: int,
    p_step: int,
    noise_std: float,
    ridge_lambda: float,
    rho: float,
    n_trials: int,
    seed: int,
):
    """
    Run simulation across multiple p values and trials.
    Returns averaged OLS/Ridge test errors and condition numbers.
    """
    rng_master = np.random.default_rng(seed)
    p_values = list(range(p_min, p_max + 1, p_step))

    gamma_values = []
    ols_errors_mean = []
    ridge_errors_mean = []
    cond_mean = []

    ols_errors_std = []
    ridge_errors_std = []
    cond_std = []

    for p in p_values:
        ols_trial_errors = []
        ridge_trial_errors = []
        cond_trial_values = []

        for _ in range(n_trials):
            trial_seed = int(rng_master.integers(0, 1_000_000_000))
            rng = np.random.default_rng(trial_seed)

            X_train, y_train, X_test, y_test = generate_data(
                n_train=n_train,
                n_test=n_test,
                p=p,
                noise_std=noise_std,
                rho=rho,
                rng=rng,
            )

            # OLS minimum-norm solution
            w_ols = fit_ols_min_norm(X_train, y_train)
            y_pred_ols = X_test @ w_ols
            ols_trial_errors.append(mse(y_test, y_pred_ols))

            # Ridge solution
            w_ridge = fit_ridge(X_train, y_train, ridge_lambda)
            y_pred_ridge = X_test @ w_ridge
            ridge_trial_errors.append(mse(y_test, y_pred_ridge))

            cond_trial_values.append(condition_number_xtx(X_train))

        gamma = p / n_train
        gamma_values.append(gamma)

        ols_errors_mean.append(float(np.mean(ols_trial_errors)))
        ridge_errors_mean.append(float(np.mean(ridge_trial_errors)))
        cond_mean.append(float(np.mean(cond_trial_values)))

        ols_errors_std.append(float(np.std(ols_trial_errors)))
        ridge_errors_std.append(float(np.std(ridge_trial_errors)))
        cond_std.append(float(np.std(cond_trial_values)))

    return {
        "p_values": np.array(p_values),
        "gamma_values": np.array(gamma_values),
        "ols_mean": np.array(ols_errors_mean),
        "ridge_mean": np.array(ridge_errors_mean),
        "cond_mean": np.array(cond_mean),
        "ols_std": np.array(ols_errors_std),
        "ridge_std": np.array(ridge_errors_std),
        "cond_std": np.array(cond_std),
    }


def summarize_behavior(gamma_values, ols_curve, ridge_curve):
    peak_idx = int(np.argmax(ols_curve))
    peak_gamma = gamma_values[peak_idx]
    peak_value = ols_curve[peak_idx]

    ridge_peak_idx = int(np.argmax(ridge_curve))
    ridge_peak_gamma = gamma_values[ridge_peak_idx]
    ridge_peak_value = ridge_curve[ridge_peak_idx]

    lines = []
    lines.append(
        f"- **OLS peak test error** occurs around **p/n ≈ {peak_gamma:.2f}**, with error ≈ **{peak_value:.4f}**."
    )
    lines.append(
        f"- **Ridge peak test error** occurs around **p/n ≈ {ridge_peak_gamma:.2f}**, with error ≈ **{ridge_peak_value:.4f}**."
    )

    if peak_value > ridge_peak_value:
        lines.append(
            "- **Ridge reduces the height of the interpolation peak**, making the model more stable."
        )
    else:
        lines.append(
            "- In this run, ridge does not reduce the peak much; this can happen depending on the chosen settings."
        )

    if np.any((gamma_values > 0.8) & (gamma_values < 1.2)):
        lines.append(
            "- Around **p/n = 1**, the model is near the **interpolation threshold**, where instability often becomes strongest."
        )

    if ols_curve[-1] < peak_value:
        lines.append(
            "- In the highly overparameterized region, the OLS curve drops after the peak, showing the **second descent**."
        )

    return "\n".join(lines)


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

n_train = st.sidebar.slider("Training samples (n)", 30, 300, 100, 10)
n_test = st.sidebar.slider("Test samples", 100, 3000, 1000, 100)

p_min = st.sidebar.slider("Minimum features (p_min)", 5, 50, 10, 1)
p_max = st.sidebar.slider("Maximum features (p_max)", 20, 500, 250, 5)
p_step = st.sidebar.slider("Feature step", 1, 25, 5, 1)

noise_std = st.sidebar.slider("Noise level (σ)", 0.0, 3.0, 0.5, 0.1)
ridge_lambda = st.sidebar.slider("Ridge λ", 0.0, 20.0, 1.0, 0.1)
rho = st.sidebar.slider("Feature correlation (ρ)", 0.0, 0.95, 0.0, 0.05)

n_trials = st.sidebar.slider("Number of trials", 1, 50, 10, 1)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42)

show_ridge = st.sidebar.checkbox("Show Ridge comparison", value=True)
show_condition = st.sidebar.checkbox("Show condition number plot", value=True)
show_error_band = st.sidebar.checkbox("Show variability band", value=True)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run simulation", use_container_width=True)

if p_max <= p_min:
    st.error("`p_max` must be greater than `p_min`.")
    st.stop()

# First run automatically, or rerun when button pressed
if "run_once" not in st.session_state:
    st.session_state.run_once = True

if run_button or st.session_state.run_once:
    st.session_state.run_once = False

    with st.spinner("Running simulation..."):
        results = run_simulation(
            n_train=n_train,
            n_test=n_test,
            p_min=p_min,
            p_max=p_max,
            p_step=p_step,
            noise_std=noise_std,
            ridge_lambda=ridge_lambda,
            rho=rho,
            n_trials=n_trials,
            seed=int(seed),
        )

    gamma_values = results["gamma_values"]
    ols_mean = results["ols_mean"]
    ridge_mean = results["ridge_mean"]
    cond_mean = results["cond_mean"]
    ols_std = results["ols_std"]
    ridge_std = results["ridge_std"]
    cond_std = results["cond_std"]

    # -----------------------------
    # Summary metrics
    # -----------------------------
    st.subheader("Quick Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training samples", f"{n_train}")
    col2.metric("Max p/n", f"{(p_max / n_train):.2f}")
    col3.metric("Noise σ", f"{noise_std:.2f}")
    col4.metric("Correlation ρ", f"{rho:.2f}")

    # -----------------------------
    # Main plot
    # -----------------------------
    st.subheader("Main Double Descent Curve")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(gamma_values, ols_mean, marker="o", label="OLS (min-norm)")

    if show_error_band:
        ax1.fill_between(
            gamma_values,
            ols_mean - ols_std,
            ols_mean + ols_std,
            alpha=0.2,
        )

    if show_ridge:
        ax1.plot(gamma_values, ridge_mean, marker="s", label=f"Ridge (λ={ridge_lambda:.2f})")
        if show_error_band:
            ax1.fill_between(
                gamma_values,
                ridge_mean - ridge_std,
                ridge_mean + ridge_std,
                alpha=0.2,
            )

    ax1.axvline(1.0, linestyle="--", linewidth=1.5, label="Interpolation threshold (p/n = 1)")
    ax1.set_xlabel("Feature-to-sample ratio (p/n)")
    ax1.set_ylabel("Test MSE")
    ax1.set_title("Test Error vs Model Complexity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    st.pyplot(fig1)

    # -----------------------------
    # Condition number plot
    # -----------------------------
    if show_condition:
        st.subheader("Matrix Conditioning")

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(gamma_values, cond_mean, marker="o", label="Condition number of XᵀX")

        if show_error_band:
            lower = np.maximum(cond_mean - cond_std, 0)
            upper = cond_mean + cond_std
            ax2.fill_between(gamma_values, lower, upper, alpha=0.2)

        ax2.axvline(1.0, linestyle="--", linewidth=1.5, label="p/n = 1")
        ax2.set_xlabel("Feature-to-sample ratio (p/n)")
        ax2.set_ylabel("Condition number")
        ax2.set_title("Condition Number vs Model Complexity")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        st.pyplot(fig2)

    # -----------------------------
    # Interpretation box
    # -----------------------------
    st.subheader("Interpretation")
    st.markdown(
        summarize_behavior(
            gamma_values=gamma_values,
            ols_curve=ols_mean,
            ridge_curve=ridge_mean,
        )
    )

    # -----------------------------
    # Theory explanation
    # -----------------------------
    st.subheader("How to Read This")
    st.info(
        """
- When **p/n < 1**, the model is usually **underparameterized**.
- Near **p/n = 1**, the model is close to the **interpolation threshold**.
- Around this region, the matrix can become **ill-conditioned**, which amplifies noise.
- When **p/n > 1**, the model becomes **overparameterized**.
- In many cases, test error drops again after the peak, creating the **double descent** shape.
- **Ridge regression** often smooths the peak because it improves numerical stability.
"""
    )

    # -----------------------------
    # Data table
    # -----------------------------
    with st.expander("Show simulation data"):
        st.dataframe(
            {
                "p/n": gamma_values,
                "OLS mean test error": ols_mean,
                "OLS std": ols_std,
                "Ridge mean test error": ridge_mean,
                "Ridge std": ridge_std,
                "Condition number mean": cond_mean,
                "Condition number std": cond_std,
            },
            use_container_width=True,
        )

else:
    st.info("Adjust the controls and click **Run simulation**.")