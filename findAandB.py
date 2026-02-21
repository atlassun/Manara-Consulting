# findAandB.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Robust methods (may be unavailable on very old sklearn)
try:
    from sklearn.linear_model import TheilSenRegressor, RANSACRegressor
except Exception:
    TheilSenRegressor = None
    RANSACRegressor = None

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Bellerose | Find a & b (Y=aX+b) | EN/FR", layout="wide")
DEFAULT_XLSX_PATH = "bellerose.xlsx"

# -----------------------------
# BILINGUAL TEXT
# -----------------------------
T = {
    "title": {
        "en": "📈 Find a & b — Linear Relationship: Cost = a·Volume + b",
        "fr": "📈 Trouver a & b — Relation linéaire : Coût = a·Volume + b",
    },
    "caption": {
        "en": "Intuitive bilingual app to estimate a and b using different methods (Linear Regression (OLS), extremes, percentiles, robust).",
        "fr": "Application bilingue intuitive pour estimer a et b avec différentes méthodes (OLS, extrêmes, percentiles, robustes).",
    },
    "how_title": {"en": "How to read this", "fr": "Comment lire ceci"},
    "how_text": {
        "en": (
            "We model a straight line: **Y = aX + b**\n\n"
            "- **X** = Volume (predictor)\n"
            "- **Y** = Cost (target)\n\n"
            "**a (slope)**: change in Cost when Volume increases by 1.\n"
            "**b (intercept)**: predicted Cost when Volume = 0 (may be only a mathematical offset)."
        ),
        "fr": (
            "On modélise une droite : **Y = aX + b**\n\n"
            "- **X** = Volume (variable explicative)\n"
            "- **Y** = Coût (variable à prédire)\n\n"
            "**a (pente)** : variation du coût quand le volume augmente de 1.\n"
            "**b (interception)** : coût prédit quand Volume = 0 (parfois juste un décalage mathématique)."
        ),
    },
    "data": {"en": "Data", "fr": "Données"},
    "params": {"en": "Parameters", "fr": "Paramètres"},
    "use_default": {"en": "Use default file (bellerose.xlsx)", "fr": "Utiliser le fichier par défaut (bellerose.xlsx)"},
    "upload": {"en": "Or upload an Excel/CSV", "fr": "Ou téléverser un fichier Excel/CSV"},
    "preview": {"en": "Data preview", "fr": "Aperçu des données"},
    "mapping": {"en": "Column mapping", "fr": "Sélection des colonnes"},
    "x": {"en": "X column (Volume)", "fr": "Colonne X (Volume)"},
    "y": {"en": "Y column (Cost)", "fr": "Colonne Y (Coût)"},
    "clean": {"en": "Data cleaning", "fr": "Nettoyage des données"},
    "neg": {"en": "Remove negative values", "fr": "Retirer les valeurs négatives"},
    "zeros": {"en": "Remove zeros (optional)", "fr": "Retirer les zéros (optionnel)"},
    "iqr": {"en": "Trim outliers (IQR)", "fr": "Retirer les valeurs extrêmes (IQR)"},
    "k": {"en": "IQR aggressiveness (k)", "fr": "Agressivité IQR (k)"},
    "method": {"en": "Method to compute a and b", "fr": "Méthode pour calculer a et b"},
    "overlay": {"en": "Overlay all methods on chart", "fr": "Afficher toutes les méthodes sur le graphe"},
    "predict": {"en": "Predict cost for a volume", "fr": "Prédire le coût pour un volume"},
    "whatif": {"en": "What-if scenarios", "fr": "Scénarios (What-if)"},
    "charts": {"en": "Charts", "fr": "Graphiques"},
    "scatter": {"en": "Scatter + fitted line(s)", "fr": "Nuage de points + droite(s)"},
    "resid": {"en": "Residual analysis", "fr": "Analyse des résidus"},
    "download": {"en": "Download predictions (CSV)", "fr": "Télécharger les prédictions (CSV)"},
    "note": {
        "en": "Note: This app demonstrates multiple ways to estimate a and b for Y=aX+b.",
        "fr": "Note : cette application montre plusieurs façons d’estimer a et b pour Y=aX+b.",
    },
}

# -----------------------------
# HELPERS
# -----------------------------
def normalize(s: str) -> str:
    return (
        str(s).strip().lower()
        .replace("é", "e").replace("è", "e").replace("ê", "e")
        .replace("à", "a").replace("ù", "u").replace("ô", "o")
        .replace("ï", "i").replace("ç", "c")
        .replace(" ", "").replace("_", "").replace("-", "")
        .replace("/", "").replace(".", "")
    )

def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {normalize(c): c for c in df.columns}
    for cand in candidates:
        key = normalize(cand)
        if key in norm_map:
            return norm_map[key]
    for col in df.columns:
        ncol = normalize(col)
        if any(normalize(c) in ncol for c in candidates):
            return col
    return None

def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(r"[\$,]", "", regex=True)
         .str.replace(" ", "")
         .str.replace("\u00A0", "")
         .str.replace(",", "."),
        errors="coerce"
    )

def rmse_safe(y_true, y_pred) -> float:
    # Compatible with older sklearn (no squared= parameter)
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))

def fmt_num(x: float, d: int = 4) -> str:
    try:
        return f"{x:,.{d}f}"
    except Exception:
        return str(x)

def fmt_money(x: float, symbol: str = "$") -> str:
    try:
        return f"{symbol}{x:,.2f}"
    except Exception:
        return str(x)

def iqr_trim_df(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    for c in cols:
        q1 = out[c].quantile(0.25)
        q3 = out[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        mask &= out[c].between(lo, hi)
    return out.loc[mask].copy()

# --- a,b methods ---
def fit_ab_ols(x: np.ndarray, y: np.ndarray):
    m = LinearRegression()
    m.fit(x.reshape(-1, 1), y)
    return float(m.coef_[0]), float(m.intercept_), m

def fit_ab_two_point(x1, y1, x2, y2):
    if x2 == x1:
        return np.nan, np.nan
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return float(a), float(b)

def fit_ab_minmax(x: np.ndarray, y: np.ndarray):
    i_min = int(np.argmin(x))
    i_max = int(np.argmax(x))
    return fit_ab_two_point(x[i_min], y[i_min], x[i_max], y[i_max])

def fit_ab_percentiles(x: np.ndarray, y: np.ndarray, p1=10, p2=90):
    x1 = float(np.percentile(x, p1))
    x2 = float(np.percentile(x, p2))

    def local_median(x0):
        xr = max(float(x.max() - x.min()), 1e-12)
        w = 0.05 * xr
        mask = np.abs(x - x0) <= w
        if mask.sum() < 3:
            idx = np.argsort(np.abs(x - x0))[:5]
            return float(np.median(y[idx]))
        return float(np.median(y[mask]))

    y1 = local_median(x1)
    y2 = local_median(x2)
    return fit_ab_two_point(x1, y1, x2, y2)

def fit_ab_theilsen(x: np.ndarray, y: np.ndarray):
    if TheilSenRegressor is None:
        return np.nan, np.nan, None
    m = TheilSenRegressor(random_state=0)
    m.fit(x.reshape(-1, 1), y)
    return float(m.coef_[0]), float(m.intercept_), m

def fit_ab_ransac(x: np.ndarray, y: np.ndarray):
    if RANSACRegressor is None:
        return np.nan, np.nan, None
    base = LinearRegression()
    m = RANSACRegressor(estimator=base, random_state=0)
    m.fit(x.reshape(-1, 1), y)
    a = float(m.estimator_.coef_[0])
    b = float(m.estimator_.intercept_)
    return a, b, m

def evaluate_line(x: np.ndarray, y: np.ndarray, a: float, b: float):
    y_hat = a * x + b
    return {
        "a": float(a),
        "b": float(b),
        "R2": float(r2_score(y, y_hat)),
        "MAE": float(mean_absolute_error(y, y_hat)),
        "RMSE": rmse_safe(y, y_hat),
    }

def regression_confidence_band(x: np.ndarray, y: np.ndarray, a: float, b: float, x_grid: np.ndarray):
    """
    Approximate 95% confidence band for mean prediction (simple linear regression).
    Uses normal approximation (z=1.96). Good for UX; approximate for small n.
    """
    n = len(x)
    x_mean = x.mean()
    y_hat = a * x + b
    residuals = y - y_hat
    sse = np.sum(residuals ** 2)
    s2 = sse / max(n - 2, 1)
    s = np.sqrt(s2)

    denom = np.sum((x - x_mean) ** 2)
    denom = denom if denom > 0 else 1e-12

    se_mean = s * np.sqrt((1 / n) + ((x_grid - x_mean) ** 2) / denom)
    z = 1.96
    y_hat_grid = a * x_grid + b
    y_lo = y_hat_grid - z * se_mean
    y_hi = y_hat_grid + z * se_mean
    return y_hat_grid, y_lo, y_hi

# -----------------------------
# SIDEBAR (Language + Inputs)
# -----------------------------
with st.sidebar:
    st.header("Language / Langue")
    lang = st.radio("Choose / Choisir", ["EN", "FR"], index=1, horizontal=True)
    L = "en" if lang == "EN" else "fr"

    st.divider()
    st.header(T["data"][L])
    use_default = st.checkbox(T["use_default"][L], value=True)
    uploaded = None if use_default else st.file_uploader(T["upload"][L], type=["xlsx", "xls", "csv"])

    st.divider()
    st.header(T["params"][L])
    show_explain = st.checkbox("Explain like I'm new / Explication simple", value=True)

    st.subheader(T["clean"][L])
    remove_neg = st.checkbox(T["neg"][L], value=False)
    remove_zero = st.checkbox(T["zeros"][L], value=False)
    trim_outliers = st.checkbox(T["iqr"][L], value=False)
    iqr_k = st.slider(T["k"][L], 1.0, 3.0, 1.5, 0.1)

    st.divider()
    st.header("Display / Affichage")
    use_currency = st.checkbox("Currency formatting / Format monétaire", value=False)
    currency_symbol = st.selectbox("Symbol", ["$", "CAD $", "US $", "€"], index=0)
    money_sym = "$" if currency_symbol == "$" else currency_symbol + " "

    st.divider()
    show_residuals = st.checkbox("Show residual analysis / Résidus", value=True)
    show_band = st.checkbox("Show 95% band (OLS only) / Bande 95% (OLS)", value=True)

def YFMT(v: float) -> str:
    return fmt_money(v, symbol=money_sym) if use_currency else fmt_num(v, 4)

# -----------------------------
# HEADER
# -----------------------------
st.title(T["title"][L])
st.caption(T["caption"][L])

if show_explain:
    with st.expander(T["how_title"][L], expanded=True):
        st.markdown(T["how_text"][L])

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data(use_default: bool, uploaded_file):
    if use_default:
        df0 = pd.read_excel(DEFAULT_XLSX_PATH)
        source = DEFAULT_XLSX_PATH
    else:
        if uploaded_file is None:
            return None, None
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df0 = pd.read_csv(uploaded_file)
        else:
            df0 = pd.read_excel(uploaded_file)
        source = uploaded_file.name
    return df0, source

df_raw, source = load_data(use_default, uploaded)
if df_raw is None:
    st.warning("Please use the default file or upload one. / Veuillez utiliser le fichier par défaut ou en téléverser un.")
    st.stop()

st.success(f"Loaded / Chargé : {source}")
st.subheader(T["preview"][L])
st.dataframe(df_raw.head(25), use_container_width=True)

# -----------------------------
# COLUMN MAPPING
# -----------------------------
x_candidates = ["volume", "vol", "quantite", "quantité", "qty", "qte", "units", "unit"]
y_candidates = ["cout", "coût", "cost", "prix", "montant", "amount", "total"]

x_guess = pick_column(df_raw, x_candidates)
y_guess = pick_column(df_raw, y_candidates)

st.subheader(T["mapping"][L])
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    x_col = st.selectbox(
        T["x"][L],
        options=list(df_raw.columns),
        index=(list(df_raw.columns).index(x_guess) if x_guess in df_raw.columns else 0),
        help="Predictor: Volume / Variable explicative : Volume",
    )
with c2:
    y_col = st.selectbox(
        T["y"][L],
        options=list(df_raw.columns),
        index=(list(df_raw.columns).index(y_guess) if y_guess in df_raw.columns else min(1, len(df_raw.columns) - 1)),
        help="Target: Cost / Variable cible : Coût",
    )
with c3:
    st.info("Auto-detection helps, but you can override. / L’auto-détection aide, mais vous pouvez choisir manuellement.")

# -----------------------------
# CLEAN & PREPARE
# -----------------------------
df = df_raw.copy()
df[x_col] = to_numeric_series(df[x_col])
df[y_col] = to_numeric_series(df[y_col])
df = df.dropna(subset=[x_col, y_col])

if remove_neg:
    df = df[(df[x_col] >= 0) & (df[y_col] >= 0)]
if remove_zero:
    df = df[(df[x_col] != 0) & (df[y_col] != 0)]
if trim_outliers and len(df) >= 10:
    df = iqr_trim_df(df, [x_col, y_col], k=iqr_k)

if len(df) < 3:
    st.error("Not enough numeric rows after cleaning. / Pas assez de lignes numériques après nettoyage.")
    st.stop()

# Arrays
x_all = df[x_col].values.astype(float)
y_all = df[y_col].values.astype(float)
xmin, xmax = float(x_all.min()), float(x_all.max())

# -----------------------------
# METHOD SELECTOR (a,b)
# -----------------------------
st.markdown("## " + (T["method"][L]))

method_options = [
    "OLS (Least Squares) / Moindres carrés (standard)",
    "Two-point (min/max extremes) / Deux points (min/max)",
    "Two-point (10th/90th percentiles) / Deux points (10%/90%)",
]

if TheilSenRegressor is not None:
    method_options.append("Theil–Sen (robust) / Theil–Sen (robuste)")
else:
    st.warning("Theil–Sen not available in this scikit-learn. / Theil–Sen indisponible.")

if RANSACRegressor is not None:
    method_options.append("RANSAC (robust) / RANSAC (robuste)")
else:
    st.warning("RANSAC not available in this scikit-learn. / RANSAC indisponible.")

chosen_method = st.selectbox(T["method"][L], method_options, index=0)
overlay = st.checkbox(T["overlay"][L], value=True)

def pick_active(chosen: str) -> str:
    if chosen.startswith("OLS"):
        return "OLS"
    if "min/max" in chosen.lower() or "min/max" in chosen:
        return "Min/Max"
    if "10th/90th" in chosen or "10%/90%" in chosen:
        return "P10/P90"
    if chosen.startswith("Theil"):
        return "Theil–Sen"
    if chosen.startswith("RANSAC"):
        return "RANSAC"
    return "OLS"

# Fit all methods (for comparison + overlay)
results = []
lines = {}  # name -> (a, b)

# OLS
a_ols, b_ols, _m_ols = fit_ab_ols(x_all, y_all)
lines["OLS"] = (a_ols, b_ols)
m = evaluate_line(x_all, y_all, a_ols, b_ols)
results.append({"Method": "OLS", **m})

# Min/Max
a_mm, b_mm = fit_ab_minmax(x_all, y_all)
lines["Min/Max"] = (a_mm, b_mm)
if np.isfinite(a_mm) and np.isfinite(b_mm):
    m = evaluate_line(x_all, y_all, a_mm, b_mm)
    results.append({"Method": "Min/Max", **m})

# P10/P90
a_p, b_p = fit_ab_percentiles(x_all, y_all, 10, 90)
lines["P10/P90"] = (a_p, b_p)
if np.isfinite(a_p) and np.isfinite(b_p):
    m = evaluate_line(x_all, y_all, a_p, b_p)
    results.append({"Method": "P10/P90", **m})

# Theil–Sen
if TheilSenRegressor is not None:
    a_ts, b_ts, _ = fit_ab_theilsen(x_all, y_all)
    lines["Theil–Sen"] = (a_ts, b_ts)
    if np.isfinite(a_ts) and np.isfinite(b_ts):
        m = evaluate_line(x_all, y_all, a_ts, b_ts)
        results.append({"Method": "Theil–Sen", **m})

# RANSAC
if RANSACRegressor is not None:
    a_ra, b_ra, _ = fit_ab_ransac(x_all, y_all)
    lines["RANSAC"] = (a_ra, b_ra)
    if np.isfinite(a_ra) and np.isfinite(b_ra):
        m = evaluate_line(x_all, y_all, a_ra, b_ra)
        results.append({"Method": "RANSAC", **m})

active_key = pick_active(chosen_method)
a, b = lines.get(active_key, lines["OLS"])

# Predictions with active method
df = df.sort_values(by=x_col).reset_index(drop=True)
df["y_hat"] = a * df[x_col].astype(float) + b
df["residual"] = df[y_col].astype(float) - df["y_hat"]

active_metrics = evaluate_line(x_all, y_all, a, b)

# -----------------------------
# PRODUCT-LIKE TABS
# -----------------------------
tabs = st.tabs([
    "🧾 Data / Données",
    "📐 Model / Modèle",
    "🧠 Insights / Interprétation",
    "⬇️ Export / Export"
])

# TAB 1: DATA
with tabs[0]:
    st.subheader("Data / Données")
    st.write(("Rows after cleaning" if L == "en" else "Lignes après nettoyage") + f": **{len(df):,}**")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("**Quick stats / Statistiques rapides**")
    q1, q2, q3 = st.columns(3)
    q1.metric("X min", fmt_num(float(df[x_col].min()), 4))
    q2.metric("X median", fmt_num(float(df[x_col].median()), 4))
    q3.metric("X max", fmt_num(float(df[x_col].max()), 4))

# TAB 2: MODEL
with tabs[1]:
    st.subheader("Model / Modèle")

    # Comparison table
    st.markdown("### Comparison / Comparaison")
    res_df = pd.DataFrame(results)
    # Sort by RMSE (lower is better) if present
    if "RMSE" in res_df.columns:
        res_df = res_df.sort_values("RMSE", ascending=True)
    st.dataframe(res_df, use_container_width=True)

    st.markdown("### Active method / Méthode active")
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Method / Méthode", active_key)
    cB.metric("a / slope", fmt_num(a, 6))
    cC.metric("b / intercept", fmt_num(b, 6))
    cD.metric("RMSE", fmt_num(active_metrics["RMSE"], 4))

    cE, cF = st.columns(2)
    cE.metric("R²", fmt_num(active_metrics["R2"], 4))
    cF.metric("MAE", fmt_num(active_metrics["MAE"], 4))

    st.markdown(f"### Equation / Équation\n**{y_col} = {a:.6f} × {x_col} + {b:.6f}**")

    # Prediction widget
    st.subheader(T["predict"][L])
    x_input = st.slider("Volume (X)", min_value=float(xmin), max_value=float(xmax), value=float(np.median(x_all)))
    y_out = float(a * x_input + b)
    st.success(
        (f"✅ Predicted {y_col} for Volume={fmt_num(x_input,4)} → **{YFMT(y_out)}**"
         if L == "en"
         else f"✅ {y_col} prédit pour Volume={fmt_num(x_input,4)} → **{YFMT(y_out)}**")
    )

    # What-if scenarios table
    st.subheader(T["whatif"][L])
    n_points = st.slider("Number of scenarios / Nombre de scénarios", 3, 25, 10)
    vol_grid = np.linspace(xmin, xmax, n_points)
    pred_grid = a * vol_grid + b
    whatif_df = pd.DataFrame({x_col: vol_grid, (y_col + "_pred") if L == "en" else (y_col + "_predit"): pred_grid})
    st.dataframe(whatif_df, use_container_width=True)

    # Chart: scatter + fitted line(s)
    st.subheader(T["charts"][L])
    st.markdown(f"#### {T['scatter'][L]}")
    x_line = np.linspace(xmin, xmax, 200)

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(df[x_col], df[y_col], label="Data / Données")

    if overlay:
        for name, (aa, bb) in lines.items():
            if np.isfinite(aa) and np.isfinite(bb):
                plt.plot(x_line, aa * x_line + bb, label=name)
    else:
        plt.plot(x_line, a * x_line + b, label=f"Active: {active_key}")

    # Optional confidence band (only meaningful for OLS in this simple implementation)
    if show_band and active_key == "OLS" and len(df) >= 3:
        y_hat_grid, y_lo, y_hi = regression_confidence_band(x_all, y_all, a_ols, b_ols, x_line)
        plt.fill_between(x_line, y_lo, y_hi, alpha=0.2, label="95% band (OLS)")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Y = aX + b")
    plt.legend()
    st.pyplot(fig, clear_figure=True)

# TAB 3: INSIGHTS
with tabs[2]:
    st.subheader("Insights / Interprétation")

    if L == "en":
        st.markdown(
            f"- **Slope (a) = {fmt_num(a,6)}** → When **{x_col}** increases by **1**, predicted **{y_col}** changes by about **{YFMT(a)}**.\n"
            f"- **Intercept (b) = {YFMT(b)}** → Predicted **{y_col}** when **{x_col}=0** (sometimes only a mathematical offset).\n"
            f"- **R² = {fmt_num(active_metrics['R2'],4)}** → Closer to 1 means the line explains the relationship well.\n"
            f"- **RMSE = {fmt_num(active_metrics['RMSE'],4)}** → Typical prediction error size (penalizes big errors)."
        )
    else:
        st.markdown(
            f"- **Pente (a) = {fmt_num(a,6)}** → Quand **{x_col}** augmente de **1**, **{y_col}** change d’environ **{YFMT(a)}**.\n"
            f"- **Intercept (b) = {YFMT(b)}** → **{y_col}** prédit quand **{x_col}=0** (parfois seulement un décalage mathématique).\n"
            f"- **R² = {fmt_num(active_metrics['R2'],4)}** → Plus proche de 1 = meilleure explication par la droite.\n"
            f"- **RMSE = {fmt_num(active_metrics['RMSE'],4)}** → Taille typique de l’erreur (pénalise les grosses erreurs)."
        )

    if show_residuals:
        st.markdown("### " + (T["resid"][L]))
        c1, c2 = st.columns(2)

        with c1:
            fig2 = plt.figure(figsize=(7, 4))
            plt.scatter(df[x_col], df["residual"])
            plt.axhline(0)
            plt.xlabel(x_col)
            plt.ylabel("Residual (Y - Ŷ)")
            plt.title("Residuals vs X / Résidus vs X")
            st.pyplot(fig2, clear_figure=True)

        with c2:
            fig3 = plt.figure(figsize=(7, 4))
            plt.hist(df["residual"].dropna().values, bins=20)
            plt.xlabel("Residual (Y - Ŷ)")
            plt.ylabel("Count")
            plt.title("Residual distribution / Distribution des résidus")
            st.pyplot(fig3, clear_figure=True)

# TAB 4: EXPORT
with tabs[3]:
    st.subheader("Export / Export")
    export_df = df[[x_col, y_col, "y_hat", "residual"]].copy()
    st.dataframe(export_df.head(100), use_container_width=True)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        T["download"][L],
        data=csv_bytes,
        file_name="bellerose_predictions.csv",
        mime="text/csv"
    )

    params = {
        "x_column": x_col,
        "y_column": y_col,
        "method": active_key,
        "a_slope": float(a),
        "b_intercept": float(b),
        "r2": float(active_metrics["R2"]),
        "mae": float(active_metrics["MAE"]),
        "rmse": float(active_metrics["RMSE"]),
        "n_rows_used": int(len(df)),
        "options": {
            "remove_negatives": bool(remove_neg),
            "remove_zeros": bool(remove_zero),
            "iqr_trim": bool(trim_outliers),
            "iqr_k": float(iqr_k),
            "overlay_lines": bool(overlay),
            "show_residuals": bool(show_residuals),
            "show_95_band_ols": bool(show_band),
        }
    }

    st.download_button(
        "⬇️ Model parameters (JSON) / Paramètres du modèle (JSON)",
        data=pd.Series(params).to_json(indent=2).encode("utf-8"),
        file_name="findAandB_model_params.json",
        mime="application/json"
    )

st.caption(T["note"][L])
