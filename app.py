
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Universal Bank - Personal Loan Propensity", layout="wide")

# ---------------------- Data Loading & Preprocessing ----------------------

@st.cache_data
def load_data(path: str = "UniversalBank.csv"):
    df = pd.read_csv(path)
    # Clean column names
    df.columns = [c.strip().replace("\\xa0"," ") for c in df.columns]
    # Identify target
    target_col = None
    for c in df.columns:
        if c.strip().lower().replace(" ", "") == "personalloan":
            target_col = c
            break
    if target_col is None:
        raise ValueError("Could not find 'Personal Loan' column in dataset.")
    # Drop ID if present
    for id_col in ["ID", "Id", "id"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
    # Basic imputation
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df, target_col


def train_models(df, target_col):
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=300),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    roc_data = {}
    cms = {}
    fitted = {}
    cv_table = {}

    for name, model in models.items():
        # 5-fold CV AUC
        cv_auc = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
        cv_table[name] = cv_auc

        # Fit & evaluate
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_tr_proba = model.predict_proba(X_train)[:, 1]
            y_te_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_tr_proba = model.decision_function(X_train)
            y_te_proba = model.decision_function(X_test)

        y_tr_pred = model.predict(X_train)
        y_te_pred = model.predict(X_test)

        metrics = {
            "Training Accuracy": accuracy_score(y_train, y_tr_pred),
            "Testing Accuracy": accuracy_score(y_test, y_te_pred),
            "Precision": precision_score(y_test, y_te_pred, zero_division=0),
            "Recall": recall_score(y_test, y_te_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_te_pred, zero_division=0),
            "AUC": roc_auc_score(y_test, y_te_proba),
        }

        fpr_te, tpr_te, _ = roc_curve(y_test, y_te_proba)
        roc_data[name] = (fpr_te, tpr_te)

        cms[name] = {
            "train": confusion_matrix(y_train, y_tr_pred),
            "test": confusion_matrix(y_test, y_te_pred),
        }

        results[name] = metrics
        fitted[name] = model

    # Format CV table
    cv_df = pd.DataFrame(cv_table)
    cv_df.index = [f"Fold {i+1}" for i in range(cv_df.shape[0])]
    cv_df.loc["Mean AUC"] = cv_df.mean(axis=0)

    # Summary table
    summary_rows = []
    for name, m in results.items():
        summary_rows.append([
            name,
            m["Training Accuracy"],
            m["Testing Accuracy"],
            m["Precision"],
            m["Recall"],
            m["F1-Score"],
            m["AUC"],
        ])
    summary_df = pd.DataFrame(
        summary_rows,
        columns=["Algorithm", "Training Accuracy", "Testing Accuracy",
                 "Precision", "Recall", "F1-Score", "AUC"]
    )

    return X_train, X_test, y_train, y_test, fitted, summary_df, cv_df, roc_data, cms, X.columns.tolist()


@st.cache_resource
def get_trained_models():
    df, target_col = load_data()
    return (df, target_col) + train_models(df, target_col)


# ---------------------- Helper Plots ----------------------

def show_customer_insights(df, target_col):
    st.subheader("Customer Insights & Complex Patterns")

    y = df[target_col]

    # 1) Target distribution (bar)
    st.markdown("**1. Portfolio view: Personal Loan adoption vs non-adoption**")
    fig1, ax1 = plt.subplots()
    counts = y.value_counts().sort_index()
    ax1.bar(["No (0)", "Yes (1)"], counts.values)
    ax1.set_ylabel("Number of Customers")
    st.pyplot(fig1)

    # 2) Income vs Loan adoption (boxplot)
    st.markdown("**2. Income profile of adopters vs non-adopters (Boxplot)**")
    fig2, ax2 = plt.subplots()
    groups = [df[df[target_col] == 0]["Income"], df[df[target_col] == 1]["Income"]]
    ax2.boxplot(groups, labels=["No Loan", "Took Loan"])
    ax2.set_ylabel("Income ($000)")
    st.pyplot(fig2)

    # 3) CD Account vs Loan (stacked bar - cross-sell opportunity)
    st.markdown("**3. Cross-sell hotspot: CD Account vs Personal Loan**")
    cd_col = "CDAccount" if "CDAccount" in df.columns else "CD Account"
    ct = pd.crosstab(df[cd_col], df[target_col], normalize="index")
    fig3, ax3 = plt.subplots()
    bottom = np.zeros(ct.shape[0])
    for loan_val in ct.columns:
        ax3.bar(ct.index.astype(str), ct[loan_val].values, bottom=bottom)
        bottom += ct[loan_val].values
    ax3.set_xlabel("CD Account (0=No, 1=Yes)")
    ax3.set_ylabel("Proportion")
    st.pyplot(fig3)

    # 4) Adoption rate by Income decile (line chart)
    st.markdown("**4. Income deciles vs loan uptake (Line chart)**")
    df_temp = df.copy()
    df_temp["IncomeDecile"] = pd.qcut(df_temp["Income"], 10, labels=False, duplicates="drop")
    rate_by_decile = df_temp.groupby("IncomeDecile")[target_col].mean()
    fig4, ax4 = plt.subplots()
    ax4.plot(rate_by_decile.index.astype(str), rate_by_decile.values, marker="o")
    ax4.set_xlabel("Income Decile (Low → High)")
    ax4.set_ylabel("Adoption Rate")
    st.pyplot(fig4)

    # 5) Correlation heatmap for numeric features (heatmap)
    st.markdown("**5. Behavioural drivers: Correlation heatmap**")
    fig5, ax5 = plt.subplots()
    corr = df.select_dtypes(include=[np.number]).corr()
    im = ax5.imshow(corr, aspect="auto")
    fig5.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    ax5.set_xticks(range(len(corr.columns)))
    ax5.set_xticklabels(corr.columns, rotation=90)
    ax5.set_yticks(range(len(corr.index)))
    ax5.set_yticklabels(corr.index)
    st.pyplot(fig5)


def plot_roc_curves(roc_data, summary_df):
    fig, ax = plt.subplots()
    for name, (fpr, tpr) in roc_data.items():
        auc_val = float(summary_df.loc[summary_df["Algorithm"] == name, "AUC"].values[0])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test Set)")
    ax.legend(loc="lower right")
    return fig


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No (0)", "Yes (1)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No (0)", "Yes (1)"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    return fig


def plot_feature_importances(model, feature_names, title):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), importances[idx])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=90)
    ax.set_ylabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------- Main Layout ----------------------

st.title("Universal Bank - Personal Loan Propensity Dashboard")
st.caption("Head of Marketing view: Understand, Predict & Act on Personal Loan Opportunities")

# Load data & pre-train models once
df, target_col, X_train, X_test, y_train, y_test, fitted_models, summary_df, cv_df, roc_data, cms, feature_names = get_trained_models()

tab1, tab2, tab3 = st.tabs(["📊 Customer Insights", "🤖 Model Performance", "📈 Predict New Customers"])

# ---- TAB 1: Customer Insights ----
with tab1:
    st.info("Explore customer behaviour and identify segments with high loan propensity.")
    show_customer_insights(df, target_col)

# ---- TAB 2: Model Performance ----
with tab2:
    st.info("Compare three ML models built on Universal Bank data.")
    if st.button("Run 3 Algorithms & Show Metrics"):
        st.subheader("1. Cross-Validation (5-fold) - AUC Scores")
        st.dataframe(cv_df.style.format("{:.4f}"))

        st.subheader("2. Overall Performance Summary")
        st.dataframe(summary_df.style.format("{:.4f}"))

        # ROC curves
        st.subheader("3. ROC Curve Comparison")
        fig_roc = plot_roc_curves(roc_data, summary_df)
        st.pyplot(fig_roc)

        # Confusion matrices
        st.subheader("4. Confusion Matrices (Train & Test)")
        for name in fitted_models.keys():
            st.markdown(f"**{name} - Train**")
            st.pyplot(plot_confusion_matrix(cms[name]["train"], f"{name} - Train"))
            st.markdown(f"**{name} - Test**")
            st.pyplot(plot_confusion_matrix(cms[name]["test"], f"{name} - Test"))

        # Feature importance
        st.subheader("5. Feature Importances")
        for name, model in fitted_models.items():
            fig_fi = plot_feature_importances(model, feature_names, f"{name} - Feature Importances")
            if fig_fi is not None:
                st.pyplot(fig_fi)
    else:
        st.caption("Click the button above to generate metrics & diagnostic plots.")

# ---- TAB 3: Predict New Customers ----
with tab3:
    st.info("Upload a new customer file to score personal loan propensity using the best model (Gradient Boosting).")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        scored_df = new_df.copy()

        # Drop ID-like columns
        for id_col in ["ID", "Id", "id"]:
            if id_col in scored_df.columns:
                scored_df = scored_df.drop(columns=[id_col])

        # Drop target if present
        for c in list(scored_df.columns):
            if c.strip().lower().replace(" ", "") == "personalloan":
                scored_df = scored_df.drop(columns=[c])

        # Ensure all required features exist
        for col in feature_names:
            if col not in scored_df.columns:
                scored_df[col] = 0

        # Keep only training features
        scored_df = scored_df[feature_names]

        # Impute
        for c in scored_df.columns:
            if scored_df[c].dtype.kind in "biufc":
                scored_df[c] = scored_df[c].fillna(scored_df[c].median())
            else:
                scored_df[c] = scored_df[c].fillna(scored_df[c].mode().iloc[0])

        gb_model = fitted_models["Gradient Boosting"]
        prob = gb_model.predict_proba(scored_df)[:, 1]
        pred = (prob >= 0.5).astype(int)

        output = new_df.copy()
        output["Personal Loan Probability"] = prob
        output["Personal Loan Predicted"] = pred

        st.subheader("Scored Sample")
        st.dataframe(output.head(20))

        csv_bytes = output.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Scored File (CSV)",
            data=csv_bytes,
            file_name="scored_personal_loan_customers.csv",
            mime="text/csv"
        )

        st.markdown(f"**High Propensity (P ≥ 0.6):** {int((prob >= 0.6).sum())}")
        st.markdown(f"**Very High Propensity (P ≥ 0.8):** {int((prob >= 0.8).sum())}")
    else:
        st.caption("Upload a CSV with similar schema as UniversalBank.csv to get predictions.")
