import os
import glob
import time
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    balanced_accuracy_score
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import display

warnings.filterwarnings("ignore")

# ---------------------------
# 2. Global plotting style
# ---------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["figure.dpi"] = 130

# ---------------------------
# 3. Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# 4. File discovery
# ---------------------------
def auto_find_unsw_files():
    search_roots = [
        ".",
        "/content",
        "/content/drive/MyDrive",
    ]

    train_candidates = []
    test_candidates = []

    for root in search_roots:
        if os.path.exists(root):
            train_candidates.extend(glob.glob(os.path.join(root, "**", "*training*.csv"), recursive=True))
            test_candidates.extend(glob.glob(os.path.join(root, "**", "*testing*.csv"), recursive=True))

    train_candidates = sorted(list(set(train_candidates)))
    test_candidates = sorted(list(set(test_candidates)))

    train_path = train_candidates[0] if train_candidates else None
    test_path = test_candidates[0] if test_candidates else None

    return train_path, test_path

# ---------------------------
# 5. Autoencoder model
# ---------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden1 = max(64, input_dim // 2)
        hidden2 = max(32, input_dim // 4)
        bottleneck = max(16, input_dim // 8)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x);
        out = self.decoder(z)
        return out

@dataclass
class AutoencoderConfig:
    epochs: int = 18
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5

def train_autoencoder(X_fit, X_val, config: AutoencoderConfig, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    model = Autoencoder(X_fit.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    n = X_fit_t.shape[0]

    for epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0

        for i in range(0, n, config.batch_size):
            idx = perm[i:i+config.batch_size]
            batch = X_fit_t[idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= n
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_t)
            val_loss = criterion(val_recon, X_val_t).item()
        val_losses.append(val_loss)

    return model, train_losses, val_losses, device

def autoencoder_scores(model, X, device):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(X_t)
        mse = ((X_t - recon) ** 2).mean(dim=1).cpu().numpy()
    return mse

# ---------------------------
# 6. Preprocessing
# ---------------------------
def build_preprocessor(df, categorical_features, numeric_features):
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, categorical_features),
        ("num", num_pipe, numeric_features)
    ], remainder="drop")

    return preprocessor

# ---------------------------
# 7. Metrics
# ---------------------------
def compute_metrics(y_true, y_pred, scores):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "avg_precision": ap,
        "fpr": fpr,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

# ---------------------------
# 8. Model-specific scoring
# ---------------------------
def fit_and_score_model(model_name, X_fit, X_val, X_test, seed):
    set_seed(seed)

    fit_start = time.perf_counter()

    if model_name == "IsolationForest":
        model = IsolationForest(
            n_estimators=250,
            contamination="auto",
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_fit)
        fit_time = time.perf_counter() - fit_start

        score_start = time.perf_counter()
        s_val = -model.decision_function(X_val)
        s_test = -model.decision_function(X_test)
        score_time = time.perf_counter() - score_start

        meta = {"model": model}
        return s_val, s_test, fit_time, score_time, meta

    elif model_name == "OneClassSVM":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        model.fit(X_fit)
        fit_time = time.perf_counter() - fit_start

        score_start = time.perf_counter()
        s_val = -model.decision_function(X_val).ravel()
        s_test = -model.decision_function(X_test).ravel()
        score_time = time.perf_counter() - score_start

        meta = {"model": model}
        return s_val, s_test, fit_time, score_time, meta

    elif model_name == "LocalOutlierFactor":
        model = LocalOutlierFactor(
            n_neighbors=35,
            novelty=True,
            contamination="auto",
            n_jobs=-1
        )
        model.fit(X_fit)
        fit_time = time.perf_counter() - fit_start

        score_start = time.perf_counter()
        s_val = -model.decision_function(X_val)
        s_test = -model.decision_function(X_test)
        score_time = time.perf_counter() - score_start

        meta = {"model": model}
        return s_val, s_test, fit_time, score_time, meta

    elif model_name == "Autoencoder":
        ae_cfg = AutoencoderConfig()
        model, train_losses, val_losses, device = train_autoencoder(X_fit, X_val, ae_cfg, seed=seed)
        fit_time = time.perf_counter() - fit_start

        score_start = time.perf_counter()
        s_val = autoencoder_scores(model, X_val, device)
        s_test = autoencoder_scores(model, X_test, device)
        score_time = time.perf_counter() - score_start

        meta = {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        return s_val, s_test, fit_time, score_time, meta

    else:
        raise ValueError(f"Unknown model: {model_name}")

# ---------------------------
# 9. Pretty printing helper
# ---------------------------
def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

# ---------------------------
# 10. High-quality plots
# ---------------------------
def plot_dataset_overview(test_df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    label_counts = test_df["label"].value_counts().sort_index()
    label_names = ["Normal (0)", "Attack (1)"]
    axes[0].pie(
        label_counts.values,
        labels=label_names,
        autopct="%1.1f%%",
        startangle=120,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    axes[0].set_title("Test Set Class Distribution")

    if "attack_cat" in test_df.columns:
        attack_df = test_df[test_df["label"] == 1]["attack_cat"].fillna("Unknown")
        top_attack = attack_df.value_counts().head(12)
        sns.barplot(
            x=top_attack.values,
            y=top_attack.index,
            ax=axes[1]
        )
        axes[1].set_title("Top Attack Categories in Test Set")
        axes[1].set_xlabel("Count")
        axes[1].set_ylabel("Attack Category")

    plt.tight_layout()
    plt.show()

def plot_threshold_sensitivity(summary_df):
    metrics = ["fpr", "precision", "recall", "f1", "balanced_accuracy"]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.ravel()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.lineplot(
            data=summary_df,
            x="threshold_percentile",
            y=f"{metric}_mean",
            hue="model",
            marker="o",
            linewidth=2.6,
            ax=ax
        )
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Threshold")
        ax.set_xlabel("Threshold Percentile")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    plt.tight_layout()
    plt.show()

def plot_runtime_comparison(summary_df):
    rt = summary_df.groupby("model", as_index=False)[["fit_time_mean", "score_time_mean"]].mean()
    rt_m = rt.melt(id_vars="model", var_name="runtime_type", value_name="seconds")

    plt.figure(figsize=(13, 7))
    sns.barplot(data=rt_m, x="model", y="seconds", hue="runtime_type")
    plt.title("Runtime Comparison Across Models")
    plt.xlabel("Model")
    plt.ylabel("Seconds")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

def plot_stability_boxplots(results_df):
    metrics = ["precision", "recall", "f1", "fpr", "balanced_accuracy"]
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.ravel()

    for i, metric in enumerate(metrics):
        sns.boxplot(
            data=results_df,
            x="model",
            y=metric,
            hue="threshold_percentile",
            ax=axes[i]
        )
        axes[i].set_title(f"Run Stability: {metric.replace('_', ' ').title()}")
        axes[i].tick_params(axis="x", rotation=15)
        axes[i].legend(title="Threshold", fontsize=9)

    axes[-1].axis("off")
    plt.tight_layout()
    plt.show()

def plot_score_distributions(best_result_rows, score_dict, y_test):
    n_models = len(best_result_rows)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, best_result_rows.iterrows()):
        model = row["model"]
        thr = row["threshold"]
        s_test = score_dict[model]["s_test"]

        score_df = pd.DataFrame({
            "score": s_test,
            "class": np.where(y_test == 1, "Attack", "Normal")
        })

        sns.kdeplot(
            data=score_df,
            x="score",
            hue="class",
            fill=True,
            common_norm=False,
            alpha=0.35,
            linewidth=2,
            ax=ax
        )
        ax.axvline(thr, linestyle="--", linewidth=2, label=f"Threshold = {thr:.4f}")
        ax.set_title(f"Anomaly Score Distribution - {model}")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(best_result_rows, best_pred_dict, y_test):
    n_models = len(best_result_rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, best_result_rows.iterrows()):
        model = row["model"]
        y_pred = best_pred_dict[model]
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Normal", "Pred Attack"],
            yticklabels=["True Normal", "True Attack"]
        )
        ax.set_title(f"Confusion Matrix - {model}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

def plot_roc_curves(best_result_rows, score_dict, y_test):
    plt.figure(figsize=(12, 8))
    for _, row in best_result_rows.iterrows():
        model = row["model"]
        s_test = score_dict[model]["s_test"]
        auc = roc_auc_score(y_test, s_test)
        fpr, tpr, _ = roc_curve(y_test, s_test);
        plt.plot(fpr, tpr, linewidth=2.5, label=f"{model} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.8)
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pr_curves(best_result_rows, score_dict, y_test):
    plt.figure(figsize=(12, 8))
    for _, row in best_result_rows.iterrows():
        model = row["model"]
        s_test = score_dict[model]["s_test"]
        ap = average_precision_score(y_test, s_test)
        precision, recall, _ = precision_recall_curve(y_test, s_test)
        plt.plot(recall, precision, linewidth=2.5, label=f"{model} (AP={ap:.3f})")

    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True, alpha=0.3);
    plt.tight_layout()
    plt.show()

def plot_autoencoder_loss(ae_meta):
    if ae_meta is None:
        return
    plt.figure(figsize=(11, 6))
    plt.plot(ae_meta["train_losses"], marker="o", linewidth=2.3, label="Train Loss")
    plt.plot(ae_meta["val_losses"], marker="s", linewidth=2.3, label="Validation Loss")
    plt.title("Autoencoder Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_attack_detection_heatmap(best_result_rows, best_pred_dict, test_df):
    if "attack_cat" not in test_df.columns:
        return

    attack_only = test_df[test_df["label"] == 1].copy()
    if attack_only.empty:
        return

    heatmap_data = {}
    for _, row in best_result_rows.iterrows():
        model = row["model"]
        y_pred = pd.Series(best_pred_dict[model], index=test_df.index);
        attack_pred = y_pred.loc[attack_only.index]

        tmp = pd.DataFrame({
            "attack_cat": attack_only["attack_cat"].fillna("Unknown"),
            "pred": attack_pred.values
        })
        rates = tmp.groupby("attack_cat")["pred"].mean().sort_values(ascending=False)
        heatmap_data[model] = rates

    heatmap_df = pd.DataFrame(heatmap_data).fillna(0.0)
    heatmap_df = heatmap_df.sort_values(by=heatmap_df.columns[0], ascending=False)

    plt.figure(figsize=(13, max(7, 0.45 * len(heatmap_df))))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title("Detection Rate by Attack Category and Model")
    plt.xlabel("Model")
    plt.ylabel("Attack Category")
    plt.tight_layout()
    plt.show()

def plot_metric_correlation(results_df):
    corr_cols = ["precision", "recall", "f1", "roc_auc", "avg_precision", "fpr", "balanced_accuracy"]
    corr = results_df[corr_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Between Evaluation Metrics")
    plt.tight_layout()
    plt.show()

def plot_best_config_scatter(best_result_rows):
    plt.figure(figsize=(12, 8))
    print(best_result_rows.columns)
    print(best_result_rows.head())
    sns.scatterplot(
        data=best_result_rows,
        x="fpr",
        y="recall",
        hue="model",
        size="f1",
        sizes=(200, 650),
        style="model",
        s=260
    )
    for _, row in best_result_rows.iterrows():
        plt.text(row["fpr"] + 0.001, row["recall"], row["model"], fontsize=11);
    plt.title("Trade-off Plot: Recall vs FPR (Bubble Size = F1)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recall")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------------------------
# 11. Main experiment
# ---------------------------
def run_experiment(
    train_path=None,
    test_path=None,
    runs=5,
    thresholds=(90, 95, 97.5, 99),
    validation_size=0.2,
    best_rule="best_f1_under_low_fpr",
    max_train_normal_samples=None
):
    if train_path is None or test_path is None:
        auto_train, auto_test = auto_find_unsw_files()
        train_path = train_path or auto_train
        test_path = test_path or auto_test

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Could not auto-detect UNSW-NB15 train/test CSV files. "
            "Please pass train_path and test_path explicitly."
        )

    print_section("DATA LOADING")
    print("Train file:", train_path)
    print("Test file: ", test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape: ", test_df.shape)

    required_cols = {"label"}
    missing_train = required_cols - set(train_df.columns)
    missing_test = required_cols - set(test_df.columns)
    if missing_train or missing_test:
        raise ValueError(f"Missing required label column. Train missing: {missing_train}, Test missing: {missing_test}")

    drop_cols = [c for c in ["id", "label", "attack_cat"] if c in train_df.columns];
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    categorical_features = [c for c in ["proto", "service", "state"] if c in feature_cols]
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["label"].astype(int).copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df["label"].astype(int).copy()

    print("Number of features:", len(feature_cols))
    print("Categorical features:", categorical_features)
    print("Numeric features:", len(numeric_features))

    plot_dataset_overview(test_df)

    # Normal-only training
    X_train_normal = X_train[y_train == 0].copy()

    if max_train_normal_samples is not None and max_train_normal_samples < len(X_train_normal):
        X_train_normal = X_train_normal.sample(max_train_normal_samples, random_state=42)
        print(f"Reduced normal training size to {len(X_train_normal)} rows for faster Colab execution.")

    print_section("NORMAL-ONLY TRAINING SETUP")
    print("Normal-only train shape:", X_train_normal.shape)

    X_fit_raw, X_val_raw = train_test_split(
        X_train_normal,
        test_size=validation_size,
        random_state=42,
        shuffle=True
    )

    preprocessor = build_preprocessor(X_train_normal, categorical_features, numeric_features);
    X_fit = preprocessor.fit_transform(X_fit_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test_p = preprocessor.transform(X_test)

    print("Transformed train-fit shape:", X_fit.shape)
    print("Transformed validation shape:", X_val.shape)
    print("Transformed test shape:", X_test_p.shape)

    model_names = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "Autoencoder"]

    results = []
    score_dict = {}
    ae_meta = None

    print_section("MODEL TRAINING + THRESHOLDING")
    for model_name in model_names:
        print(f"\nRunning model: {model_name}")

        for run in range(1, runs + 1):
            print(f"  Run {run}/{runs}")
            s_val, s_test, fit_time, score_time, meta = fit_and_score_model(
                model_name=model_name,
                X_fit=X_fit,
                X_val=X_val,
                X_test=X_test_p,
                seed=run
            )

            if model_name not in score_dict:
                score_dict[model_name] = {"s_test": s_test, "s_val": s_val}

            if model_name == "Autoencoder" and run == 1:
                ae_meta = meta

            for t in thresholds:
                thr = np.percentile(s_val, t)
                y_pred = (s_test >= thr).astype(int)
                metrics = compute_metrics(y_test.values, y_pred, s_test)

                results.append({
                    "model": model_name,
                    "run": run,
                    "threshold_percentile": t,
                    "threshold": thr,
                    "fit_time": fit_time,
                    "score_time": score_time,
                    **metrics
                })

    results_df = pd.DataFrame(results)

    summary_df = (
        results_df
        .groupby(["model", "threshold_percentile"], as_index=False)
        .agg({
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
            "roc_auc": ["mean", "std"],
            "avg_precision": ["mean", "std"],
            "fpr": ["mean", "std"],
            "specificity": ["mean", "std"],
            "balanced_accuracy": ["mean", "std"],
            "fit_time": ["mean", "std"],
            "score_time": ["mean", "std"],
        })
    )
    summary_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary_df.columns.values
    ]

    # Ranking rule
    if best_rule == "best_f1_under_low_fpr":
        eligible = summary_df[summary_df["fpr_mean"] <= 0.10].copy();
        if eligible.empty:
            eligible = summary_df.copy()
        ranking_df = eligible.sort_values(
            by=["f1_mean", "recall_mean", "fpr_mean"],
            ascending=[False, False, True]
        ).reset_index(drop=True)
    elif best_rule == "lowest_fpr_at_acceptable_recall":
        eligible = summary_df[summary_df["recall_mean"] >= 0.60].copy()
        if eligible.empty:
            eligible = summary_df.copy()
        ranking_df = eligible.sort_values(
            by=["fpr_mean", "recall_mean", "f1_mean"],
            ascending=[True, False, False]
        ).reset_index(drop=True)
    else:
        summary_df["tradeoff_score"] = (
            0.45 * summary_df["f1_mean"] +
            0.30 * summary_df["recall_mean"] +
            0.20 * summary_df["roc_auc_mean"] -
            0.25 * summary_df["fpr_mean"]
        )
        ranking_df = summary_df.sort_values("tradeoff_score", ascending=False).reset_index(drop=True);

    best_rows = []
    for model_name in model_names:
        subset = ranking_df[ranking_df["model"] == model_name].copy()
        if not subset.empty:
            best_rows.append(subset.iloc[0])
    best_configs_df = pd.DataFrame(best_rows).reset_index(drop=True)

    # Calculate the actual threshold for each best configuration
    best_configs_df["threshold"] = best_configs_df.apply(
        lambda row: np.percentile(score_dict[row["model"]]["s_val"], row["threshold_percentile"]),
        axis=1
    )

    # Regenerate best predictions for confusion matrices and attack heatmap
    best_pred_dict = {}
    for _, row in best_configs_df.iterrows():
        model_name = row["model"]
        # The 'threshold' column is now available in best_configs_df
        threshold_val = row["threshold"]

        # use run 1 stored scores as representative for plots
        s_test = score_dict[model_name]["s_test"]
        s_val = score_dict[model_name]["s_val"]

        # Use the derived threshold_val for predictions
        best_pred_dict[model_name] = (s_test >= threshold_val).astype(int);

    # Console tables
    print_section("DETAILED RUN RESULTS")
    display(results_df.head(20))

    print_section("SUMMARY RESULTS (MEAN ± STD)")
    display(summary_df)

    print_section("RANKING TABLE")
    display(ranking_df)

    print_section("BEST CONFIG PER MODEL")
    display(best_configs_df)

    # Plots
    print_section("VISUAL ANALYSIS")
    plot_threshold_sensitivity(summary_df)
    plot_runtime_comparison(summary_df)
    plot_stability_boxplots(results_df)
    plot_roc_curves(best_configs_df, score_dict, y_test.values)
    plot_pr_curves(best_configs_df, score_dict, y_test.values)
    plot_score_distributions(best_configs_df, score_dict, y_test.values)
    plot_confusion_matrices(best_configs_df, best_pred_dict, y_test.values)
    plot_attack_detection_heatmap(best_configs_df, best_pred_dict, test_df)
    plot_metric_correlation(results_df)
    plot_best_config_scatter(best_configs_df)
    plot_autoencoder_loss(ae_meta)

    print_section("TOP RECOMMENDATION")
    top = ranking_df.iloc[0]
    print(f"Recommended model: {top['model']}")
    print(f"Threshold percentile: {top['threshold_percentile']}")
    print(f"Mean F1: {top['f1_mean']:.4f}")
    print(f"Mean Recall: {top['recall_mean']:.4f}")
    print(f"Mean FPR: {top['fpr_mean']:.4f}")
    print(f"Mean ROC-AUC: {top['roc_auc_mean']:.4f}")

    return results_df, summary_df, ranking_df, best_configs_df

# ---------------------------
# 12. Example usage in Colab
# ---------------------------
# Option A: auto-detect files
# results_df, summary_df, ranking_df, best_configs_df = run_experiment()

# Option B: provide exact paths
# results_df, summary_df, ranking_df, best_configs_df = run_experiment(
#     train_path="/content/UNSW_NB15_training-set (1).csv",
#     test_path="/content/UNSW_NB15_testing-set (1).csv",
#     runs=5,
#     thresholds=(90, 95, 97.5, 99),
#     max_train_normal_samples=80000  # optional, to speed up Colab
# )

if __name__ == "__main__":
    results_df, summary_df, ranking_df, best_configs_df = run_experiment()