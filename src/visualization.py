"""
Visualization Module
====================
All charts and plots for the fraud detection dashboard.
Designed for both standalone use and Streamlit embedding.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional
import io
import os

# Color palette consistent with fintech aesthetic
PALETTE = {
    'fraud': '#ef4444',      # Red
    'legitimate': '#22c55e', # Green
    'suspicious': '#f59e0b', # Amber
    'primary': '#3b82f6',    # Blue
    'dark': '#1e293b',       # Dark slate
    'grid': '#e2e8f0',       # Light gray
    'bg': '#f8fafc'          # Near white
}

plt.rcParams.update({
    'figure.facecolor': PALETTE['bg'],
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.color': PALETTE['grid'],
    'grid.linewidth': 0.5,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, name: str, dpi: int = 120):
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_class_distribution(df: pd.DataFrame) -> str:
    """Fraud vs Legitimate transaction distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Transaction Class Distribution', fontsize=14, fontweight='bold', color=PALETTE['dark'])
    
    counts = df['Class'].value_counts()
    labels = ['Legitimate', 'Fraud']
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = [PALETTE['legitimate'], PALETTE['fraud']]
    
    # Bar chart
    bars = ax1.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                 f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Number of Transactions', fontsize=11)
    ax1.set_title('Count by Class', fontsize=12, pad=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    # Donut chart
    wedge_props = {'width': 0.4, 'edgecolor': 'white', 'linewidth': 2}
    ax2.pie(values, labels=labels, colors=colors, autopct='%1.3f%%',
            startangle=90, wedgeprops=wedge_props, textprops={'fontsize': 11},
            pctdistance=0.75)
    ax2.set_title('Class Balance', fontsize=12, pad=10)
    
    plt.tight_layout()
    return _save(fig, 'class_distribution')


def plot_amount_distribution(df: pd.DataFrame) -> str:
    """Transaction amount distribution by class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Transaction Amount Analysis', fontsize=14, fontweight='bold', color=PALETTE['dark'])
    
    legit = df[df['Class'] == 0]['Amount']
    fraud = df[df['Class'] == 1]['Amount']
    
    # Log-scale histogram
    ax = axes[0]
    bins = np.logspace(np.log10(max(df['Amount'].min(), 0.01)), np.log10(df['Amount'].max()), 50)
    ax.hist(legit, bins=bins, alpha=0.6, color=PALETTE['legitimate'], label='Legitimate', density=True)
    ax.hist(fraud, bins=bins, alpha=0.6, color=PALETTE['fraud'], label='Fraud', density=True)
    ax.set_xscale('log')
    ax.set_xlabel('Transaction Amount ($)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Amount Distribution (log scale)', fontsize=12)
    ax.legend()
    
    # Box plot comparison
    ax2 = axes[1]
    data_to_plot = [legit.clip(upper=legit.quantile(0.99)),
                    fraud.clip(upper=fraud.quantile(0.99))]
    bp = ax2.boxplot(data_to_plot, labels=['Legitimate', 'Fraud'],
                     patch_artist=True, widths=0.4,
                     medianprops=dict(color='white', linewidth=2))
    bp['boxes'][0].set_facecolor(PALETTE['legitimate'])
    bp['boxes'][1].set_facecolor(PALETTE['fraud'])
    for patch in bp['boxes']:
        patch.set_alpha(0.7)
    ax2.set_ylabel('Transaction Amount ($)', fontsize=11)
    ax2.set_title('Amount by Class (99th percentile cap)', fontsize=12)
    
    plt.tight_layout()
    return _save(fig, 'amount_distribution')


def plot_time_trends(df: pd.DataFrame) -> str:
    """Fraud rate over time (hourly bins)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Temporal Analysis', fontsize=14, fontweight='bold', color=PALETTE['dark'])
    
    df = df.copy()
    df['hour'] = (df['Time'] / 3600) % 24
    df['time_bin'] = (df['Time'] / 3600).astype(int)  # 1-hour bins
    
    # Hourly transaction volume
    ax1 = axes[0]
    hourly = df.groupby('time_bin').agg(
        total=('Class', 'count'),
        fraud=('Class', 'sum')
    ).reset_index()
    hourly['legit'] = hourly['total'] - hourly['fraud']
    
    x = hourly['time_bin']
    ax1.fill_between(x, hourly['legit'], alpha=0.5, color=PALETTE['legitimate'], label='Legitimate')
    ax1.fill_between(x, hourly['fraud'], alpha=0.8, color=PALETTE['fraud'], label='Fraud')
    ax1.set_xlabel('Time (hours from start)', fontsize=11)
    ax1.set_ylabel('Transaction Count', fontsize=11)
    ax1.set_title('Transaction Volume Over Time', fontsize=12)
    ax1.legend()
    
    # Fraud rate by hour of day
    ax2 = axes[1]
    df['hour_int'] = df['hour'].astype(int)
    hourly_rate = df.groupby('hour_int').agg(
        fraud_rate=('Class', 'mean'),
        count=('Class', 'count')
    ).reset_index()
    
    bar_colors = [PALETTE['fraud'] if r > 0.005 else PALETTE['legitimate']
                  for r in hourly_rate['fraud_rate']]
    bars = ax2.bar(hourly_rate['hour_int'], hourly_rate['fraud_rate'] * 100,
                   color=bar_colors, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Hour of Day', fontsize=11)
    ax2.set_ylabel('Fraud Rate (%)', fontsize=11)
    ax2.set_title('Fraud Rate by Hour of Day', fontsize=12)
    ax2.set_xticks(range(0, 24))
    
    plt.tight_layout()
    return _save(fig, 'time_trends')


def plot_confusion_matrix(cm: list, model_name: str) -> str:
    """Annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    cm_arr = np.array(cm)
    labels = np.array([
        [f'TN\n{cm_arr[0,0]:,}', f'FP\n{cm_arr[0,1]:,}'],
        [f'FN\n{cm_arr[1,0]:,}', f'TP\n{cm_arr[1,1]:,}']
    ])
    
    # Custom colormap: green for correct, red for incorrect
    custom_cm = [[cm_arr[0,0], -cm_arr[0,1]], [-cm_arr[1,0], cm_arr[1,1]]]
    
    sns.heatmap(cm_arr, annot=labels, fmt='', cmap='RdYlGn',
                xticklabels=['Predicted: Legit', 'Predicted: Fraud'],
                yticklabels=['Actual: Legit', 'Actual: Fraud'],
                linewidths=2, linecolor='white',
                annot_kws={'size': 13, 'weight': 'bold'}, ax=ax,
                cbar_kws={'label': 'Count'})
    
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold',
                 color=PALETTE['dark'], pad=15)
    ax.set_xlabel('Predicted Class', fontsize=11, labelpad=10)
    ax.set_ylabel('Actual Class', fontsize=11, labelpad=10)
    
    plt.tight_layout()
    return _save(fig, f'confusion_matrix_{model_name.lower().replace(" ", "_")}')


def plot_roc_curves(metrics_list: list) -> str:
    """ROC curves for all models on one plot."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    colors = [PALETTE['primary'], PALETTE['fraud'], '#8b5cf6', '#06b6d4']
    
    for i, metrics in enumerate(metrics_list):
        fpr = metrics['roc_curve']['fpr']
        tpr = metrics['roc_curve']['tpr']
        auc = metrics['roc_auc']
        name = metrics['model_name']
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC = {auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold', color=PALETTE['dark'])
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    return _save(fig, 'roc_curves')


def plot_pr_curves(metrics_list: list) -> str:
    """Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    colors = [PALETTE['primary'], PALETTE['fraud'], '#8b5cf6', '#06b6d4']
    
    for i, metrics in enumerate(metrics_list):
        prec = metrics['pr_curve']['precision']
        rec = metrics['pr_curve']['recall']
        pr_auc = metrics['pr_auc']
        name = metrics['model_name']
        color = colors[i % len(colors)]
        ax.plot(rec, prec, lw=2, color=color, label=f'{name} (PR-AUC = {pr_auc:.4f})')
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — Model Comparison', fontsize=14, fontweight='bold', color=PALETTE['dark'])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold', color=PALETTE['dark'])
    
    plt.tight_layout()
    return _save(fig, 'pr_curves')


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, model_name: str = 'Model') -> str:
    """Horizontal bar chart of feature importances."""
    df = importance_df.head(top_n).copy()
    df = df.sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    
    # Color code by feature type
    colors = []
    for feat in df['feature']:
        if feat.startswith('V'):
            colors.append(PALETTE['primary'])
        elif 'Amount' in feat or 'amount' in feat:
            colors.append(PALETTE['fraud'])
        else:
            colors.append('#8b5cf6')
    
    bars = ax.barh(df['feature'], df['importance_pct'], color=colors,
                   edgecolor='white', linewidth=0.5)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.05, bar.get_y() + bar.get_height()/2,
                f'{w:.2f}%', va='center', fontsize=9)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=PALETTE['primary'], label='PCA Features (V1-V28)'),
        mpatches.Patch(color=PALETTE['fraud'], label='Amount Features'),
        mpatches.Patch(color='#8b5cf6', label='Engineered Features'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
    
    ax.set_xlabel('Feature Importance (%)', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importances — {model_name}', fontsize=13,
                 fontweight='bold', color=PALETTE['dark'])
    
    plt.tight_layout()
    return _save(fig, f'feature_importance_{model_name.lower().replace(" ", "_")}')


def plot_model_comparison(metrics_list: list) -> str:
    """Side-by-side bar chart comparing model metrics."""
    models = [m['model_name'] for m in metrics_list]
    metric_names = ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    metric_labels = ['Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC']
    
    x = np.arange(len(models))
    n_metrics = len(metric_names)
    width = 0.7 / n_metrics
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bar_colors = [PALETTE['primary'], PALETTE['fraud'], '#8b5cf6', '#06b6d4', '#f59e0b']
    
    for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        values = [m[metric] for m in metrics_list]
        offset = (i - n_metrics/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label,
                      color=bar_colors[i], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', color=PALETTE['dark'])
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    return _save(fig, 'model_comparison')


def plot_score_distribution(y_prob: np.ndarray, y_true: np.ndarray,
                             model_name: str, threshold: float = 0.5) -> str:
    """Fraud score distribution histogram with threshold line."""
    fig, ax = plt.subplots(figsize=(11, 5))
    
    scores_legit = y_prob[y_true == 0] * 100
    scores_fraud = y_prob[y_true == 1] * 100
    
    bins = np.linspace(0, 100, 51)
    ax.hist(scores_legit, bins=bins, alpha=0.6, color=PALETTE['legitimate'],
            label=f'Legitimate (n={len(scores_legit):,})', density=True)
    ax.hist(scores_fraud, bins=bins, alpha=0.7, color=PALETTE['fraud'],
            label=f'Fraud (n={len(scores_fraud):,})', density=True)
    
    # Threshold line
    ax.axvline(x=threshold*100, color=PALETTE['dark'], linestyle='--', lw=2,
               label=f'Threshold ({threshold*100:.0f})')
    
    # Risk zones
    ax.axvspan(0, 30, alpha=0.05, color=PALETTE['legitimate'], label='Safe Zone (0-30)')
    ax.axvspan(30, 70, alpha=0.05, color=PALETTE['suspicious'], label='Suspicious Zone (30-70)')
    ax.axvspan(70, 100, alpha=0.05, color=PALETTE['fraud'], label='Fraud Zone (70-100)')
    
    ax.set_xlabel('Fraud Score (0-100)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Fraud Score Distribution — {model_name}', fontsize=13,
                 fontweight='bold', color=PALETTE['dark'])
    ax.legend(fontsize=9, ncol=2)
    
    plt.tight_layout()
    return _save(fig, f'score_distribution_{model_name.lower().replace(" ", "_")}')


def plot_local_explanation(explanation: dict, feature_names: list) -> str:
    """Waterfall-style local explanation chart for a single transaction."""
    top = explanation['top_features'][:10]
    features = [t['feature'] for t in top]
    contribs = [t['contribution'] for t in top]
    values = [t['value'] for t in top]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [PALETTE['fraud'] if c > 0 else PALETTE['legitimate'] for c in contribs]
    bars = ax.barh(features, contribs, color=colors, alpha=0.8, edgecolor='white')
    
    for bar, val, c in zip(bars, values, contribs):
        label = f'value={val:.3f}'
        x_pos = bar.get_width() + 0.002 if c >= 0 else bar.get_width() - 0.002
        ha = 'left' if c >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8, ha=ha)
    
    ax.axvline(x=0, color=PALETTE['dark'], linewidth=1.5)
    ax.set_xlabel('Feature Contribution (correlation with fraud probability)', fontsize=11)
    ax.set_title(
        f"Transaction Explanation\nFraud Score: {explanation['fraud_score']:.1f}/100 | "
        f"Risk: {explanation['risk_level']}",
        fontsize=12, fontweight='bold', color=PALETTE['dark']
    )
    
    plt.tight_layout()
    return _save(fig, 'local_explanation')


def plot_threshold_analysis(y_prob: np.ndarray, y_true: np.ndarray,
                             model_name: str = 'Model') -> str:
    """Show Precision/Recall/F1 across different threshold values."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, f1s = [], [], []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(thresholds, precisions, color=PALETTE['primary'], lw=2, label='Precision')
    ax.plot(thresholds, recalls, color=PALETTE['fraud'], lw=2, label='Recall')
    ax.plot(thresholds, f1s, color='#8b5cf6', lw=2.5, label='F1 Score', linestyle='--')
    
    # Mark optimal F1 threshold
    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    ax.axvline(x=best_t, color=PALETTE['dark'], linestyle=':', lw=1.5,
               label=f'Best F1 threshold ({best_t:.2f})')
    ax.plot(best_t, f1s[best_idx], 'o', color='#8b5cf6', markersize=10)
    
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Threshold Analysis — {model_name}', fontsize=14,
                 fontweight='bold', color=PALETTE['dark'])
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    return _save(fig, f'threshold_analysis_{model_name.lower().replace(" ", "_")}')
