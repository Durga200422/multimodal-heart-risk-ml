#!/usr/bin/env python3
"""
Comprehensive Model Comparison: Traditional ML vs Hybrid Models

This script evaluates and compares:
1. Traditional ML models (from existing results)
2. Hybrid ML+DL (Deep Neural Network meta-learner)
3. Hybrid ML+GA (Genetic Algorithm weight optimization)

Generates comparison visualizations and tables.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data_prep import load_scenario_a, load_scenario_b
from src.eval_utils import compute_classification_metrics

# Import hybrid models
from src.hybrid_ml_dl import HybridMLDL, train_and_evaluate_hybrid_scenario_a as train_dl_a
from src.hybrid_ml_dl import train_and_evaluate_hybrid_scenario_b as train_dl_b
from src.hybrid_ml_ga import HybridMLGA, train_and_evaluate_ga_scenario_a as train_ga_a
from src.hybrid_ml_ga import train_and_evaluate_ga_scenario_b as train_ga_b

RESULTS_DIR_A = Path(__file__).resolve().parents[1] / "results" / "scenario_a"
RESULTS_DIR_B = Path(__file__).resolve().parents[1] / "results" / "scenario_b"


def load_existing_metrics(scenario: str) -> pd.DataFrame:
    """Load metrics from existing model evaluations."""
    if scenario == "a":
        csv_path = RESULTS_DIR_A / "metrics_scenario_a.csv"
    else:
        csv_path = RESULTS_DIR_B / "metrics_scenario_b.csv"
    
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0)
    else:
        return pd.DataFrame()


def create_comprehensive_comparison(scenario: str):
    """
    Create comprehensive comparison of all models for a given scenario.
    """
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE COMPARISON - SCENARIO {scenario.upper()}")
    print('='*70)
    
    # Load existing metrics
    df_existing = load_existing_metrics(scenario)
    
    # Load hybrid metrics
    if scenario == "a":
        hybrid_dl_csv = RESULTS_DIR_A / "hybrid_ml_dl_metrics.csv"
        hybrid_ga_csv = RESULTS_DIR_A / "hybrid_ml_ga_metrics.csv"
    else:
        hybrid_dl_csv = RESULTS_DIR_B / "hybrid_ml_dl_metrics.csv"
        hybrid_ga_csv = RESULTS_DIR_B / "hybrid_ml_ga_metrics.csv"
    
    # Combine all metrics
    dfs = [df_existing]
    
    if hybrid_dl_csv.exists():
        df_dl = pd.read_csv(hybrid_dl_csv, index_col=0)
        dfs.append(df_dl)
    
    if hybrid_ga_csv.exists():
        df_ga = pd.read_csv(hybrid_ga_csv, index_col=0)
        dfs.append(df_ga)
    
    df_all = pd.concat(dfs, axis=0)
    
    # Sort by ROC-AUC
    df_all = df_all.sort_values('roc_auc', ascending=False)
    
    print("\n📊 Complete Model Comparison:")
    print(df_all.to_string())
    
    # Save combined metrics
    if scenario == "a":
        out_path = RESULTS_DIR_A / "all_models_comparison.csv"
    else:
        out_path = RESULTS_DIR_B / "all_models_comparison.csv"
    
    df_all.to_csv(out_path)
    print(f"\n💾 Saved to: {out_path}")
    
    return df_all


def plot_model_comparison_bar(df: pd.DataFrame, scenario: str):
    """
    Create bar plot comparing all models across multiple metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Scenario {scenario.upper()} - Comprehensive Model Comparison', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['roc_auc', 'f1', 'precision', 'recall']
    metric_names = ['ROC-AUC', 'F1-Score', 'Precision', 'Recall']
    
    for ax, metric, name in zip(axes.flat, metrics, metric_names):
        if metric not in df.columns:
            continue
        
        # Sort by current metric
        df_sorted = df.sort_values(metric, ascending=True)
        
        # Create horizontal bar plot
        colors = ['#1f77b4' if 'Hybrid' not in idx else '#ff7f0e' 
                  for idx in df_sorted.index]
        
        y_pos = np.arange(len(df_sorted))
        ax.barh(y_pos, df_sorted[metric], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted.index, fontsize=9)
        ax.set_xlabel(name, fontsize=10)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(df_sorted[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', 
                   va='center', fontsize=8)
    
    plt.tight_layout()
    
    if scenario == "a":
        out_path = RESULTS_DIR_A / "comprehensive_comparison_bar.png"
    else:
        out_path = RESULTS_DIR_B / "comprehensive_comparison_bar.png"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Comparison bar plot saved to: {out_path}")


def plot_radar_chart(df: pd.DataFrame, scenario: str):
    """
    Create radar chart comparing top models across all metrics.
    """
    # Select top 5 models by ROC-AUC
    top_models = df.nlargest(5, 'roc_auc')
    
    metrics = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']
    metrics = [m for m in metrics if m in df.columns]
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_models)))
    
    for idx, (model_name, row) in enumerate(top_models.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
               color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis to go from 0 to 1
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper().replace('_', '-') for m in metrics], 
                       fontsize=10)
    ax.set_title(f'Scenario {scenario.upper()} - Top 5 Models Radar Comparison',
                size=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)
    
    plt.tight_layout()
    
    if scenario == "a":
        out_path = RESULTS_DIR_A / "radar_comparison_top5.png"
    else:
        out_path = RESULTS_DIR_B / "radar_comparison_top5.png"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Radar chart saved to: {out_path}")


def plot_performance_heatmap(df: pd.DataFrame, scenario: str):
    """
    Create heatmap of model performance across metrics.
    """
    metrics = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']
    metrics = [m for m in metrics if m in df.columns]
    
    # Create heatmap data
    heatmap_data = df[metrics].T
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
               cbar_kws={'label': 'Score'}, linewidths=0.5,
               vmin=0.5, vmax=1.0, ax=ax)
    
    ax.set_title(f'Scenario {scenario.upper()} - Performance Heatmap',
                fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Models', fontsize=11)
    ax.set_ylabel('Metrics', fontsize=11)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if scenario == "a":
        out_path = RESULTS_DIR_A / "performance_heatmap.png"
    else:
        out_path = RESULTS_DIR_B / "performance_heatmap.png"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🔥 Performance heatmap saved to: {out_path}")


def generate_summary_report(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """
    Generate text summary report of hybrid model performance.
    """
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("HYBRID MODELS PERFORMANCE SUMMARY")
    report_lines.append("="*70)
    report_lines.append("")
    
    # Scenario A
    report_lines.append("SCENARIO A (Lifestyle):")
    report_lines.append("-" * 40)
    
    if 'HybridMLDL' in df_a.index:
        roc_dl = df_a.loc['HybridMLDL', 'roc_auc']
        report_lines.append(f"  Hybrid ML+DL  ROC-AUC: {roc_dl:.4f}")
    
    if 'HybridMLGA' in df_a.index:
        roc_ga = df_a.loc['HybridMLGA', 'roc_auc']
        report_lines.append(f"  Hybrid ML+GA  ROC-AUC: {roc_ga:.4f}")
    
    best_traditional = df_a[~df_a.index.str.contains('Hybrid')].sort_values(
        'roc_auc', ascending=False
    ).iloc[0] if len(df_a[~df_a.index.str.contains('Hybrid')]) > 0 else None
    
    if best_traditional is not None:
        report_lines.append(f"  Best Traditional: {best_traditional.name} "
                          f"(ROC-AUC: {best_traditional['roc_auc']:.4f})")
    
    report_lines.append("")
    
    # Scenario B
    report_lines.append("SCENARIO B (Clinical):")
    report_lines.append("-" * 40)
    
    if 'HybridMLDL' in df_b.index:
        roc_dl = df_b.loc['HybridMLDL', 'roc_auc']
        report_lines.append(f"  Hybrid ML+DL  ROC-AUC: {roc_dl:.4f}")
    
    if 'HybridMLGA' in df_b.index:
        roc_ga = df_b.loc['HybridMLGA', 'roc_auc']
        report_lines.append(f"  Hybrid ML+GA  ROC-AUC: {roc_ga:.4f}")
    
    best_traditional = df_b[~df_b.index.str.contains('Hybrid')].sort_values(
        'roc_auc', ascending=False
    ).iloc[0] if len(df_b[~df_b.index.str.contains('Hybrid')]) > 0 else None
    
    if best_traditional is not None:
        report_lines.append(f"  Best Traditional: {best_traditional.name} "
                          f"(ROC-AUC: {best_traditional['roc_auc']:.4f})")
    
    report_lines.append("")
    report_lines.append("="*70)
    
    # Print report
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    # Save report
    report_path = Path(__file__).resolve().parents[1] / "results" / "hybrid_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n💾 Summary report saved to: {report_path}")


def main():
    """
    Main execution: Train hybrids, compare all models, generate visualizations.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE HYBRID MODEL EVALUATION")
    print("="*70)
    
    # Train hybrid models
    print("\n🚀 Training Hybrid ML+DL models...")
    hybrid_dl_a, metrics_dl_a = train_dl_a()
    hybrid_dl_b, metrics_dl_b = train_dl_b()
    
    print("\n🚀 Training Hybrid ML+GA models...")
    hybrid_ga_a, metrics_ga_a = train_ga_a()
    hybrid_ga_b, metrics_ga_b = train_ga_b()
    
    # Create comparisons
    print("\n📊 Creating comprehensive comparisons...")
    df_a = create_comprehensive_comparison("a")
    df_b = create_comprehensive_comparison("b")
    
    # Generate visualizations
    print("\n📈 Generating visualization plots...")
    
    plot_model_comparison_bar(df_a, "a")
    plot_model_comparison_bar(df_b, "b")
    
    plot_radar_chart(df_a, "a")
    plot_radar_chart(df_b, "b")
    
    plot_performance_heatmap(df_a, "a")
    plot_performance_heatmap(df_b, "b")
    
    # Generate summary report
    generate_summary_report(df_a, df_b)
    
    print("\n✅ All evaluations and visualizations complete!")
    print(f"Results saved to:")
    print(f"  - {RESULTS_DIR_A}")
    print(f"  - {RESULTS_DIR_B}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    RESULTS_DIR_A.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR_B.mkdir(parents=True, exist_ok=True)
    
    main()