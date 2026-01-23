import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math

# ==========================================
# 1. CONFIGURATION & STANDARDS
# ==========================================

# Colors
TASK_COLORS = {'binary': '#2E86AB', 'multi': '#A23B72'}
MODEL_COLORS = {'lr': '#E63946', 'rf': '#06A77D', 'xgb': '#F77F00', 'LR': '#E63946', 'RF': '#06A77D', 'XGB': '#F77F00'}
STRATEGY_COLORS = {'s0': '#6C757D', 's1': '#0077B6', 's2a': '#38B000', 'S0': '#6C757D', 'S1': '#0077B6', 'S2A': '#38B000'}
CLASS_COLORS = {
    'Normal': '#2A9D8F', 'Generic': '#E76F51', 'Exploits': '#F4A261',
    'Fuzzers': '#E9C46A', 'DoS': '#8338EC', 'Reconnaissance': '#3A86FF',
    'Analysis': '#FB5607', 'Backdoor': '#FF006E', 'Shellcode': '#FFBE0B',
    'Worms': '#D62828'
}

# Fonts & Sizes
plt.rcParams['font.family'] = 'DejaVu Sans'
FONT_SIZES = {'title': 16, 'axis_label': 14, 'tick_label': 12, 'legend': 11, 'annotation': 10}

# Output Directory
OUTPUT_DIR = 'results/figures/comprehensive'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Data
def load_data():
    df_summary = pd.read_csv('results/tables/final_summary_tables.csv')
    df_class = pd.read_csv('results/tables/per_class_metrics.csv')
    df_rare = pd.read_csv('results/tables/rare_class_report.csv')
    df_log = pd.read_csv('results/experiment_log.csv')
    return df_summary, df_class, df_rare, df_log

# Helper: Save Figure
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close(fig)

# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

def plot_performance_comparison(df):
    """Grouped Bar Charts for Macro-F1 and G-Mean"""
    metrics = [('Macro_F1', 'Macro-F1 Score'), ('G_Mean', 'G-Mean Score'), ('ROC_AUC', 'ROC-AUC Score')]
    
    for metric, label in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        models = df['Model'].unique()
        strategies = df['Strategy'].unique()
        bar_width = 0.25
        
        # Position of bars on x-axis
        r = np.arange(len(models))
        
        for i, strategy in enumerate(strategies):
            subset = df[df['Strategy'] == strategy]
            # Ensure order matches 'models'
            values = [subset[subset['Model'] == m][metric].values[0] if not subset[subset['Model'] == m].empty else 0 for m in models]
            
            pos = [x + (i * bar_width) for x in r]
            ax.bar(pos, values, color=STRATEGY_COLORS.get(strategy, '#333'), width=bar_width, 
                   edgecolor='white', label=strategy)

        # Formatting
        ax.set_xlabel('Model', fontsize=FONT_SIZES['axis_label'])
        ax.set_ylabel(label, fontsize=FONT_SIZES['axis_label'])
        ax.set_title(f'{label} Comparison Across Strategies', fontsize=FONT_SIZES['title'], fontweight='bold')
        ax.set_xticks([r + bar_width for r in range(len(models))])
        ax.set_xticklabels(models, fontsize=FONT_SIZES['tick_label'])
        ax.legend(title='Strategy', fontsize=FONT_SIZES['legend'])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        save_fig(fig, f"bar_comparison_{metric.lower()}")

def plot_training_efficiency(df):
    """Scatter Plot: Training Time vs Macro-F1"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Map markers to models
    markers = {'lr': 'o', 'rf': 's', 'xgb': '^'}
    
    for idx, row in df.iterrows():
        strategy = row['strategy']
        model = row['model']
        
        ax.scatter(row['training_time'], row['macro_f1'], 
                   color=STRATEGY_COLORS.get(strategy, '#333'),
                   marker=markers.get(model, 'o'),
                   s=150, alpha=0.8, edgecolors='black', linewidth=0.5,
                   label=f"{model.upper()}-{strategy.upper()}")

    ax.set_xlabel('Training Time (seconds)', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel('Macro-F1 Score', fontsize=FONT_SIZES['axis_label'])
    ax.set_title('Efficiency Frontier: Time vs Performance', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Custom Legend
    # We can't use the default legend easily because of the mix, so we'll simplify
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='LR', markersize=10),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', label='RF', markersize=10),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', label='XGB', markersize=10),
        Line2D([0], [0], color=STRATEGY_COLORS['s0'], lw=4, label='S0 (Baseline)'),
        Line2D([0], [0], color=STRATEGY_COLORS['s1'], lw=4, label='S1 (Weight)'),
        Line2D([0], [0], color=STRATEGY_COLORS['s2a'], lw=4, label='S2a (ROS)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    save_fig(fig, "scatter_efficiency")

def plot_class_heatmap(df):
    """Heatmap of F1 Score per Class per Experiment"""
    # Pivot: Index=Class, Columns=Experiment(Model_Strategy), Values=F1
    df['Experiment'] = df['Model'] + '_' + df['Strategy']
    pivot = df.pivot(index='Class', columns='Experiment', values='F1')
    
    # Sort Index by standard class order/frequency if possible
    class_order = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 
                   'Analysis', 'Backdoor', 'Shellcode', 'Worms']
    # Filter to existing classes
    class_order = [c for c in class_order if c in pivot.index]
    pivot = pivot.reindex(class_order)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', 
                linewidths=.5, ax=ax, vmin=0, vmax=1.0)
    
    ax.set_title('Class-Wise F1-Score Heatmap', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.set_xlabel('Experiment Configuration', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel('Class', fontsize=FONT_SIZES['axis_label'])
    plt.xticks(rotation=45, ha='right')
    
    save_fig(fig, "heatmap_class_f1")

def plot_rare_trajectory(df):
    """Line Chart: Recall Trajectory for Rare Classes"""
    # Filter for Rare Classes
    rare_classes = ['Worms', 'Shellcode', 'Backdoor', 'Analysis']
    df_rare = df[df['Class'].isin(rare_classes)].copy()
    
    # We need to map Strategy to a numeric X or ordered categorical
    strategy_order = ['S0', 'S1', 'S2A'] # Uppercase as per CSV likely
    
    # Check if CSV uses S0 or s0
    if 'S0' not in df_rare['Strategy'].unique():
         strategy_order = ['s0', 's1', 's2a']
            
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cls in rare_classes:
        subset = df_rare[df_rare['Class'] == cls]
        # Aggregate if multiple models (take mean) or plot specific model
        # Let's plot mean across models to show general strategy effect
        means = []
        for strat in strategy_order:
            val = subset[subset['Strategy'] == strat]['Recall'].mean()
            means.append(val)
            
        ax.plot(strategy_order, means, marker='o', linewidth=3, 
                label=cls, color=CLASS_COLORS.get(cls, 'black'))

    ax.set_title('Rare Class Recall Trajectory (Avg across Models)', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.set_ylabel('Recall Score', fontsize=FONT_SIZES['axis_label'])
    ax.set_xlabel('Strategy', fontsize=FONT_SIZES['axis_label'])
    ax.legend(title='Rare Class')
    ax.grid(True, alpha=0.3)
    
    save_fig(fig, "line_rare_trajectory")

def plot_radar(df):
    """Radar Chart for S0 vs S2a (Multiclass Task)"""
    # Filter for Multiclass task and one model
    df_multi = df[df['Task'] == 'multi']
    model = 'RF'
    metrics = ['Accuracy', 'Macro_F1', 'Weighted_F1', 'G_Mean', 'ROC_AUC']
    
    strategies = ['S0', 'S2A']
    
    # Setup data
    labels = metrics
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for strat in strategies:
        row = df_multi[(df_multi['Model'] == model) & (df_multi['Strategy'] == strat)]
        if row.empty: 
            print(f"Warning: No data for Radar {model} {strat}")
            continue
        
        # Take the first match if multiple exist (though should be unique per task/model/strategy)
        values = row[metrics].iloc[0].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=strat, color=STRATEGY_COLORS.get(strat, 'black'))
        ax.fill(angles, values, alpha=0.1, color=STRATEGY_COLORS.get(strat, 'black'))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    ax.set_title(f'Radar Chart: {model} S0 vs S2A (Multiclass)', fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    save_fig(fig, "radar_rf_s0_s2a_multi")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Loading data...")
    df_sum, df_cls, df_rare, df_log = load_data()
    
    print("Generating figures...")
    
    # 1. Summary Bars
    plot_performance_comparison(df_sum)
    
    # 2. Efficiency Scatter
    # Normalize log column names if needed
    if 'task' in df_log.columns: # It's lowercase in log csv
        plot_training_efficiency(df_log)
        
    # 3. Class Heatmap
    # Normalize per_class csv
    # Check if 'Model' and 'Strategy' exist or need extraction
    if 'Model' in df_cls.columns:
        plot_class_heatmap(df_cls)
        
    # 4. Rare Trajectory
    # Check if rare_class_report has data
    if not df_rare.empty:
        # Use full per_class for trajectory to be safe if rare report is partial
        # But rare_report is specifically for this.
        # Let's use per_class df but filter for rare
        plot_rare_trajectory(df_cls)
        
    # 5. Radar
    plot_radar(df_sum)
    
    print("Done! All figures saved to results/figures/comprehensive/")
