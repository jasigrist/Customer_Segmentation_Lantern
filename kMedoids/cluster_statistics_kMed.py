# =============================================================================
# CLUSTERED LOAD PROFILE STATISTICAL SUMMARY VISUALIZATION - K-MEDOIDS
# =============================================================================
# Purpose: Generate percentile plots for k-medoids clusters
# Input: Normalized daily profiles (n_households × 96), cluster labels (0-indexed)
# Output: Multi-panel plot + Polars DataFrame of 10th/90th percentiles per cluster
# Percentiles: 10/90 (main), 30/70 (dark shade), 40/60 (mid shade) + mean/median lines
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import polars as pl

def plot_cluster_percentiles(train_data, clusters_train, n_clusters, building_type, dataset, categorical_features):
    """
    Visualize statistical summaries of k-medoids clusters using multiple percentiles.
    
    Parameters
    ----------
    train_data : np.ndarray
        Shape (n_households, 96); normalized daily load profiles
    clusters_train : np.ndarray  
        Cluster labels (0-indexed: 0,1,2,...); length n_households
    n_clusters : int
        Number of clusters (determines subplot rows)
    building_type : str
        'house', 'flat', etc. (for filename and y-label)
    dataset : str
        'GroupE', 'SWW' (for filename)
    categorical_features : str
        'technical', 'sociodemographic' (for filename)
        
    Returns
    -------
    pl.DataFrame
        2×n_clusters table: lower_percentile_cluster_X, upper_percentile_cluster_X
    """

    # =============================================================================
    # PERCENTILE DEFINITIONS: Multi-level uncertainty visualization
    # =============================================================================
    # Main bounds: 10th-90th (80% coverage, robust to outliers)
    # Shade bands: 30-70% (dark), 40-60% (mid) for nested uncertainty
    
    lower_percentile = 10
    median_percentile = 50
    upper_percentile = 90
    lower_shade_percentile = 30
    mid_shade_percentile = 40
    upper_shade_percentile = 70
    top_shade_percentile = 60

    lower_percentile_columns = []
    upper_percentile_columns = []

    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 3.5 * n_clusters), sharex=True)

    if n_clusters == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    # =============================================================================
    # PER-CLUSTER PROCESSING: Compute statistics + plot
    # =============================================================================

    for subplot_idx, cluster_idx in enumerate((range(n_clusters))):
        ax = axes[subplot_idx]
        cluster_idx_real = cluster_idx + 1
        cluster_data = train_data[clusters_train == cluster_idx_real]

        if cluster_data.shape[0] == 0:
            print(f"Warning: Cluster {cluster_idx_real} is empty, skipping.")
            continue

        mean_time_series = np.mean(cluster_data, axis=0)
        median_time_series = np.median(cluster_data, axis=0)
        lower_percentile_series = np.percentile(cluster_data, lower_percentile, axis=0)
        median_percentile_series = np.percentile(cluster_data, median_percentile, axis=0)
        lower_shade_series = np.percentile(cluster_data, lower_shade_percentile, axis=0)
        mid_shade_series = np.percentile(cluster_data, mid_shade_percentile, axis=0)
        top_shade_series = np.percentile(cluster_data, top_shade_percentile, axis=0)
        upper_shade_series = np.percentile(cluster_data, upper_shade_percentile, axis=0)
        upper_percentile_series = np.percentile(cluster_data, upper_percentile, axis=0)

        lower_percentile_series = np.ravel(lower_percentile_series)
        upper_percentile_series = np.ravel(upper_percentile_series)
        lower_shade_series = np.ravel(lower_shade_series)
        mid_shade_series = np.ravel(mid_shade_series)
        top_shade_series = np.ravel(top_shade_series)
        upper_shade_series = np.ravel(upper_shade_series)

        lower_percentile_columns.append(pl.Series(f'lower_percentile_cluster_{cluster_idx_real}', lower_percentile_series))
        upper_percentile_columns.append(pl.Series(f'upper_percentile_cluster_{cluster_idx_real}', upper_percentile_series))

        x = range(1, 97)

        line_mean, = ax.plot(x, mean_time_series, label='Mean', color='crimson', linewidth=2)
        line_median, = ax.plot(x, median_percentile_series, label='Median', color='black', linewidth=1.5)

        fill_main = ax.fill_between(x, lower_percentile_series, upper_percentile_series, color='lightgray', alpha=0.3,
                                   label=f'{lower_percentile}th - {upper_percentile}th Percentile')
        fill_dark = ax.fill_between(x, lower_shade_series, upper_shade_series, color='darkgray', alpha=0.2,
                                   label=f'{lower_shade_percentile}th - {upper_shade_percentile}th Percentile')
        fill_mid = ax.fill_between(x, mid_shade_series, top_shade_series, color='gray', alpha=0.2,
                                   label=f'{mid_shade_percentile}th - {top_shade_percentile}th Percentile')

        if subplot_idx == 0:
            legend_handles = [line_mean, line_median, fill_main, fill_dark, fill_mid]
            legend_labels = ['Mean', 'Median', f'{lower_percentile}th - {upper_percentile}th Percentile',
                             f'{lower_shade_percentile}th - {upper_shade_percentile}th Percentile',
                             f'{mid_shade_percentile}th - {top_shade_percentile}th Percentile']

        ax.set_title(f"Cluster {cluster_idx_real}", loc='center', fontsize=14, fontweight='bold', color='black')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(color='lightgray', zorder=0)
        ax.legend([],[], frameon=False)

        if subplot_idx == n_clusters - 1:
            ax.set_xlabel("Time", fontsize=12)
            ax.set_xticks(range(1, 120, 24))
            ax.set_xticklabels(['0:00', '6:00', '12:00', '18:00', '0:00'], fontsize=10)
            ax.set_xlim(1, 96)
    
    fig.suptitle("Statistical Description of Clustered Consumption Patterns", fontsize=20, color='black', y=0.96)

    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.915),
               ncol=5, fontsize='large', frameon=False)
    
    fig.text(0.04, 0.5, f"Normalized Energy Consumption - {building_type.capitalize()}s", va='center', rotation='vertical', fontsize=16)

    plt.tight_layout(rect=[0.06, 0, 0.99, 0.90])

    plt.savefig(f'/Users/jansigrist/Documents/SP/Customer_Segmentation_Lantern/Results/{dataset}/Plots/Cluster/HourlyAveraged_percentiles_KMed_{building_type}_{categorical_features}.png',
                bbox_inches='tight')
    plt.show()

    #cluster_percentiles_df = pl.DataFrame(lower_percentile_columns + upper_percentile_columns)
