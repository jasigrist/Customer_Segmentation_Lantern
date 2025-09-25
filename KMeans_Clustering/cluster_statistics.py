# %%
import numpy as np
import matplotlib.pyplot as plt
import polars as pl


def plot_cluster_percentiles(train_data, clusters_train, n_clusters):

    lower_percentile = 10
    median_percentile = 50
    upper_percentile = 90

    # Percentiles to be used for shading
    lower_shade_percentile = 30
    mid_shade_percentile = 40
    upper_shade_percentile = 70
    top_shade_percentile = 60

    # Initialize lists to store percentile data for each cluster
    lower_percentile_columns = []
    upper_percentile_columns = []

    # Assuming train_weekdays and clusters_train are defined and n_clusters is set
    for cluster_idx in range(n_clusters):
        # Get all time series in this cluster
        cluster_data = train_data[clusters_train == cluster_idx]
    
        # Calculate the mean and percentiles for the cluster
        mean_time_series = np.mean(cluster_data, axis=0)
        median_time_series = np.median(cluster_data, axis=0)
        lower_percentile_series = np.percentile(cluster_data, lower_percentile, axis=0)
        median_percentile_series = np.percentile(cluster_data, median_percentile, axis=0)
    
        lower_shade_series = np.percentile(cluster_data, lower_shade_percentile, axis=0)
        mid_shade_series = np.percentile(cluster_data, mid_shade_percentile, axis=0)
        top_shade_series = np.percentile(cluster_data, top_shade_percentile, axis=0)
        upper_shade_series = np.percentile(cluster_data, upper_shade_percentile, axis=0)
    
        upper_percentile_series = np.percentile(cluster_data, upper_percentile, axis=0)

        # Convert to flat arrays if needed
        lower_percentile_series = np.ravel(lower_percentile_series)
        upper_percentile_series = np.ravel(upper_percentile_series)
        lower_shade_series = np.ravel(lower_shade_series)
        mid_shade_series = np.ravel(mid_shade_series)
        top_shade_series = np.ravel(top_shade_series)
        upper_shade_series = np.ravel(upper_shade_series)

        # Store results in lists
        lower_percentile_columns.append(pl.Series(f'lower_percentile_cluster_{cluster_idx+1}', lower_percentile_series))
        upper_percentile_columns.append(pl.Series(f'upper_percentile_cluster_{cluster_idx+1}', upper_percentile_series))
    
        # Plot the mean and percentile bands
        plt.figure(figsize=(10, 6))
        plt.grid( color='lightgray')
        plt.plot(range(1, 97), mean_time_series, label='Mean', color='crimson', linewidth=2)
        plt.plot(range(1, 97), median_time_series, label='Median', color='black', linewidth=1.5)
    
        # Fill between percentiles for shading
        plt.fill_between(range(1, 97), lower_percentile_series, upper_percentile_series, color='lightgray', alpha=0.3, label=f'{lower_percentile}th - {upper_percentile}th Percentile')
        plt.fill_between(range(1, 97), lower_shade_series, upper_shade_series, color='darkgray', alpha=0.2, label=f'{lower_shade_percentile}th - {upper_shade_percentile}th Percentile')
        plt.fill_between(range(1, 97), mid_shade_series, top_shade_series, color='gray', alpha=0.2, label=f'{mid_shade_percentile}th - {top_shade_percentile}th Percentile')
    
        plt.title(f"Cluster {cluster_idx + 1} of {n_clusters} - Mean and Percentiles")
        plt.xlabel("Time", fontsize = 14)
        plt.ylabel("Normalized Energy Consumption")
        plt.ylim(0, 1)
        plt.yticks(fontsize = 14)
        plt.xticks(range(1, 120, 24), labels=['0:00', '6:00', '12:00', '18:00', '0:00'], fontsize = 14)
        plt.margins(0)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    # Combine results into a Polars DataFrame
    cluster_percentiles_df = pl.DataFrame(lower_percentile_columns + upper_percentile_columns)
    print(cluster_percentiles_df)
