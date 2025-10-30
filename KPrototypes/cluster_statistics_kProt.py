# %%
import numpy as np
import matplotlib.pyplot as plt
import polars as pl


def plot_cluster_percentiles(train_data, clusters_train, n_clusters):
    lower_percentile = 10
    median_percentile = 50
    upper_percentile = 90
    lower_shade_percentile = 30
    mid_shade_percentile = 40
    upper_shade_percentile = 70
    top_shade_percentile = 60

    lower_percentile_columns = []
    upper_percentile_columns = []

    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten() 

    for cluster_idx in range(n_clusters):
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

        ax = axes[cluster_idx]  # select subplot

        ax.grid(color='lightgray')
        ax.plot(range(1, 97), mean_time_series, label='Mean', color='crimson', linewidth=2)
        ax.plot(range(1, 97), median_time_series, label='Median', color='black', linewidth=1.5)
        ax.fill_between(range(1, 97), lower_percentile_series, upper_percentile_series, color='lightgray', alpha=0.3,
                        label=f'{lower_percentile}th - {upper_percentile}th Percentile')
        ax.fill_between(range(1, 97), lower_shade_series, upper_shade_series, color='darkgray', alpha=0.2,
                        label=f'{lower_shade_percentile}th - {upper_shade_percentile}th Percentile')
        ax.fill_between(range(1, 97), mid_shade_series, top_shade_series, color='gray', alpha=0.2,
                        label=f'{mid_shade_percentile}th - {top_shade_percentile}th Percentile')

        ax.set_title(f"Cluster {cluster_idx_real} of {n_clusters} - Mean and Percentiles")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Normalized Energy Consumption", fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticks(range(1, 120, 24))
        ax.set_xticklabels(['0:00', '6:00', '12:00', '18:00', '0:00'], fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
        #ax.legend(fontsize=8)
    fig.legend(handles, labels,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.00),  # centered below plot with some vertical offset
           ncol=5,  # number of columns in legend
           fontsize='large')

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at the bottom for the legend
    plt.savefig(
    "/Users/jansigrist/Documents/SP/Customer_Segmentation_Lantern/Results/Plots/Cluster/HourlyAveraged_percentiles_KProt.png",
    bbox_inches='tight'
    )
    plt.show()

    cluster_percentiles_df = pl.DataFrame(lower_percentile_columns + upper_percentile_columns)
    #print(cluster_percentiles_df)