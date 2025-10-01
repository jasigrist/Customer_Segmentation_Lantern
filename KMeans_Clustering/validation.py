# %% [markdown]
# # Generating Weekday and Weekend profiles

# %%
import polars as pl
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans


# %%
def validation(train_data):


    seed_random = 42

    fitted_kmeans = {}
    labels_kmeans = {}
    df_scores = []
    k_values_to_try = np.arange(2,26)
    train_data = train_data.reshape((train_data.shape[0], train_data.shape[1]))

    #fig, ax = plt.subplots(12, 2, figsize=(15,8))
    for n_clusters in k_values_to_try:
        #Perform clustering.
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=seed_random,
                        )
        labels_clusters = kmeans.fit_predict(train_data)
        q, mod = divmod(n_clusters, 2)
    
        #Insert fitted model and calculated cluster labels in dictionaries,
        #for further reference.
        fitted_kmeans[n_clusters] = kmeans
        labels_kmeans[n_clusters] = labels_clusters
    
        #Calculate various scores, and save them for further reference.
        silhouette = silhouette_score(train_data, labels_clusters)
        ch = calinski_harabasz_score(train_data, labels_clusters)
        db = davies_bouldin_score(train_data, labels_clusters)
        tmp_scores = {"no_of_clusters": n_clusters,
                    "silhouette_score": silhouette,
                    "calinski_harabasz_score": ch,
                    "davies_bouldin_score": db
                    }
        df_scores.append(tmp_scores)
    
    #Create a DataFrame of clustering scores, using `n_clusters` as index, for easier plotting.
    df_scores = pl.DataFrame(df_scores)
    #df_scores.set_index("no_of_clusters", inplace=True)

    plt.figure(figsize=(6, 4))
    #plt.plot(df_scores['no_of_clusters'][:25],df_scores['silhouette_score'][:25],label="silhouette_score", marker ='o', color='orange')
    plt.plot(df_scores['no_of_clusters'],df_scores['davies_bouldin_score'],label= "Davies_Bouldin_score", marker = 'o', color='black')
    #plt.plot(df_scores['no_of_clusters'],df_scores['calinski_harabasz_score'],label= "calinski_harabasz_score", marker = 'o')
    plt.title(f"Scores vs Number of clusters")
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Scores')
    #plt.xticks(k_values_to_try)
    plt.ylim(0.0,3)
    plt.legend(loc='upper right') 
    plt.grid(False)
    #plt.savefig(rf'C:\Desktop\plots\\normalized\Davies_Score_{season}_{days}.png')
    plt.show()

    print(min(df_scores['davies_bouldin_score']))

# %%



