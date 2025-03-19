# visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Plot pairplot
def plot_pairplot(df):
    plt.figure(figsize=(10, 6))
    sns.pairplot(df[['Age', 'Annual_Income', 'Spending_Score']])
    plt.savefig('pairplot.png')
    plt.close()

# Plot clusters
def plot_clusters(df, kmodel, features, x, y):
    
    # Predict cluster labels
    df['Cluster'] = kmodel.predict(df[features])
    
    # Create scatter plots for each pair of features
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual_Income', y='Spending_Score', data=df, hue='Cluster', palette='colorblind')
    plt.title('Scatter plot')
    plt.savefig('scatter_customer_segmentation.png') 
    plt.close()
    
# Plot the Elbow Plot
def plot_elbow_method(K, WCSS):
    wss = pd.DataFrame({'cluster':K, 'WSS_Score': WCSS})
    plt.figure(figsize=(10, 6))
    wss.plot(x='cluster', y='WSS_Score')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS Score')
    plt.savefig("elbow.png")
    plt.close()

# Plot silhouette plot
def plot_silhouette_clusters(df):
    
    k = range(3,9)
    K = []
    ss = []
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[['Annual_Income','Spending_Score']], )
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Annual_Income','Spending_Score']], ypred)
        K.append(i)
        ss.append(sil_score)
    
    # Store the number of clusters and their respective Silhouette scores in a dataframe
    wss = pd.DataFrame({'Cluster': K, 'Silhouette_Score': ss})

    # Plot Silhouette Plot
    plt.figure(figsize=(10, 6))
    wss.plot(x='Cluster', y='Silhouette_Score')
    plt.title('Silhouette Plot')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig('silhouette.png')
    plt.close()

    return wss

    

      
    
    








