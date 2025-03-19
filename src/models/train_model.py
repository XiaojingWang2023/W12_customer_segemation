# train_model.py
from sklearn.cluster import KMeans
import pickle
import pandas as pd

# Using elbow method find the best k
def elbow_method(df):
    k = range(3,9)
    K = []
    WCSS = []
    for i in k:
        kmodel = KMeans(n_clusters=i).fit(df[['Annual_Income','Spending_Score']])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)

    # Store the number of clusters and their respective WSS scores in a dataframe
    wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})
    
    return K, WCSS, wss


# Train the kmeans model by using the best k
def train_kmodel(df, K, features):
    
    # train_kmodel model 
    kmodel = KMeans(n_clusters=K).fit(df[features])
    
    # Get the centroids of the clusters
    centroids = kmodel.cluster_centers_

    # Get cluster labels  
    ypred = kmodel.labels_
       
    # Save the trained model
    with open('models/Kmodel.pkl', 'wb') as f:
        pickle.dump(kmodel, f)

    return kmodel, ypred


    
    
    
    





    

