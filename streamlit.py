import pandas as pd
import pickle
import streamlit as st



# Set the page title and description
st.title("Mall Customer Segmentation")


# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()


# Load the pre-trained model
with open("models/kmodel.pkl", "rb") as k_pickle:
    kmodel = pickle.load(k_pickle)
    k_pickle.close()

# Function to create dummy variables for gender
def create_dummy_vars(df):
    # Convert gender into numerical values (Male: 1, Female: 0)
    df['Gender_Male'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['Gender_Female'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
    # Drop original 'Gender' column
    df = df.drop(columns=['Gender'])
    return df

# Define the cluster centroids (replace with your actual centroids)
centroids = [
    [43.93421053, 55.21052632, 49.44736842],  # Cluster 0 centroid
    [40.32432432, 87.43243243, 18.18918919],  # Cluster 1 centroid
    [32.69230769, 86.53846154, 82.12820513],  # Cluster 2 centroid
    [24.96, 28.04, 77.0],                     # Cluster 3 centroid
    [45.2173913, 26.30434783, 20.91304348],   # Cluster 4 centroid
]

# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Customer Details")
    
    # Gender input
    Gender = st.selectbox("Gender", options=["Male", "Female"])
    
    # Age input
    Age = st.number_input("Age", min_value=0, step=1)
    
    # Annual Income input
    Annual_Income = st.number_input("Annual Income (k$)", min_value=10, max_value=150, step=10)
    
    # Spending Score input
    Spending_Score = st.number_input("Spending Score (1-100)", min_value=10, max_value=100, step=10)
    
    # Submit button
    submitted = st.form_submit_button("Predict Customer Clasification")


# Handle the dummy variables and make prediction
if submitted:
    Gender_Male = 0 if Gender == "Female" else 1
    Gender_Female = 1 if Gender == "Female" else 0

     # Create a DataFrame with the user inputs
    user_data = pd.DataFrame({
        'Gender': [Gender],
        'Age': [Age],
        'Annual_Income': [Annual_Income],
        'Spending_Score': [Spending_Score]
    })

    # Feature engineering
    X_user = user_data[['Age', 'Annual_Income', 'Spending_Score']]

      # Make prediction
    label = kmodel.predict(X_user)[0]

    # Display result
    st.write(f"The customer is predicted to be in Cluster {label}.")

    # Add information based clustering analysis
    st.write("This cluster typically represents customers with the following characteristics:")
    if label == 0:
        st.write("Cluster 0: Individuals with moderate income and spending.")
    elif label == 1:
        st.write("Cluster 1: High-income individuals with low spending.")
    elif label == 2:
        st.write("Cluster 2: High income individuals with high spending.")
    elif label == 3:
        st.write("Cluster 3: Low-income individuals with high spending.")
    elif label == 4:
        st.write("Cluster 4: Low income individuals with low spending.")

st.write(
    """We used a machine learning (KMeans) model to predict customer segementation. The images as following"""
)
st.image("scatter_customer_segmentation.png") 
st.image("silhouette.png")
st.image("elbow.png")
st.image("pairplot.png")
