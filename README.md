# customer_segemation_application

This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://w12customersegemation.streamlit.app/)

password - streamlit

This application performs customer classification for a mall using the KMeans algorithm. The model helps businesses understand different customer segments based on features like age, annual income, and spending score.

## Overview

The application uses the `mall_customers.csv` dataset to train a KMeans model and segment customers into distinct groups. It leverages machine learning techniques to identify patterns and provide insights for targeted marketing strategies.

## Features

- User-friendly interface powered by Streamlit.
- Input form to explore different customer segments.
- Real-time visualization of customer clusters.
- Accessible via Streamlit Community Cloud.

## Dataset

The application is trained on the **mall_customers.csv** dataset, which includes features like:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Technologies Used

- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Seaborn** and **Matplotlib**: For exploratory data analysis and visualization.
- **pickle**: For saving and loading machine learning models.

## Model

The predictive model is trained using the KMeans algorithm. It applies preprocessing steps like creating dummy variables, using elbow method to find the best k value. The model aims to segment customers into clusters based on their annual income and spending score.

## Future Enhancements

- Adding support for additional datasets.
- Incorporating more advanced clustering algorithms.
- Adding visualizations to better represent customer segments.

## Installation (for local deployment)

If you want to run the application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/credit_eligibility_application.git
   cd credit_eligibility_application

   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt

   ```

4. Run the Streamlit application:
   ```bash
   streamlit run streamlit.py
   ```

#### Thank you for using the Credit Eligibility Application! Feel free to share your feedback.
