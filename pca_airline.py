import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from ml_cs7641.HW3.pca import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

class AirlinePCA(PCA):
    
    def __init__(self):
        super().__init__()
        
        
    def read_data(self, csv_file_name):
        df = pd.read_csv(csv_file_name)
        print(df.info())
        print(df.describe())
        print(df.isnull().sum())
        
        # print(df[:5])
        return df
    
    
    def clean_data(self, df):
        #* Drop duplicates
        df = df.drop_duplicates()
        
        #* Handle missing values
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        # Fill categorical columns with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')

        #* Convert time features
        # Convert DepTime and ArrTime to minutes since midnight
        # Make sure DepTime and ArrTime are safe integers first
        df['DepTime'] = df['DepTime'].fillna(0).astype(float).astype(int)
        df['ArrTime'] = df['ArrTime'].fillna(0).astype(float).astype(int)

        # Then safely convert to minutes since midnight
        df['DepMinutes'] = df['DepTime'].astype(str).str.zfill(4).str[:2].astype(int) * 60 + \
                        df['DepTime'].astype(str).str.zfill(4).str[2:].astype(int)

        df['ArrMinutes'] = df['ArrTime'].astype(str).str.zfill(4).str[:2].astype(int) * 60 + \
                        df['ArrTime'].astype(str).str.zfill(4).str[2:].astype(int)


        # Drop the original columns
        df = df.drop(columns=['DepTime','ArrTime'])

        #* Encode categorical variables using one-hot encoding
        categorical_features_to_encode = ['UniqueCarrier', 'Origin', 'Dest']
        for feature in categorical_features_to_encode:
            if feature in df.columns:
                df = pd.get_dummies(df, columns=[feature], drop_first=True)

        #* Drop irrelevant or unhelpful columns
        drop_cols = ['Year', 'FlightNum', 'TailNum', 'Cancelled', 'CancellationCode', 'Diverted']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        #* Remove rows with missing target
        df = df.dropna(subset=['ArrDelay'])

        #* Define features and target
        y = df['ArrDelay']
        X = df.drop('ArrDelay', axis=1)

        #* Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Data cleaning and splitting done!")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    
    def check_plot(self, X):
        # X needs to be 2D
        plt.scatter(X[:, 0], X[:, 1])
        plt.title('PCA Projection')
        plt.show()


if "__main__" == __name__:
    pca = AirlinePCA()
    
    data_file_name = "hflights.csv"
    output_file_name = "transformed_airline_data.csv"
    
    df = pca.read_data(csv_file_name=data_file_name)
    X_train, X_test, y_train_df, y_test_df = pca.clean_data(df)
    
    X_train_np = X_train.astype(float).to_numpy()
    X_test_np = X_test.astype(float).to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit training data to a model
    pca.fit(X_train_scaled)
    
    X_train_pca = pca.transform(X_train_scaled, K=3)
    X_test_pca = pca.transform(X_test_scaled, K=3)
    
    print(f"X train pca shape: {X_train_pca.shape}")
    print(f"X test pca shape: {X_test_pca.shape}")
    
    sample_size = 5000
    # Randomly choose indices
    indices = np.random.choice(X_train_pca.shape[0], size=sample_size, replace=False)
    # Subset your data
    X_sample = X_train_pca[indices]
    y_sample = y_train_df.astype(float).to_numpy()[indices]
    pca.check_plot(X=X_sample)
    pca.visualize(X=X_sample, y=y_sample, fig_title="Airline PCA Projection")
    
    # Convert PCA results back to DataFrames for saving
    X_train_pca_df = pd.DataFrame(X_train_pca)
    X_test_pca_df = pd.DataFrame(X_test_pca)

    # Match shapes to avoid index issues when reloading
    X_train_pca_df.to_csv("X_train_pca.csv", index=False)
    X_test_pca_df.to_csv("X_test_pca.csv", index=False)
    y_train_df.to_csv("y_train.csv", index=False)
    y_test_df.to_csv("y_test.csv", index=False)
    
    print(f"Data has been transformed and saved to a new CSV file named {output_file_name}!")