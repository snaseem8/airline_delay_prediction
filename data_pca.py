from pca import PCA
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

class DataPCA(PCA):
    
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
        
        # Replace "N" with 0 and "B" with 1 in the cancelled_code column
        df['cancelled_code'] = np.where(df['cancelled_code'] == 'N', 0, 1)
        
        ###### FLIGHT Columns
        flight_columns = ["flight_number", "scheduled_elapsed_time", "cancelled_code"]
        flight_data = df[flight_columns]
        flight_data_cleaned = flight_data.dropna()

        ###### WEATHER Columns
        weather_columns = [
            "HourlyDryBulbTemperature_x", 
            "HourlyPrecipitation_x", "HourlyStationPressure_x", "HourlyVisibility_x", 
            "HourlyWindSpeed_x", "HourlyDryBulbTemperature_y", 
            "HourlyPrecipitation_y", "HourlyStationPressure_y", 
            "HourlyVisibility_y", "HourlyWindSpeed_y"
        ]
        weather_data = df[weather_columns]
        weather_data_cleaned = weather_data.dropna()
        
        # For the arrival_delay, convert to numeric and handle non-numeric values
        arrival_delay = pd.to_numeric(df['arrival_delay'], errors='coerce')
        # Drop NaN values
        arrival_delay = arrival_delay.dropna()
        # Convert to numpy array for scaling
        y = arrival_delay.values
        
        #* Split into train and test sets
        ###### FLIGHT Columns 
        X_flight_train, X_flight_test, y_train, y_test = train_test_split(
            flight_data_cleaned, y, test_size=0.2, random_state=42
        )
        
        ###### WEATHER Columns weather_
        X_weather_train, X_weather_test, y_train, y_test = train_test_split(
            weather_data_cleaned, y, test_size=0.2, random_state=42
        )
        
        return X_weather_train, X_weather_test, X_flight_train, X_flight_test, y_train, y_test
    
    def inspect_weights(self, pca, X_train, pc_count, feat_count=5):
        for i in range(pc_count):
            # Turn PC1 into a pandas Series for easier inspection
            pc_weights = pd.Series(pca.V[i], index=X_train.columns)

            # Sort by absolute contribution
            pc_top_features = pc_weights.abs().sort_values(ascending=False)
            print(f"Top features contributing to PC{i}:")
            print(pc_top_features.head(feat_count))  # top 10
            
    def delay_viz(self, X_sample, y_sample):
        vmin = np.percentile(y_sample, 5)
        vmax = np.percentile(y_sample, 95)
        # Plot with color representing delay
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            X_sample[:, 1], X_sample[:, 2],     #! These features being visualized can be changed!
            c=y_sample,
            cmap='coolwarm',  # blue = early, red = late
            vmin=vmin, vmax=vmax,  # ← clip color range manually
            s=15,
            alpha=0.8
        )

        plt.colorbar(scatter, label='Arrival Delay (minutes)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Projection Colored by Arrival Delay')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # plt.scatter(X[:, 0], X[:, 1])
        # plt.title('PCA Projection')
        # plt.show()
        
    def airport_viz(self, X_sample, X_train, sample_indices):
        origin_iah_values = X_train.iloc[sample_indices]['Origin_IAH'].values
        # Convert Origin one-hot column to label
        # X_train_sample = X_train.loc[sample_indices]
        airport_labels = np.where(origin_iah_values == 1, 'IAH', 'HOU')

        # Create a color map
        color_map = {'IAH': 'blue', 'HOU': 'orange'}
        colors = [color_map[label] for label in airport_labels]

        plt.figure(figsize=(8,6))
        plt.scatter(X_sample[:, 0], X_sample[:, 1], c=colors, alpha=0.5)

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Projection Colored by Origin Airport")
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='IAH', markerfacecolor='blue', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='HOU', markerfacecolor='orange', markersize=8)
        ])
        plt.grid(True)
        plt.show()
    
    def inspect_weights(self, pca, X_train, pc_count):
        for i in range(pc_count):
            # Turn PC1 into a pandas Series for easier inspection
            pc_weights = pd.Series(pca.V[i], index=X_train.columns)

            # Sort by absolute contribution
            pc_top_features = pc_weights.abs().sort_values(ascending=False)
            print(f"Top features contributing to PC{i}:")
            print(pc_top_features.head(10))  # top 10
            
            
if "__main__" == __name__:
    pca = DataPCA()
    
    current_directory = os.getcwd()
    data_file_name = os.path.join(current_directory, "archive", "05-2019.csv")
    
    df = pca.read_data(csv_file_name=data_file_name)
    X_weather_train, X_weather_test, X_flight_train, X_flight_test, y_train, y_test = pca.clean_data(df)
    
    X_weather_train_np = X_weather_train.astype(float).to_numpy()
    X_weather_test_np = X_weather_test.astype(float).to_numpy()
    X_flight_train_np = X_flight_train.astype(float).to_numpy()
    X_flight_test_np = X_flight_test.astype(float).to_numpy()

    scaler = StandardScaler()
    X_weather_train_scaled = scaler.fit_transform(X_weather_train)
    X_weather_test_scaled = scaler.transform(X_weather_test)
    X_flight_train_scaled = scaler.fit_transform(X_flight_train)
    X_flight_test_scaled = scaler.transform(X_flight_test)
    
    # Fit training data to a model
    pca.fit(X_weather_train_scaled)
    pca.fit(X_flight_train_scaled)
    
    #! This can be flipped to grabbing specifc number of principal components (K) if needed
    # X_train_pca = pca.transform(X_train_scaled, K=3)
    # X_test_pca = pca.transform(X_test_scaled, K=3)
    X_weather_train_pca = pca.transform_rv(X_weather_train_scaled, retained_variance=0.90)
    X_weather_test_pca  = pca.transform_rv(X_weather_test_scaled, retained_variance=0.90)
    X_flight_train_pca = pca.transform_rv(X_flight_train_scaled, retained_variance=0.90)
    X_flight_test_pca  = pca.transform_rv(X_flight_test_scaled, retained_variance=0.90)
    
    print(f"X train weather pca shape: {X_weather_train_pca.shape}")
    print(f"X test weather pca shape: {X_weather_test_pca.shape}")
    print(f"X train flight pca shape: {X_flight_train_pca.shape}")
    print(f"X test flight pca shape: {X_flight_test_pca.shape}")
    
    sample_size = 5000
    # Randomly choose indices
    weather_indices = np.random.choice(X_weather_train_pca.shape[0], size=sample_size, replace=False)
    flight_indices  = np.random.choice(X_flight_train_pca.shape[0], size=sample_size, replace=False)
    
    
    # Subset the data
    X_weather_sample = X_weather_train_pca[weather_indices]
    X_flight_sample  = X_flight_train_pca[flight_indices]
    y_sample = y_train.astype(float).to_numpy()[flight_indices]
    
    pca.delay_viz(X_weather_sample, y_sample)
    pca.delay_viz(X_flight_sample, y_sample)
    
    # pca.airport_viz(X_sample, X_train, indices)
    # pca.airport_viz(X_sample, X_train, indices)
    
    pca.visualize(X=X_weather_sample, y=y_sample, fig_title="Weather PCA Projection")
    pca.visualize(X=X_flight_sample, y=y_sample, fig_title="Airline PCA Projection")
    
    
    # See which original features are weighted the most in the principal components
    pca.inspect_weights(pca, X_weather_train, pc_count=X_weather_train_pca.shape[1])
    pca.inspect_weights(pca, X_flight_train, pc_count=X_flight_train_pca.shape[1])
    
    # Convert PCA results back to DataFrames for saving
    X_weather_train_pca_df = pd.DataFrame(X_weather_train_pca)
    X_weather_test_pca_df = pd.DataFrame(X_weather_test_pca)
    X_flight_train_pca_df = pd.DataFrame(X_flight_train_pca)
    X_flight_test_pca_df = pd.DataFrame(X_flight_test_pca)

    # Match shapes to avoid index issues when reloading
    # X_weather_train.to_csv("X_weather_train_cleaned.csv", index=False)    #! Uncomment these for just cleaned data w/ no PCA to save to csv
    # X_weather_test.to_csv("X_weather_test_cleaned.csv", index=False)
    # X_weather_train_pca_df.to_csv("X_weather_train_pca.csv", index=False)
    # X_weather_test_pca_df.to_csv("X_weather_test_pca.csv", index=False)
    
    
    # X_flight_train.to_csv("X_flight_train_cleaned.csv", index=False)    #! Uncomment these for just cleaned data w/ no PCA to save to csv
    # X_flight_test.to_csv("X_flight_test_cleaned.csv", index=False)
    # X_flight_train_pca_df.to_csv("X_flight_train_pca.csv", index=False)
    # X_flight_test_pca_df.to_csv("X_flight_test_pca.csv", index=False)
    
    # y_train.to_csv("y_train.csv", index=False)
    # y_test.to_csv("y_test.csv", index=False)
    
    print(f"Data has been cleaned, transformed, and saved!")