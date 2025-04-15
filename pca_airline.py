
from pca import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

class AirlinePCA(PCA):
    
    def __init__(self):
        super().__init__()
        self.sample_size = 120_000
        self.data_to_use = "Both"
        
        
    def read_data(self, csv_file_name):
        df = pd.read_csv(csv_file_name)
        print(df.info())
        print(df.describe())
        print(df.isnull().sum())
        
        # print(df[:5])
        return df
    
    
    def remove_rare_categories(self, df, column_name, min_count=2):
        print(f"\nChecking column: {column_name}")
        value_counts = df[column_name].value_counts()
        print(f"Value counts:\n{value_counts}")

        valid_values = value_counts[value_counts >= min_count].index
        print(f"Categories kept (appeared at least {min_count} times): {list(valid_values)}")

        original_rows = len(df)
        df_filtered = df[df[column_name].isin(valid_values)]
        removed_rows = original_rows - len(df_filtered)

        print(f"Removed {removed_rows} rows (from {original_rows} to {len(df_filtered)})")

        return df_filtered  
    
    
    def clean_data(self, df, data_to_use="Airline"):
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
        # Convert DepTime and ArrTime from datetime to minutes since midnight
        # Make sure DepTime and ArrTime are safe integers first
        df['scheduled_departure_dt'] = pd.to_datetime(df['scheduled_departure_dt'])
        df['scheduled_arrival_dt'] = pd.to_datetime(df['scheduled_arrival_dt'])

        # Minutes since midnight (use hour and minute)
        df['dep_minutes'] = df['scheduled_departure_dt'].dt.hour * 60 + df['scheduled_departure_dt'].dt.minute
        df['arr_minutes'] = df['scheduled_arrival_dt'].dt.hour * 60 + df['scheduled_arrival_dt'].dt.minute

        # Drop the original columns
        df = df.drop(columns=['scheduled_departure_dt', 'scheduled_arrival_dt'])
        
        # df = self.remove_rare_categories(df, 'carrier_code')
        # df = self.remove_rare_categories(df, 'origin_airport')
        # df = self.remove_rare_categories(df, 'destination_airport')

        #* Remove cancelled flights
        if 'cancelled_code' in df.columns:
            df = df[df['cancelled_code'] != 1]

        #* Drop irrelevant or unhelpful columns
        # Note that delay_late_aircraft_arrival has aircraft mispelled as "aircarft" in dataset columns
        all_cols = ["carrier_code", "flight_number", "origin_airport", "destination_airport", "date", 
                    "scheduled_elapsed_time", "tail_number", "departure_delay", "arrival_delay", 
                    "delay_carrier", "delay_weather", "delay_national_aviation_system", "delay_security", 
                    "delay_late_aircarft_arrival", "cancelled_code", "year", "month", "day", "weekday", 
                    "scheduled_departure_dt", "scheduled_arrival_dt", "actual_departure_dt", 
                    "actual_arrival_dt", "STATION_x", "HourlyDryBulbTemperature_x", "HourlyPrecipitation_x", 
                    "HourlyStationPressure_x", "HourlyVisibility_x", "HourlyWindSpeed_x", "STATION_y", 
                    "HourlyDryBulbTemperature_y", "HourlyPrecipitation_y", "HourlyStationPressure_y", 
                    "HourlyVisibility_y", "HourlyWindSpeed_y"]
        
        weather_drop_cols = ["STATION_x", "STATION_y"]
        
        weather_keep_cols = ["HourlyDryBulbTemperature_x", "HourlyPrecipitation_x", 
                    "HourlyStationPressure_x", "HourlyVisibility_x", "HourlyWindSpeed_x", 
                    "HourlyDryBulbTemperature_y", "HourlyPrecipitation_y", "HourlyStationPressure_y", 
                    "HourlyVisibility_y", "HourlyWindSpeed_y"]
        
        airport_drop_cols = ["flight_number", "date", "tail_number", "arrival_delay", "delay_carrier", 
                    "delay_weather", "delay_national_aviation_system", "delay_security", 
                    "delay_late_aircarft_arrival", "cancelled_code", "year", "actual_departure_dt", 
                    "actual_arrival_dt"]
        
        airport_keep_cols = ["carrier_code", "origin_airport", "destination_airport",
                    "scheduled_elapsed_time", "departure_delay", "month", "day", "weekday", 
                    "dep_minutes", "arr_minutes"]
        
        # df = df[airport_keep_cols]
        
        if data_to_use == "Airline":
            drop_cols = airport_drop_cols + weather_drop_cols + weather_keep_cols
        elif data_to_use == "Weather":
            drop_cols = airport_drop_cols + [col for col in airport_keep_cols if col != "departure_delay"] + weather_drop_cols
        elif data_to_use == "Both":
            drop_cols = airport_drop_cols + weather_drop_cols
        else:
            raise ValueError(f"Invalid value for data_to_use: '{data_to_use}'. Must be 'Airline', 'Weather', or 'Both'.")
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        #* Remove rows with missing target
        df = df.dropna(subset=['departure_delay'])
        
        #* Encode categorical variables using one-hot encoding
        categorical_features_to_encode = ['carrier_code', 'origin_airport', 'destination_airport']
        for feature in categorical_features_to_encode:
            if feature in df.columns:
                df = pd.get_dummies(df, columns=[feature], drop_first=True)
        
        #* Sample subset for memory/performance
        if self.sample_size is not None:
            df = df.sample(n=self.sample_size, random_state=42)

        #* Define features and target
        y = df['departure_delay']
        X = df.drop('departure_delay', axis=1)

        #* Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Data cleaning and splitting done!")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    
    def delay_viz(self, X_sample, y_sample):
        vmin = np.percentile(y_sample, 5)
        vmax = np.percentile(y_sample, 95)
        x_axis_pc = 6
        y_axis_pc = 7
        # Plot with color representing delay
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            X_sample[:, x_axis_pc], X_sample[:, y_axis_pc],     #! These features being visualized can be changed!
            c=y_sample,
            cmap='coolwarm',  # blue = early, red = late
            vmin=vmin, vmax=vmax,  # ← clip color range manually
            s=15,
            alpha=0.8
        )

        plt.colorbar(scatter, label='Arrival Delay (minutes)')
        plt.xlabel(f'Principal Component {x_axis_pc+1}')
        plt.ylabel(f'Principal Component {y_axis_pc+1}')
        plt.title(f'PCA Projection Colored by Arrival Delay ({self.data_to_use} Data)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    
    def inspect_weights(self, pca, X_train, pc_count):
        pc_count = min(pc_count, 10)
        for i in range(pc_count):
            # Turn PC1 into a pandas Series for easier inspection
            pc_weights = pd.Series(pca.V[i], index=X_train.columns)

            # Sort by absolute contribution
            pc_top_features = pc_weights.abs().sort_values(ascending=False)
            print(f"Top features contributing to PC{i}:")
            print(pc_top_features.head(10))  # top 10


if "__main__" == __name__:
    pca = AirlinePCA()
    
    # data_file_name = "hflights.csv"
    data_file_name = "mixed_data/05-2019.csv"   #! Have to unzip the .zip file of the csv in the mixed_data folder
    
    df = pca.read_data(csv_file_name=data_file_name)
    X_train, X_test, y_train_df, y_test_df = pca.clean_data(df, data_to_use=pca.data_to_use)  #! change this to use different portions of dataset in init()
        
    X_train_np = X_train.astype(float).to_numpy()
    X_test_np = X_test.astype(float).to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit training data to a model
    pca.fit(X_train_scaled)
    
    # This can be flipped to grabbing specifc number of principal components (K) if needed
    # X_train_pca = pca.transform(X_train_scaled, K=3)
    # X_test_pca = pca.transform(X_test_scaled, K=3)
    X_train_pca = pca.transform_rv(X_train_scaled, retained_variance=0.90)    #! adjust retained_variance to see what level does better with regression models
    X_test_pca = pca.transform_rv(X_test_scaled, retained_variance=0.90)
    
    print(f"X train pca shape: {X_train_pca.shape}")
    print(f"X test pca shape: {X_test_pca.shape}")
    
    # Pick a subsample for visualization
    sample_size = 5000
    # Randomly choose indices
    indices = np.random.choice(X_train_pca.shape[0], size=sample_size, replace=False)
    # Subset the data
    X_sample = X_train_pca[indices]
    y_sample = y_train_df.astype(float).to_numpy()[indices]
    pca.delay_viz(X_sample, y_sample)
    # pca.visualize(X=X_sample, y=y_sample, fig_title="Airline PCA Projection")
    
    # See which original features are weighted the most in the principal components
    pca.inspect_weights(pca, X_train, pc_count=X_train_pca.shape[1])
    
    # See what the correlation is of different principal components with departure_delay
    # Track the most correlated component
    max_corr = 0
    best_pc_index = -1

    pc_count = min(X_train_pca.shape[0], 10)
    for i in range(pc_count):
        pc = X_train_pca[:, i]
        corr = np.corrcoef(pc, y_train_df.astype(float).to_numpy())[0, 1]
        
        if abs(corr) > abs(max_corr):
            max_corr = corr
            best_pc_index = i
        print(f"PC correlation: PC{i+1} = {corr:.4f}")
        

    print(f"Most correlated PC: PC{best_pc_index+1}")
    print(f"Correlation with departure delay: {max_corr:.4f}")
    
    # Convert PCA results back to DataFrames for saving
    X_train_pca_df = pd.DataFrame(X_train_pca)
    X_test_pca_df = pd.DataFrame(X_test_pca)

    # Match shapes to avoid index issues when reloading
    # X_train.to_csv("X_train_cleaned.csv", index=False)    #! Uncomment these to save to csv
    # X_test.to_csv("X_test_cleaned.csv", index=False)
    # X_train_pca_df.to_csv("X_train_pca.csv", index=False)
    # X_test_pca_df.to_csv("X_test_pca.csv", index=False)
    # y_train_df.to_csv("y_train.csv", index=False)
    # y_test_df.to_csv("y_test.csv", index=False)
    
    print(f"Data has been cleaned, transformed, and saved!")
