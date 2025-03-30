import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load the dataset
df = pd.read_csv("your_dataset.csv")

# 2. Quick data check
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 3. Drop duplicates
df = df.drop_duplicates()

# 4. Handle missing values
# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with 'Unknown'
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

# 5. Convert time features
if 'FlightDate' in df.columns:
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])

if 'ScheduledDepTime' in df.columns:
    df['DepHour'] = pd.to_datetime(df['ScheduledDepTime'].astype(str).str.zfill(4), format='%H%M', errors='coerce').dt.hour

# 6. Encode categorical variables using one-hot encoding
categorical_features_to_encode = ['Airline', 'Origin', 'Dest']
for feature in categorical_features_to_encode:
    if feature in df.columns:
        df = pd.get_dummies(df, columns=[feature], drop_first=True)

# 7. Drop irrelevant or unhelpful columns
drop_cols = ['FlightNumber', 'TailNumber', 'ScheduledDepTime', 'ScheduledArrTime']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# 8. Remove rows with missing target
df = df.dropna(subset=['ArrivalDelay'])

# 9. Define features and target
X = df.drop('ArrivalDelay', axis=1)
y = df['ArrivalDelay']

# 10. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data cleaning and splitting done!")
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
