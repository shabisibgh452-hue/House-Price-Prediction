# Step 1: Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load Dataset
df = pd.read_csv("House Price Prediction Dataset (1).csv")

# Step 3: Display Basic Information
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Check Missing Values
print(df.isnull().sum())

# Step 5: Remove Duplicate Rows
df = df.drop_duplicates()

# Step 6: Convert Categorical Columns into Numbers
le = LabelEncoder()

categorical_columns = ['Location', 'Condition', 'Garage']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Step 7: Separate Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Step 8: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 9: Convert Scaled Data into DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Step 10: Display Preprocessed Data
print("\nPreprocessed Features:")
print(X_scaled.head())

print("\nTarget Variable:")
print(y.head())