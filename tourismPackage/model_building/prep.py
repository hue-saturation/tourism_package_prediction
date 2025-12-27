import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# HF API setup
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/kesavak/tourism-package-prediction/Tourism.csv"  # Keep your dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully. Shape:", df.shape)

# Drop ID column
df.drop(columns=['CustomerID'], inplace=True, errors='ignore')

# Define categorical columns for YOUR tourism dataset
categorical_cols = [
    'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 
    'Passport', 'CityTier', 'ProductPitched', 'PreferredPropertyStar', 
    'OwnCar', 'Designation'
]

# Encode ALL categorical columns
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le  # Save for later inference
    else:
        print(f"Warning: Column {col} not found")

# Target for tourism: ProdTaken (binary: package purchased or not)
target_col = 'ProdTaken'

# Features and target
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Features shape: {X.shape}, Target distribution:\n{y.value_counts()}")

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for imbalanced classes
)

# Save splits
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload to YOUR HF dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
repo_id = "kesavak/tourism-package-prediction"  # Your repo

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_id,
        repo_type="dataset",
    )
print("All files uploaded to HF dataset repo!")
