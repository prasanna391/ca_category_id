import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# 1. Tell Python where your data is
data_path = "data/processed/processed_data.csv"
df = pd.read_csv(data_path)

# --- [Your Model Training Code] ---
# Separating features (X) and target (y)
# WARNING: Replace 'category' with the actual column name you want to predict
X = df.drop(columns=['category']) 
y = df['category']

# Creating and training the model
trained_model = RandomForestClassifier()
trained_model.fit(X, y)

# 2. Tell Python to create the outputs folder if it is missing
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. Save the trained model into the outputs folder
model_save_path = os.path.join(output_dir, "youtube_category_assignable_model.pkl")
with open(model_save_path, "wb") as f:
    pickle.dump(trained_model, f)

print("Model saved successfully in the outputs folder!")
