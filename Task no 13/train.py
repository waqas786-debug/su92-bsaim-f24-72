import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

print("--- Script Shuru ho rahi hai ---")

filename = 'road_accident_dataset.csv'

if os.path.exists(filename):
    print(f"File '{filename}' mil gayi hai!")
    df = pd.read_csv(filename)
    print("Data load ho gaya hai. Rows count:", len(df))
else:
    print(f"ERROR: '{filename}' file nahi mili! Kya ye isi folder mein hai?")
    exit()

# Features select karein
features = ['Speed Limit', 'Driver Alcohol Level', 'Weather Conditions', 'Road Type', 'Vehicle Condition']
X = df[features].copy()
y = df['Accident Severity'].copy()

print("Encoding shuru ho rahi hai...")
encoders = {}
for col in ['Weather Conditions', 'Road Type', 'Vehicle Condition']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

print("Model training shuru ho rahi hai (Isme thora waqt lag sakta hai)...")
model = RandomForestClassifier(n_estimators=50, max_depth=10)
model.fit(X, y)

print("Files save ho rahi hain...")
with open('accident_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

if os.path.exists('accident_model.pkl'):
    print("SUCCESS: 'accident_model.pkl' ban gayi hai!")
else:
    print("FAILED: File abhi bhi nahi bani.")

    