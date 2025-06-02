import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("credit.csv")

# Rename target column if needed
if 'default.payment.next.month' in df.columns:
    df.rename(columns={'default.payment.next.month': 'DEFAULT'}, inplace=True)

# Drop 'ID' column (not useful for prediction)
df.drop(columns=['ID'], inplace=True)

# Ensure correct data types
df['DEFAULT'] = df['DEFAULT'].astype(int)

# Split into features and target
X = df.drop("DEFAULT", axis=1)
y = df["DEFAULT"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Confusion matrix and accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# Feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

top_features = feature_importances.head(4)

# Output results
print("Confusion Matrix:\n", conf_matrix)
print("\nAccuracy:", acc)
print("\nTop 4 Important Features:\n", top_features)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
buffer.close()
plt.close()

