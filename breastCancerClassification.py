# Import necessary libraries
#import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import plot_tree


# Load the data
df = pd.read_csv('breast_cancer_dataset.csv')

# Display first few rows
df.head()
# Check for datatypes and missing values
df.info()
df.isnull().sum()
#Assigning y to 'class' and X to the rest of the variables
X = df.drop('class', axis=1)
y = df['class']

# Convert class labels to categorical if needed (2=Malignant, 4=Benign)
y = y.astype('category')
# Step 1: Split the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 2: Create a decision tree model using training data
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
# Step 3: Test the model and show confusion matrix and accuracy
y_pred = clf.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Decision Tree Classifier: {accuracy:.2f}")
# Step 4: Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Malignant', 'Benign'])
plt.title("Decision Tree for Breast Cancer Classification")
plt.show()
