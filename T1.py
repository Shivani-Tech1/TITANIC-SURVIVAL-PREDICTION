# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Step 2: Load the Dataset
data = pd.read_csv('Titanic-Dataset.csv')

# Step 3: Explore the Data
print("First 5 rows of the dataset:")
print(data.head())
print("\nData Information:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Step 4: Data Preprocessing
# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked with the most frequent value

# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Step 5: Define Features and Target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Step 6: Standardize the Data (Optional)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Train the Model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 9: Make Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Feature Importance (for Logistic Regression, we can use the coefficients)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
feature_importance['AbsCoefficient'] = feature_importance['Coefficient'].abs()
feature_importance.sort_values(by='AbsCoefficient', ascending=False, inplace=True)

# Plot feature importance
sns.barplot(x='AbsCoefficient', y='Feature', data=feature_importance)
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.show()
