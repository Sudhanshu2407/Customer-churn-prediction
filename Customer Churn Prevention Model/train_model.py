import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
file=pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\Customer Churn Prevention Model\Churn_Modelling.csv")
data = pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\Customer Churn Prevention Model\Churn_Modelling.csv")

# Prepare data
data = data.drop(['apst','CustomerId', 'Surname', 'Exited'], axis=1)
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data['Geography'] = LabelEncoder().fit_transform(data['Geography'])

X = data
y = file['Exited']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model,r'C:\sudhanshu_projects\project-task-training-course\Customer Churn Prevention Model\random_forest_model.pkl')
joblib.dump(scaler,r'C:\sudhanshu_projects\project-task-training-course\Customer Churn Prevention Model\scaler.pkl')
