import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
data_path = "Data1.csv"
columns = [
    'Age', 'Gender', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBloodSugar',
    'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'Slope', 'Ca', 'Thal', 'HeartDisease'
]
df = pd.read_csv(data_path, names=columns)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df = df.dropna(subset=['Age'])
df['Age'] = pd.cut(df['Age'], bins=4, labels=[0, 1, 2, 3])
df['Cholesterol'] = pd.to_numeric(df['Cholesterol'], errors='coerce')
df = df.dropna(subset=['Cholesterol'])
df['Cholesterol'] = pd.cut(df['Cholesterol'], bins=4, labels=[0, 1, 2, 3])
categorical_columns = ['ChestPainType', 'RestingECG', 'ExerciseAngina', 'Slope', 'Ca', 'Thal']
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
model = BayesianNetwork([
    ('Age', 'HeartDisease'),
    ('Gender', 'HeartDisease'),
    ('FastingBloodSugar', 'HeartDisease'),
    ('Cholesterol', 'HeartDisease')])
model.fit(df, estimator=MaximumLikelihoodEstimator)
assert model.check_model()
inference = VariableElimination(model)
result = inference.query(variables=['HeartDisease'], evidence={'Age': 1, 'Cholesterol': 2})
print("Prediction of Heart Disease based on Age = 1 and Cholesterol = 2:")
print(result)
