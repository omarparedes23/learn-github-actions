from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

os.makedirs("model", exist_ok=True)

author = os.environ.get("MODEL_AUTHOR", "desconocido")

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "model/model.pkl")
print(f"Modelo entrenado por: {author}")
print("Guardado en model/model.pkl")
