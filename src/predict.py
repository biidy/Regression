import joblib
import numpy as np
import json

def predict(tv, radio, newspaper):
    """
    Prédit les ventes à partir des budgets publicitaires.

    Paramètres :
        tv        : budget TV (float)
        radio     : budget Radio (float)
        newspaper : budget Newspaper (float)

    Retourne :
        prediction : ventes prédites (float)
        model_name : nom du modèle utilisé (str)
    """

    # 1. Charger le meilleur modèle
    model = joblib.load("models/best_model.pkl")

    # 2. Charger le nom du modèle utilisé
    with open("models/all_results.json") as f:
        data = json.load(f)
    model_name = data["best_model"]

    # 3. Préparer les features
    #    Note : pas besoin de scaling car le meilleur modèle
    #    est CatBoost (arbre) — pas sensible à l'échelle
    features = np.array([[tv, radio, newspaper]])

    # 4. Prédiction
    prediction = model.predict(features)

    return float(prediction[0]), model_name


if __name__ == "__main__":
    # Test rapide
    sales, model_name = predict(tv=150, radio=30, newspaper=20)
    print(f"Modèle utilisé : {model_name}")
    print(f"Ventes prédites : {sales:.2f} (milliers $)")