import joblib
import numpy as np
import json
import pandas as pd

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

    # 3. Charger les noms de colonnes ← NOUVEAU
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)

    # 4. Créer un DataFrame avec les bons noms de colonnes ← NOUVEAU
    #    Les valeurs sont dans le même ordre que lors de l'entraînement
    features = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=feature_names        # ← noms exacts utilisés à l'entraînement
    )


    # 5. Prédiction
    prediction = model.predict(features)

    return float(prediction[0]), model_name


if __name__ == "__main__":
    # Test rapide
    sales, model_name = predict(tv=150, radio=30, newspaper=20)
    print(f"Modèle utilisé : {model_name}")
    print(f"Ventes prédites : {sales:.2f} (milliers $)")