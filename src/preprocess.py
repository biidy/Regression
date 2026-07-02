import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare(filepath):
    """
    Charge le dataset et prépare les données pour l'entraînement.
    Retourne : X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
    """

    # 1. Chargement
    data = pd.read_csv(filepath)
    print(f"✅ Dataset chargé : {data.shape[0]} lignes, {data.shape[1]} colonnes")

    # 2. Vérifications basiques
    if data.isnull().sum().sum() > 0:
        print("⚠️  Valeurs manquantes détectées — imputation par médiane")
        data = data.fillna(data.median(numeric_only=True))
    else:
        print("✅ Aucune valeur manquante")

    if data.duplicated().sum() > 0:
        print(f"⚠️  {data.duplicated().sum()} doublons supprimés")
        data = data.drop_duplicates()
    else:
        print("✅ Aucun doublon")

    # 3. Séparation X et y
    target_col = "Sales ($)"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 4. Nettoyage des noms de colonnes (pour LightGBM)
    X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(col)) for col in X.columns]
    print(f"✅ Features : {X.columns.tolist()}")
    
    # 5. Sauvegarder les noms de colonnes ← NOUVEAU
    os.makedirs("models", exist_ok=True)
    feature_names = X.columns.tolist()
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print(f"✅ Noms de colonnes sauvegardés : {feature_names}")

    # 6. Split train/test AVANT le scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Split : {X_train.shape[0]} train / {X_test.shape[0]} test")

    # 7. Scaling pour les modèles linéaires
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 8. Sauvegarde du scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler sauvegardé : models/scaler.pkl")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test