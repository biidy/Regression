import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import json
import joblib
import os


def load_and_prepare(filepath):
    #1 chargement de data
    data = pd.read_csv(filepath)
    print(f"dataset charge:{data.shape[0]} lignes et {data.shape[1]} colonnes")

    #2 verification basique
    if data.isnull().sum()>0:
        print("valeurs manquantes detectees")

    else:
        print("Aucune valeur manquantes")

    if data.duplicated().sum() > 0:
        print("il y a des doublons dans le dataset")
    else:
        print("aucune doublons")

    #3.separation des donnees 
    x= data.drop(columns=["Sales ($)"])
    y= data["Sales ($)"]

    #4 netoyage des colonnes pour eviter l'erreur durant l'entrainement de lightgbm
    x.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(col)) for col in x.columns]
    print(f"features:{x.columns.tolist()}")

    #5 sauvegarde des noms de colonnes 
    os.makedirs("models", exist_ok=True)
    feature_names = x.columns.tolist()
    with open ("models/feature_name.json", w) as f:
        json.dump(feature_names, f)
    print(f"les noms des colonnes sauvegardes:{feature_names}")

    #6 split train/ test avant scaling
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"split : {x_train.shape[0]} train / {x_test.shape[0]} test")

    #7 standardisastion pour les modele lineaire
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)#pas de fit

    #8 sauvegarde du standardisation
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("scaler sauvegarder")

    return x_train, x_test, x_train_scaled, x_test_scaled, y_train, y_test
    