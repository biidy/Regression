import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Modèles linéaires
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Modèles à base d'arbres
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import sys
sys.path.append("src")
from preprocess import load_and_prepare

def evaluate_model(name, model, x_test, y_test):
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"model": name, "RMSE": round(rmse, 6),
                           "MAE" : round(mae, 6),
                           "R2"  : round(r2, 6)}

def train_all_models():
    #1 preparation des donnes
    print("preparation des donnees")
    x_train, x_test, x_train_scaled, x_test_scaled, y_train, y_test = \
         load_and_prepare("data/advertising.csv")
    
    # 2. Définition des modèles
    #    Tuple : (modèle, utilise_scaling)
    models = {
        "Linear Regression": (LinearRegression(),
                               True),
        "Ridge":             (Ridge(alpha=1.0),
                               True),
        "Lasso":             (Lasso(alpha=0.1),
                               True),
        "ElasticNet":        (ElasticNet(alpha=0.1, l1_ratio=0.5),
                               True),
        "Decision Tree":     (DecisionTreeRegressor(random_state=42),
                               False),
        "Random Forest":     (RandomForestRegressor(n_estimators=100,
                               random_state=42),          False),
        "AdaBoost":          (AdaBoostRegressor(n_estimators=100,
                               random_state=42),          False),
        "XGBoost":           (XGBRegressor(n_estimators=100,
                               random_state=42,
                               verbosity=0),              False),
        "LightGBM":          (LGBMRegressor(n_estimators=100,
                               random_state=42,
                               verbose=-1),               False),
        "CatBoost":          (CatBoostRegressor(iterations=100,
                               random_state=42,
                               verbose=0),                False),
    }
    #3. entrainement et evaluation des models
    
    results =[]
    trained_models ={}

    for name, (model, use_scaling) in models.items():
        x_tr = x_train_scaled if use_scaling else x_train
        x_te = x_test_scaled if use_scaling else x_test

        model.fit(x_tr, y_train)
        metrics = evaluate_model(name, model, x_te, y_test)
        results.append(metrics)
        trained_models[name] = model
        print(f"{name:<20} RMSE: {metrics['RMSE']:.4f} | "
              f"MAE: {metrics['MAE']:.4f} | R²: {metrics['R2']:.4f}")
        
    #4 selection de meilleur model
    best_result = min(results, key=lambda x: x["RMSE"])
    best_name = best_result["model"] 
    best_model  = trained_models[best_name]    


    print("-" * 60)
    print(f"\n Meilleur modèle : {best_name}")
    print(f"   RMSE : {best_result['RMSE']:.4f}")
    print(f"   MAE  : {best_result['MAE']:.4f}")
    print(f"   R²   : {best_result['R2']:.4f}")

    # 5. Sauvegarde du meilleur modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    print("\n Meilleur modèle sauvegardé : models/best_model.pkl")

    # 6. Sauvegarde des résultats complets (utilisé par app.py)
    with open("models/all_results.json", "w") as f:
        json.dump({
            "results"     : results,
            "best_model"  : best_name,
            "best_metrics": best_result
        }, f, indent=2)
    print(" Résultats sauvegardés : models/all_results.json")

    return best_result

if __name__ == "__main__":
    train_all_models()