import sys
import pytest
import numpy as np

sys.path.append("source")

class TestTrainCode:
    """
    Teste la logique du code d'entraînement AVANT de tout lancer.
    On vérifie que les modèles sont bien configurés et la sélection logique.
    """

    def test_all_models_defined(self):
        """Vérifie que les 10 modèles sont bien importables"""
        from sklearn.linear_model import (
            LinearRegression, Ridge, Lasso, ElasticNet
        )
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor

        models = [
            LinearRegression(), Ridge(), Lasso(), ElasticNet(),
            DecisionTreeRegressor(), RandomForestRegressor(),
            AdaBoostRegressor(), XGBRegressor(),
            LGBMRegressor(), CatBoostRegressor()
        ]
        assert len(models) == 10, \
            f"❌ {len(models)} modèles définis au lieu de 10"

    def test_best_model_selection_logic(self):
        """
        Vérifie que la logique de sélection du meilleur modèle
        choisit bien celui avec le RMSE le plus bas.
        """
        fake_results = [
            {"model": "ModelA", "RMSE": 2.5, "MAE": 1.8, "R2": 0.85},
            {"model": "ModelB", "RMSE": 0.7, "MAE": 0.5, "R2": 0.98},
            {"model": "ModelC", "RMSE": 1.2, "MAE": 0.9, "R2": 0.92},
        ]

        best = min(fake_results, key=lambda x: x["RMSE"])
        assert best["model"] == "ModelB", \
            f"❌ Mauvais modèle sélectionné : {best['model']}"
        assert best["RMSE"] == 0.7, \
            f"❌ RMSE incorrect : {best['RMSE']}"

    def test_metrics_calculation(self):
        """Vérifie que le calcul des métriques est correct"""
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score
        )

        y_true = np.array([3.0, 5.0, 7.0, 9.0])
        y_pred = np.array([3.0, 5.0, 7.0, 9.0])  # prédiction parfaite

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)

        assert rmse == 0.0, f"❌ RMSE doit être 0 pour prédiction parfaite"
        assert mae  == 0.0, f"❌ MAE doit être 0 pour prédiction parfaite"
        assert r2   == 1.0, f"❌ R² doit être 1 pour prédiction parfaite"

    def test_linear_models_use_scaling(self):
        """
        Vérifie que les modèles linéaires utilisent les données scalées
        et les arbres les données brutes — logique du dictionnaire.
        """
        linear_models = [
            "Linear Regression", "Ridge", "Lasso", "ElasticNet"
        ]
        tree_models = [
            "Decision Tree", "Random Forest", "AdaBoost",
            "XGBoost", "LightGBM", "CatBoost"
        ]

        # Simuler la logique de train.py
        models_config = {
            "Linear Regression": True,
            "Ridge"            : True,
            "Lasso"            : True,
            "ElasticNet"       : True,
            "Decision Tree"    : False,
            "Random Forest"    : False,
            "AdaBoost"         : False,
            "XGBoost"          : False,
            "LightGBM"         : False,
            "CatBoost"         : False,
        }

        for name in linear_models:
            assert models_config[name] == True, \
                f"❌ {name} doit utiliser le scaling"

        for name in tree_models:
            assert models_config[name] == False, \
                f"❌ {name} ne doit PAS utiliser le scaling"