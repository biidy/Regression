import sys
import os
import json
import numpy as np
import pytest

# Accès aux fichiers source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# ── Tests Preprocessing ─────────────────────────────────────────
class TestPreprocessing:

    def test_data_file_exists(self):
        """Vérifie que le fichier de données existe"""
        assert os.path.exists("data/advertising.csv"), \
            "❌ data/advertising.csv introuvable"
        print("✅ Fichier data trouvé")

    def test_load_and_prepare(self):
        """Vérifie que le preprocessing s'exécute sans erreur"""
        from preprocess import load_and_prepare
        result = load_and_prepare("data/advertising.csv")

        # Doit retourner 6 éléments
        assert len(result) == 6, \
            f"❌ load_and_prepare doit retourner 6 éléments, retourne {len(result)}"
        print("✅ Preprocessing retourne 6 éléments")

    def test_split_sizes(self):
        """Vérifie que le split 80/20 est correct"""
        from preprocess import load_and_prepare
        X_train, X_test, _, _, y_train, y_test = \
            load_and_prepare("data/advertising.csv")

        total = len(X_train) + len(X_test)
        ratio = len(X_test) / total

        assert 0.18 <= ratio <= 0.22, \
            f"❌ Ratio test incorrect : {ratio:.2f} (attendu ~0.20)"
        print(f"✅ Split correct : {len(X_train)} train / {len(X_test)} test")

    def test_no_missing_values(self):
        """Vérifie qu'il n'y a pas de valeurs manquantes après preprocessing"""
        from preprocess import load_and_prepare
        import pandas as pd
        X_train, X_test, _, _, _, _ = load_and_prepare("data/advertising.csv")

        X_train_df = pd.DataFrame(X_train)
        X_test_df  = pd.DataFrame(X_test)

        assert X_train_df.isnull().sum().sum() == 0, \
            "❌ Valeurs manquantes dans X_train"
        assert X_test_df.isnull().sum().sum() == 0, \
            "❌ Valeurs manquantes dans X_test"
        print("✅ Aucune valeur manquante")

    def test_scaler_saved(self):
        """Vérifie que le scaler a été sauvegardé"""
        assert os.path.exists("models/scaler.pkl"), \
            "❌ models/scaler.pkl introuvable"
        print("✅ Scaler sauvegardé")


# ── Tests Entraînement ──────────────────────────────────────────
class TestTraining:

    def test_best_model_saved(self):
        """Vérifie que le meilleur modèle a été sauvegardé"""
        assert os.path.exists("models/best_model.pkl"), \
            "❌ models/best_model.pkl introuvable"
        print("✅ best_model.pkl existe")

    def test_results_file_saved(self):
        """Vérifie que le fichier de résultats a été sauvegardé"""
        assert os.path.exists("models/all_results.json"), \
            "❌ models/all_results.json introuvable"
        print("✅ all_results.json existe")

    def test_all_models_tested(self):
        """Vérifie que les 10 modèles ont bien été testés"""
        with open("models/all_results.json") as f:
            data = json.load(f)

        assert len(data["results"]) == 10, \
            f"❌ {len(data['results'])} modèles testés au lieu de 10"
        print(f"✅ 10 modèles testés")

    def test_results_have_required_keys(self):
        """Vérifie que chaque résultat contient RMSE, MAE, R2"""
        with open("models/all_results.json") as f:
            data = json.load(f)

        for result in data["results"]:
            assert "model" in result, "❌ Clé 'model' manquante"
            assert "RMSE"  in result, "❌ Clé 'RMSE' manquante"
            assert "MAE"   in result, "❌ Clé 'MAE' manquante"
            assert "R2"    in result, "❌ Clé 'R2' manquante"
        print("✅ Toutes les métriques présentes")

    def test_best_model_performance(self):
        """Vérifie que le meilleur modèle atteint un niveau minimum"""
        with open("models/all_results.json") as f:
            data = json.load(f)

        best = data["best_metrics"]
        assert best["R2"]   > 0.90, \
            f"❌ R² insuffisant : {best['R2']:.4f} (minimum 0.90)"
        assert best["RMSE"] < 3.00, \
            f"❌ RMSE trop élevé : {best['RMSE']:.4f} (maximum 3.00)"
        print(f"✅ Performance OK — R²={best['R2']:.4f}, "
              f"RMSE={best['RMSE']:.4f}")

    def test_best_model_is_in_results(self):
        """Vérifie que le meilleur modèle fait partie des résultats"""
        with open("models/all_results.json") as f:
            data = json.load(f)

        model_names = [r["model"] for r in data["results"]]
        assert data["best_model"] in model_names, \
            f"❌ {data['best_model']} absent des résultats"
        print(f"✅ Meilleur modèle valide : {data['best_model']}")


# ── Tests Prédiction ────────────────────────────────────────────
class TestPrediction:

    def test_predict_returns_float(self):
        """Vérifie que predict retourne un float"""
        from predict import predict
        result, _ = predict(tv=150, radio=30, newspaper=20)
        assert isinstance(result, float), \
            f"❌ predict doit retourner un float, reçu {type(result)}"
        print(f"✅ Prédiction retourne un float : {result:.4f}")

    def test_predict_returns_model_name(self):
        """Vérifie que predict retourne le nom du modèle"""
        from predict import predict
        _, model_name = predict(tv=150, radio=30, newspaper=20)
        assert isinstance(model_name, str), \
            "❌ predict doit retourner le nom du modèle"
        assert len(model_name) > 0, \
            "❌ Le nom du modèle est vide"
        print(f"✅ Modèle utilisé : {model_name}")

    def test_predict_reasonable_value(self):
        """Vérifie que la prédiction est dans une plage raisonnable"""
        from predict import predict
        result, _ = predict(tv=150, radio=30, newspaper=20)
        assert 0 < result < 100, \
            f"❌ Prédiction hors limites : {result:.4f}"
        print(f"✅ Prédiction dans les limites : {result:.4f}")

    def test_predict_zero_budget(self):
        """Vérifie le comportement avec un budget nul"""
        from predict import predict
        result, _ = predict(tv=0, radio=0, newspaper=0)
        assert isinstance(result, float), \
            "❌ Erreur avec budget nul"
        print(f"✅ Budget nul géré : {result:.4f}")

    def test_predict_max_budget(self):
        """Vérifie le comportement avec un budget maximum"""
        from predict import predict
        result, _ = predict(tv=300, radio=50, newspaper=120)
        assert isinstance(result, float), \
            "❌ Erreur avec budget maximum"
        print(f"✅ Budget maximum géré : {result:.4f}")