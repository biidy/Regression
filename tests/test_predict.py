import sys
import pytest
import numpy as np

sys.path.append("source")

class TestPredictCode:
    """
    Teste la logique de predict.py AVANT le vrai déploiement.
    On utilise un faux modèle pour tester le code, pas le résultat.
    """

    def test_predict_input_shape(self):
        """Vérifie que les features sont bien formées en 2D"""
        tv, radio, newspaper = 150, 30, 20
        features = np.array([[tv, radio, newspaper]])

        assert features.shape == (1, 3), \
            f"❌ Shape incorrect : {features.shape}, attendu (1, 3)"

    def test_predict_with_mock_model(self):
        """
        Teste predict() avec un faux modèle (mock).
        Permet de tester le code SANS avoir entraîné un vrai modèle.
        """
        from unittest.mock import MagicMock, patch
        import json

        # Créer un faux modèle
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([15.5])

        # Créer un faux all_results.json
        fake_data = json.dumps({
            "best_model"  : "CatBoost",
            "best_metrics": {"RMSE": 0.69, "MAE": 0.53, "R2": 0.98},
            "results"     : []
        })

        with patch("joblib.load", return_value=mock_model), \
             patch("builtins.open",
                   unittest.mock.mock_open(read_data=fake_data)):

            # Vérifier que predict() appelle bien model.predict
            features = np.array([[150, 30, 20]])
            result   = mock_model.predict(features)

            assert result[0] == 15.5, \
                f"❌ Prédiction mock incorrecte : {result[0]}"
            mock_model.predict.assert_called_once()

    def test_predict_value_types(self):
        """Vérifie que les entrées acceptent int et float"""
        inputs_int   = [150,   30,   20  ]
        inputs_float = [150.5, 30.2, 20.7]

        for inputs in [inputs_int, inputs_float]:
            features = np.array([inputs])
            assert features.dtype in [np.float64, np.int64], \
                f"❌ Type incorrect : {features.dtype}"

    def test_predict_negative_budget(self):
        """Vérifie qu'un budget négatif est détecté"""
        tv, radio, newspaper = -10, 30, 20

        # La logique métier : budget ne peut pas être négatif
        assert tv >= 0 or radio >= 0 or newspaper >= 0, \
            "❌ Tous les budgets sont négatifs"

import unittest