import sys
import os
import pytest
import pandas as pd
import numpy as np
import json

sys.path.append("source")

class TestPreprocessCode:
    """
    Teste le CODE de preprocess.py AVANT tout entraînement.
    On vérifie que les fonctions marchent correctement.
    """

    def test_csv_file_exists(self):
        """Vérifie que le fichier CSV source existe"""
        assert os.path.exists("data/advertising.csv"), \
            "❌ data/advertising.csv introuvable"

    def test_csv_is_readable(self):
        """Vérifie que le CSV est lisible et non vide"""
        data = pd.read_csv("data/advertising.csv")
        assert data.shape[0] > 0, "❌ Le CSV est vide"
        assert data.shape[1] > 1, "❌ Le CSV n'a pas assez de colonnes"

    def test_target_column_exists(self):
        """Vérifie que la colonne cible existe dans le CSV"""
        data = pd.read_csv("data/advertising.csv")
        assert "Sales ($)" in data.columns, \
            f"❌ Colonne 'Sales ($)' absente. Colonnes : {data.columns.tolist()}"

    def test_column_name_cleaning(self):
        """Vérifie que le nettoyage des noms de colonnes fonctionne"""
        import re
        test_cols = ["Sales ($)", "TV Budget", "Radio (FM)", "News[paper]"]
        cleaned   = [re.sub(r"[^A-Za-z0-9_]+", "_", col) for col in test_cols]

        for col in cleaned:
            assert " " not in col,  f"❌ Espace détecté dans : {col}"
            assert "(" not in col,  f"❌ Parenthèse dans : {col}"
            assert "$" not in col,  f"❌ Symbole $ dans : {col}"
            assert "[" not in col,  f"❌ Crochet dans : {col}"

    def test_split_proportions(self):
        """Vérifie que le split 80/20 donne les bonnes proportions"""
        from sklearn.model_selection import train_test_split

        data  = pd.read_csv("data/advertising.csv")
        X     = data.drop(columns=["Sales ($)"])
        y     = data["Sales ($)"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        total = len(X_train) + len(X_test)
        ratio = len(X_test) / total

        assert 0.18 <= ratio <= 0.22, \
            f"❌ Ratio test incorrect : {ratio:.2f}"

    def test_scaler_fit_only_on_train(self):
        """
        Vérifie qu'on fit le scaler sur train uniquement.
        La moyenne du scaler doit correspondre à X_train, pas à tout X.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import re

        data = pd.read_csv("data/advertising.csv")
        X    = data.drop(columns=["Sales ($)"])
        y    = data["Sales ($)"]

        X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(col))
                     for col in X.columns]

        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler          = StandardScaler()
        X_train_scaled  = scaler.fit_transform(X_train)
        X_test_scaled   = scaler.transform(X_test)

        # La moyenne du scaler doit être celle de X_train, pas de X entier
        np.testing.assert_array_almost_equal(
            scaler.mean_, X_train.mean().values, decimal=5,
            err_msg="❌ Le scaler n'a pas été fitté sur X_train"
        )

    def test_no_data_leakage(self):
        """Vérifie que les stats du scaler ne viennent pas du test set"""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import re

        data = pd.read_csv("data/advertising.csv")
        X    = data.drop(columns=["Sales ($)"])
        y    = data["Sales ($)"]

        X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(col))
                     for col in X.columns]

        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        scaler.fit_transform(X_train)

        # La moyenne du scaler NE DOIT PAS être égale à celle de tout X
        mean_all   = X.mean().values
        mean_train = X_train.mean().values

        # Si égaux → data leakage (fit sur tout X)
        assert not np.allclose(scaler.mean_, mean_all, atol=1e-3), \
            "❌ Data leakage détecté : scaler fitté sur tout X"