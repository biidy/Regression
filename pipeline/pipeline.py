"""
Pipeline MLRun : orchestre toutes les étapes du projet.
Peut être lancé localement ou dans le CI/CD.
"""
import os
import sys
import json
import subprocess

sys.path.append("source")

def step_preprocess():
    """Etape 1 : Vérification et préparation des données"""
    print("\n" + "=" * 50)
    print("ETAPE 1 : Preprocessing")
    print("=" * 50)

    from preprocess import load_and_prepare
    result = load_and_prepare("data/advertising.csv")
    print("✅ Preprocessing terminé")
    return result

def step_train():
    """Etape 2 : Entraînement et sélection du meilleur modèle"""
    print("\n" + "=" * 50)
    print("ETAPE 2 : Entraînement des modèles")
    print("=" * 50)

    from train import train_all_models
    best = train_all_models()
    print(f"✅ Entraînement terminé — Meilleur : {best['model']}")
    return best

def step_test():
    """Etape 3 : Tests automatiques"""
    print("\n" + "=" * 50)
    print("ETAPE 3 : Tests automatiques")
    print("=" * 50)

    result = subprocess.run(
        ["pytest", "tests/", "-v", "--tb=short"],
        capture_output=False
    )

    if result.returncode != 0:
        raise Exception("❌ Tests échoués — pipeline arrêté")

    print("✅ Tous les tests passés")

def step_predict_example():
    """Etape 4 : Exemple de prédiction avec le meilleur modèle"""
    print("\n" + "=" * 50)
    print("ETAPE 4 : Exemple de prédiction")
    print("=" * 50)

    from predict import predict

    exemples = [
        {"tv": 150, "radio": 30, "newspaper": 20},
        {"tv": 200, "radio": 40, "newspaper": 10},
        {"tv": 50,  "radio": 10, "newspaper": 5 },
    ]

    for ex in exemples:
        sales, model_name = predict(**ex)
        print(f"TV={ex['tv']:3d} | Radio={ex['radio']:2d} | "
              f"News={ex['newspaper']:3d} → Ventes : {sales:.2f}k$")

    print(f"\n✅ Prédictions via : {model_name}")

def run_pipeline():
    """Lance toutes les étapes dans l'ordre"""
    print("\n🚀 LANCEMENT DU PIPELINE COMPLET")
    print("=" * 50)

    os.makedirs("models", exist_ok=True)

    try:
        step_preprocess()
        step_train()
        step_test()
        step_predict_example()

        # Résumé final
        with open("models/all_results.json") as f:
            data = json.load(f)

        print("\n" + "=" * 50)
        print("✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS")
        print(f"   Meilleur modèle : {data['best_model']}")
        print(f"   R²   : {data['best_metrics']['R2']:.4f}")
        print(f"   RMSE : {data['best_metrics']['RMSE']:.4f}")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ PIPELINE ÉCHOUÉ : {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()