import gradio as gr
import joblib
import numpy as np
import json
import pandas as pd
import os

# ── Chargement du modèle et des résultats ──────────────────────
model = joblib.load("models/best_model.pkl")

with open("models/all_results.json") as f:
    all_data = json.load(f)

best_name    = all_data["best_model"]
best_metrics = all_data["best_metrics"]
all_results  = all_data["results"]

# ── Fonctions ───────────────────────────────────────────────────
def predict_sales(tv, radio, newspaper):
    """Prédit les ventes à partir des budgets publicitaires"""
    features   = np.array([[tv, radio, newspaper]])
    prediction = model.predict(features)[0]

    return (
        f"💰 Ventes prédites : {prediction:.2f} (milliers $)",
        f"🏆 Modèle utilisé : {best_name}",
        f"📊 R² : {best_metrics['R2']:.4f} | "
        f"RMSE : {best_metrics['RMSE']:.4f} | "
        f"MAE : {best_metrics['MAE']:.4f}"
    )

def get_comparison_table():
    """Retourne le tableau comparatif de tous les modèles"""
    df = pd.DataFrame(all_results)
    df = df.sort_values("RMSE").reset_index(drop=True)
    df.index += 1  # commencer à 1 au lieu de 0
    df.columns = ["Modèle", "RMSE ↓", "MAE ↓", "R² ↑"]
    return df

# ── Interface Gradio ────────────────────────────────────────────
with gr.Blocks(
    title="Advertising Sales Prediction",
    theme=gr.themes.Soft()
) as app:

    # Header
    gr.Markdown("""
    # 📈 Prédiction des Ventes Publicitaires
    > Modèle sélectionné automatiquement parmi **10 algorithmes** testés et comparés.
    """)

    # ── Onglet 1 : Prédiction ──
    with gr.Tab("🎯 Prédiction"):

        gr.Markdown("### Entrez vos budgets publicitaires")

        with gr.Row():
            tv        = gr.Slider(
                minimum=0, maximum=300, value=150, step=1,
                label="📺 Budget TV ($)"
            )
            radio     = gr.Slider(
                minimum=0, maximum=50,  value=25,  step=1,
                label="📻 Budget Radio ($)"
            )
            newspaper = gr.Slider(
                minimum=0, maximum=120, value=30,  step=1,
                label="📰 Budget Newspaper ($)"
            )

        btn = gr.Button("🔍 Prédire les ventes", variant="primary")

        with gr.Row():
            out_prediction = gr.Textbox(label="Résultat")
            out_model      = gr.Textbox(label="Modèle")
            out_metrics    = gr.Textbox(label="Métriques")

        btn.click(
            fn=predict_sales,
            inputs=[tv, radio, newspaper],
            outputs=[out_prediction, out_model, out_metrics]
        )

        gr.Markdown("""
        ### 💡 Comment interpréter ?
        - **Ventes prédites** : estimation des ventes en milliers de dollars
        - **R²** : plus proche de 1 = meilleur modèle (ici > 0.98 = excellent)
        - **RMSE** : erreur moyenne en milliers de dollars
        """)

    # ── Onglet 2 : Comparaison des modèles ──
    with gr.Tab("📊 Comparaison des modèles"):

        gr.Markdown(f"""
        ### Résultats de tous les modèles testés
        **🏆 Meilleur modèle sélectionné automatiquement : {best_name}**

        Critère de sélection : **RMSE le plus bas** (erreur minimale)
        """)

        gr.Dataframe(
            value=get_comparison_table(),
            label="Comparaison des 10 modèles (triés par RMSE)",
            interactive=False
        )

        gr.Markdown("""
        ### 📖 Légende des métriques
        | Métrique | Signification | Objectif |
        |---|---|---|
        | **RMSE** | Erreur quadratique moyenne | Minimiser ↓ |
        | **MAE**  | Erreur absolue moyenne | Minimiser ↓ |
        | **R²**   | Variance expliquée (0 à 1) | Maximiser ↑ |
        """)

    # ── Onglet 3 : À propos ──
    with gr.Tab("ℹ️ À propos"):
        gr.Markdown(f"""
        ### À propos de ce projet

        **Objectif** : Prédire les ventes en fonction des budgets
        publicitaires (TV, Radio, Newspaper)

        **Dataset** : Advertising Sales Dataset (Kaggle)

        **Modèles testés** :
        - Régression Linéaire, Ridge, Lasso, ElasticNet
        - Decision Tree, Random Forest, AdaBoost
        - XGBoost, LightGBM, CatBoost

        **Pipeline CI/CD** :
        - GitHub Actions entraîne et compare automatiquement les 10 modèles
        - Le meilleur modèle est sélectionné selon le RMSE le plus bas
        - Déploiement automatique sur HuggingFace Spaces

        **Meilleur modèle** : {best_name}
        - R²   : {best_metrics['R2']:.4f}
        - RMSE : {best_metrics['RMSE']:.4f}
        - MAE  : {best_metrics['MAE']:.4f}
        """)

if __name__ == "__main__":
    app.launch()