---
title: Advertising Sales Prediction
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.19.0"
app_file: app/app.py
pinned: false
---

# 📈 Advertising Sales Prediction

Prédiction des ventes publicitaires en fonction des budgets TV, Radio et Newspaper.

##  Objectif
Comparer 10 algorithmes de régression et déployer automatiquement le meilleur modèle.

##  Modèles testés
- Régression Linéaire, Ridge, Lasso, ElasticNet
- Decision Tree, Random Forest, AdaBoost
- XGBoost, LightGBM, CatBoost

##  Sélection automatique
Le meilleur modèle est sélectionné automatiquement selon le **RMSE le plus bas**.

##  Pipeline CI/CD
- GitHub Actions entraîne et compare les 10 modèles
- Le meilleur modèle est déployé automatiquement sur HuggingFace