# sales_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

print("🚀 Début du script Sales Prediction...")

# Charger les données
try:
    df = pd.read_csv('sales_data.csv')
    print("✅ Données chargées !")
except FileNotFoundError:
    print("❌ Fichier 'sales_data.csv' introuvable !")
    print("Assure-toi qu'il est dans le même dossier que ce script.")
    exit()

# Afficher les premières lignes et les colonnes
print("\n🔍 Aperçu des données :")
print(df.head())
print(f"\n📌 Colonnes : {list(df.columns)}")

# Supprimer la première colonne (vide ou index)
df = df.drop(df.columns[0], axis=1)

# Vérifier que les colonnes sont bonnes
print(f"\n✅ Colonnes après nettoyage : {list(df.columns)}")

# Séparer X et y
X = df[['TV', 'Radio', 'Newspaper']]  # Caractéristiques
y = df['Sales']  # Cible

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
print("\n🧠 Entraînement du modèle...")
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)

# Évaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n✅ MSE : {mse:.2f}")
print(f"✅ R² Score : {r2:.2f}")

# Visualisation : Prédictions vs Réelles
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Ventes réelles')
plt.ylabel('Ventes prédites')
plt.title('Ventes réelles vs Ventes prédites')
plt.show()

# Sauvegarder le modèle
joblib.dump(model, 'sales_model.pkl')
print("\n💾 Modèle sauvegardé sous 'sales_model.pkl'")

# Insights marketing : coefficients
coefficients = pd.DataFrame(model.coef_, index=['TV', 'Radio', 'Newspaper'], columns=['Coefficient'])
print("\n📈 Insights marketing :")
print(coefficients.sort_values(by='Coefficient', ascending=False))

input("\n🎉 Projet terminé ! Appuie sur Entrée pour quitter...")