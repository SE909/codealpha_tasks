Ce projet fait partie du programme de stage de CodeAlpha.  
Il consiste à classifier les espèces de fleurs d’Iris (setosa, versicolor, virginica) à partir de mesures botaniques.

🛠️ Outils & Bibliothèques
- Python 3.14
- Scikit-learn (`load_iris`, `RandomForestClassifier`)
- Pandas, Matplotlib, Seaborn
- Joblib (sauvegarde du modèle)

📊 Résultats
- Précision du modèle : 100.0%
- Méthode : Random Forest avec 100 arbres
- Jeu de données : [Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) (intégré à scikit-learn)

▶️ Comment exécuter
```bash
pip install -r requirements.txt
python iris_classification.py