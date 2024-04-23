# SIMPLON DEV IA | Brief 5

## Développer une API REST pour exposer un modèle prédictif avec des données immobilières

### Prérequis

Avant de démarrer le projet, il est nécessaire d'installer certaines dépendances sur l'environnement de travail. Pour effectuer ces installations, vous pouvez éxécuter la commande suivante :
```bash
pip install -r requirements.txt
```

### Data

[Lien vers les données à utiliser](https://www.kaggle.com/datasets/benoitfavier/immobilier-france/data)

### Structure du projet

```bash
project/
│
├── data/
│   ├── idf_transactions.csv    # Fichier généré par data_process.py (cible les départements de l'IDF dans ce cas)
│   ├── transactions.csv    # Fichier généré par npz_to_csv.py
│   └── transactions.npz    # Fichier compressé à télécharger sur Kaggle
│
├── models/    # Modèles générés par model_training.py
│   ├── model_dtr.pkl    # Scikit-learn DecisionTreeRegressor
│   └── model_rfr.pkl    # Scikit-learn RandomForestRegressor
│
├── utilities/
│   ├── config.py    # Config des chemins de dossiers, fichiers générés par les scripts
│   ├── data_process.py    # Nettoyage des données, filtrage pour cibler les départements de l'IDF
│   ├── model_testing.py    # Utilisation de Grid Search pour tester les performances des différents types de modèles
│   ├── model_training.py    # Entraînement des modèles avec les paramètres choisis, génération des fichiers .pkl
│   └── npz_to_csv.py    # Script de décompression d'un fichier .npz en .csv
│
├── .gitignore
├── main.py    # API script
├── README.md
└── requirements.txt
```

### Procédure

1 / Télécharger le fichier transactions.npz sur kaggle (voir data) et le placer dans un dossier data/<br><br>
2 / Convertir le fichier en .npz en .csv en vous plaçant dans le dossier utilities/ et exécuter :
```bash
python npz_to_csv.py
```
3 / Toujours dans le dossier utilities/, on prépare les données pour les modèles en exécutant :
```bash
python data_process.py
```
4 / (Optionnel) Tester les modèles avec utilities/model_testing.py :
```bash
python model_testing.py
```
5 / Entraîner et exporter les modèles avec utilities/model_training.py :
```bash
python model_training.py
```
6 / Lancer l'API en vous plaçant à la racine du projet et exécuter :
```bash
python main.py
```

### Doc

Pour un hôte (localhost) et un port (8000), accédez à http://localhost:8000/docs sur votre navigateur, une fois le serveur lancé, pour accéder aux fonctionnalités de l'API.