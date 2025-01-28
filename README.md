# Introduction aux modèles de prédiction

## Contexte
Dans le cadre de ma formation de développeur en IA, et dans la continuité de l'exposition de données immobilières par API, ce projet m'a introduit au traitement de données pour l'entraînement et l'utilisation de modèles de prédiction.

## Prérequis
Avant de démarrer le projet, il est nécessaire d'installer certaines dépendances sur l'environnement de travail. Pour effectuer ces installations, vous pouvez éxécuter la commande suivante :
```bash
pip install -r requirements.txt
```

## Data
[Lien vers les données à utiliser](https://www.kaggle.com/datasets/benoitfavier/immobilier-france/data)

## Structure du projet
```bash
project/
│
├── data/
│   ├── idf_transactions.csv        # Fichier généré par data_process.py (cible les départements de l'IDF dans ce cas)
│   ├── transactions.csv            # Fichier généré par npz_to_csv.py
│   └── transactions.npz            # Fichier compressé à télécharger sur Kaggle
│
├── models/                         # Modèles générés par model_training.py
│   ├── model_dtr.pkl               # Scikit-learn DecisionTreeRegressor
│   └── model_rfr.pkl               # Scikit-learn RandomForestRegressor
│
├── src/
│   ├── scripts/
│   │   ├── data_process.py         # Nettoyage des données, filtrage pour cibler les départements de l'IDF
│   │   ├── model_testing.py        # Utilisation de Grid Search pour tester les performances des différents types de modèles
│   │   └── model_training.py       # Entraînement des modèles avec les paramètres choisis, génération des fichiers .pkl
│   │
│   ├── utils/
│   │   └── npz_to_csv.py           # Script de décompression d'un fichier .npz en .csv
│   │
│   └── main.py                     # Script de l'API
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt                # Dépendances à installer
```

## Procédure
1 / Télécharger le fichier `transactions.npz` sur kaggle (voir data) et le placer dans un dossier data/<br><br>
2 / Créer et remplir le fichier `.env` avec les chemins des différents fichiers du projet. Un template est fourni :
```py
# Data files
DATA_DIR="data/"
NPZ_FILE="transactions.npz"
MAIN_CSV_FILE="transactions.csv"
FILTERED_CSV_FILE="idf_transactions.csv"

# Regression models
MODELS_DIR="models/"
DTR_MODEL_FILE="model_dtr.pkl"
RFR_MODEL_FILE="model_rfr.pkl"
```
3 / Convertir le fichier en `.npz` en `.csv` en vous plaçant à la racine du projet et exécuter :
```bash
python src/utils/npz_to_csv.py
```
4 / Toujours depuis la racine, on prépare les données pour les modèles en exécutant :
```bash
python src/scripts/data_process.py
```
5 / (Optionnel) Tester les modèles avec `model_testing.py` :
```bash
python src/scripts/model_testing.py
```
6 / Entraîner et exporter les modèles avec `model_training.py` :
```bash
python src/scripts/model_training.py
```
7 / Lancer l'API en vous plaçant à la racine du projet et exécuter :
```bash
python src/main.py
```

## Documentation de l'API
Une fois le serveur lancé (par défaut à localhost:8000), vous pouvez accéder à la documentation interactive Swagger à l'adresse suivante : http://localhost:8000/docs

### GET /predict_dtr
Effectue une prédiction du prix au mètre carré à l'aide du modèle DecisionTreeRegressor.

#### Paramètres d'entrée :
- longitude (float, obligatoire) : Longitude de l'emplacement (ex: 48.862725)
- latitude (float, obligatoire) : Latitude de l'emplacement (ex: 2.287592)
- vefa (bool, obligatoire) : Achat en VEFA (Vente en l'État Futur d'Achèvement)
- n_pieces (int, obligatoire) : Nombre de pièces du bien
- surface_habitable (int, obligatoire) : Surface habitable en m²
- type_Batiment (enum, obligatoire) : Type de bâtiment. Valeurs possibles : Appartement, Maison

#### Réponse :
- `200 OK`:
```json
{
    "prediction": 3456.78
}
```
- `500 Internal Server Error`: En cas d'erreur interne, un message d'erreur est retourné.

### GET /predict_rfr
Effectue une prédiction du prix au mètre carré à l'aide du modèle RandomForestRegressor.

#### Paramètres d'entrée :
- longitude (float, obligatoire) : Longitude de l'emplacement (ex: 48.862725)
- latitude (float, obligatoire) : Latitude de l'emplacement (ex: 2.287592)
- vefa (bool, obligatoire) : Achat en VEFA (Vente en l'État Futur d'Achèvement)
- n_pieces (int, obligatoire) : Nombre de pièces du bien
- surface_habitable (int, obligatoire) : Surface habitable en m²
- type_Batiment (enum, obligatoire) : Type de bâtiment. Valeurs possibles : Appartement, Maison

#### Réponse :
- `200 OK`:
```json
{
    "prediction": 5678.90
}
```
- `500 Internal Server Error`: En cas d'erreur interne, un message d'erreur est retourné.

### Remarques
- Vérifiez que les fichiers `.pkl` des modèles sont bien générés et accessibles dans le dossier `models/`.
- Assurez-vous que tous les chemins dans `.env` sont corrects.