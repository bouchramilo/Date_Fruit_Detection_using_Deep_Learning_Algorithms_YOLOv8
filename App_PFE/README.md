

Installation du Projet

1. Créer un environnement virtuel:
"python -m venv venv"

2. Activer l'environnement virtuel:
".\venv\Scripts\activate"

3. Installer les dépendances:
"pip install flask flask_sqlalchemy ultralytics opencv-python"

4. Vérifier l'installation des bibliothèques:
"pip show flask" , 
"pip show flask_sqlalchemy" , 
"pip show ultralytics" , 
"pip show opencv-python" , 

5. Lancer l'application Flask:

"python app.py"



____________________
Dépendances de ce projet
Flask : pour la création de l'application web.
Flask-SQLAlchemy : pour gérer la base de données avec SQLAlchemy.
Ultralytics : pour utiliser les modèles YOLO pour la détection d'objets.
OpenCV : pour le traitement d'images et de vidéos.
____________________
