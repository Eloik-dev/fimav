fimav
=====

Logiciel Python pour la reconnaissance faciale et la classification d'émotions

### Préalables
Installation de GStreamer pour le rendu vidéo : https://gstreamer.freedesktop.org/documentation/installing/index.html?gi-language=c

### Environnement de développement
python -m venv .venv
source .venv/bin/activate

### Installation pour développement (WSL2/Autres)
pip install --no-deps -e .

### Installation Raspberry Pi 5
pip install --no-deps .
pipdeptree --reverse --packages opencv-python

### Exécution
fimav-run
