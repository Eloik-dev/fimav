.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/fimav.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/fimav
    .. image:: https://readthedocs.org/projects/fimav/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://fimav.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/fimav/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/fimav
    .. image:: https://img.shields.io/pypi/v/fimav.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/fimav/
    .. image:: https://img.shields.io/conda/vn/conda-forge/fimav.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/fimav
    .. image:: https://pepy.tech/badge/fimav/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/fimav
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/fimav

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=====
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

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
