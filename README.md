---
title: Whisper Transcription
emoji: 🎙️
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
license: mit
short_description: Application web de transcription audio utilisant Whisper
---

# 🎙️ Whisper Transcription

Application web de transcription audio automatique basée sur [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Fonctionnalités

- Upload de fichier audio ou enregistrement via microphone
- Choix du modèle Whisper : `small`, `medium`, `large-v3`, `distil-large-v3`
- Auto-détection de la langue avec score de confiance
- Transcription avec horodatage `[début -> fin]`
- Formats supportés : m4a, mp3, wav, ogg, flac, mp4, webm

---

## 🚀 Déploiement sur Hugging Face Spaces

### Prérequis

- Un compte [Hugging Face](https://huggingface.co)
- Git installé sur votre poste

### Étapes

1. Créer un nouveau Space sur https://huggingface.co/new-space
   - **SDK** : Gradio
   - **Visibilité** : Public ou Private

2. Cloner le repo du Space et y copier les fichiers du projet :
   ```bash
   git clone https://huggingface.co/spaces/<votre-username>/<nom-du-space>
   cd <nom-du-space>
   cp /chemin/vers/app.py .
   cp /chemin/vers/requirements.txt .
   cp /chemin/vers/README.md .
   ```

3. Pousser les fichiers :
   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```

4. HF Spaces installe automatiquement les dépendances depuis `requirements.txt` et démarre l'application.

> **Note :** Le modèle Whisper est téléchargé depuis HuggingFace au premier lancement. Préférez `small` ou `medium` sur le CPU gratuit — `large-v3` peut dépasser les limites de mémoire (16 Go RAM).

---

## 🐳 Lancement en local avec Docker

### Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installé et démarré

### 1. Construire l'image

```bash
docker build -t whisper-transcription .
```

### 2. Lancer le container

```bash
docker run -d --name whisper-app -p 7860:7860 whisper-transcription
```

L'interface est ensuite accessible sur **http://localhost:7860**.

### 3. Persister le cache des modèles (recommandé)

Sans volume, les modèles Whisper sont re-téléchargés à chaque recréation du container.
Pour les conserver :

```bash
docker run -d --name whisper-app -p 7860:7860 \
  -v whisper-cache:/app/.cache \
  whisper-transcription
```

### Commandes utiles

```bash
# Voir les logs en temps réel
docker logs -f whisper-app

# Arrêter le container
docker stop whisper-app

# Redémarrer
docker start whisper-app

# Supprimer le container
docker rm -f whisper-app

# Supprimer l'image
docker rmi whisper-transcription
```
