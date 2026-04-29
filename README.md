# 🏭 SupplyAI — Assistant IA pour la Chaîne d'Approvisionnement

<p align="center">
  <img src="platform.png" alt="Interface SupplyAI" width="90%"/>
  <br>
  <i>Interface principale de SupplyAI sous Streamlit — Tableau de bord, métriques d'inventaire et chat interactif</i>
</p>

---

## 📋 Table des matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Fonctionnalités](#-fonctionnalités)
3. [Capture d'écran](#-capture-décran)
4. [Architecture](#-architecture)
5. [Structure du projet](#-structure-du-projet)
6. [Prérequis](#-prérequis)
7. [Installation](#-installation)
8. [Format des données CSV](#-format-des-données-csv)
9. [Configuration](#-configuration)
10. [Utilisation](#-utilisation)
11. [Description des modules](#-description-des-modules)
12. [Moteur de détection des risques](#-moteur-de-détection-des-risques)
13. [Ingénierie des prompts](#-ingénierie-des-prompts)
14. [Dépannage](#-dépannage)
15. [Feuille de route](#-feuille-de-route)
16. [Licence](#-licence)

---

## 🎯 Vue d'ensemble

**SupplyAI** est un assistant intelligent conçu pour les responsables de la chaîne d'approvisionnement, les acheteurs et les logisticiens. Il combine la **recherche sémantique** (FAISS) et les **grands modèles de langage locaux** (Llama3 via Ollama) pour analyser un fichier d'inventaire CSV et répondre en langage naturel à des questions complexes telles que :

- *"Quels produits sont en rupture de stock imminente ?"*
- *"Quels sont les 5 articles les plus à risque ?"*
- *"Génère une recommandation de réapprovisionnement pour l'entrepôt Nord."*
- *"Analyse les risques fournisseurs et identifie les dépendances critiques."*

Le système fonctionne **entièrement en local** — aucune donnée sensible n'est envoyée dans le cloud.

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| 📥 **Chargement intelligent CSV** | Conversion automatique des lignes CSV en documents texte enrichis avec métadonnées |
| 🔍 **Recherche sémantique** | Index FAISS avec embeddings `all-MiniLM-L6-v2` (HuggingFace) pour une recherche rapide et pertinente |
| 🧠 **LLM local** | Llama3 via Ollama — inférence privée, sans clé API |
| ⚡ **Pipeline RAG** | Chaîne RetrievalQA avec prompt personnalisé pour l'expertise supply chain |
| 🚨 **Détection des risques** | Alertes automatiques sur stock bas, seuils de réapprovisionnement, délais fournisseurs |
| 💡 **Moteur de recommandations** | Suggestions quantifiées de réapprovisionnement avec niveaux de priorité |
| 🌐 **Interface Streamlit** | UI interactive avec historique de chat, affichage des sources et métriques en temps réel |
| 💾 **Persistance** | Index FAISS sauvegardé localement pour éviter de ré-embedder à chaque démarrage |
| 🏗️ **Architecture modulaire** | Code orienté objet, séparation des responsabilités, facilement extensible |

---

## 📸 Capture d'écran

Placez votre image `platform.png` dans le même dossier que ce README pour l'afficher automatiquement.

> **Note :** Si vous placez `platform.png` dans un sous-dossier (ex: `assets/`), mettez à jour le chemin dans la balise `<img>` en haut de ce fichier.

---

## 🏗️ Architecture

CSV → Loader → Documents → Embeddings → FAISS → RAG → LLM → UI

------------------------------------------------------------------------
## 🛠️ Prérequis

-   Python 3.11
-   Ollama

``` bash
ollama pull llama3
```

------------------------------------------------------------------------

## 📦 Installation

``` bash
git clone https://github.com/ton-username/SupplyAI.git
cd SupplyAI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📊 Format CSV

product_id,product_name,current_stock,reorder_level,supplier,lead_time_days,demand_forecast

------------------------------------------------------------------------

## ⚙️ Configuration

LLM: model_name = "llama3"

Embeddings: all-MiniLM-L6-v2

FAISS: k=5

------------------------------------------------------------------------

## 🚀 Utilisation

``` bash
python main.py
streamlit run app/streamlit_app.py
```

------------------------------------------------------------------------

## 🚨 Détection des risques

-   Stock critique
-   Rupture
-   Délais fournisseurs
