# Challenge OPSIE SISE

Application Streamlit d'analyse de logs firewall, avec visualisation opérationnelle, scoring SOC et module ML (ACP + KMeans).

## Objectif

Le projet permet de :
- charger et normaliser des logs réseau,
- analyser les événements `DENY` et `PERMIT`,
- identifier les IP et ports les plus ciblés,
- visualiser les tendances horaires et protocolaires,
- appliquer un score de risque SOC,
- explorer une typologie du trafic via ACP et KMeans,
- générer des commentaires d'aide à l'interprétation via Mistral (optionnel).

## Stack technique

- Python 3.12
- Streamlit
- Pandas / NumPy
- Plotly
- Scikit-learn
- Docker / Docker Compose

## Structure du projet

```text
.
├── main.py
├── config.py
├── pages/
│   ├── 1_Analyses.py
│   └── 2_ML&interpretation.py
├── data/
│   └── cleaned_logs.csv
├── requirements.txt
├── dockerfile
├── docker-compose.yml
└── .dockerignore
```

## Données

Le chargement est géré dans `config.py`.
Le fichier prioritaire attendu est :

- `data/cleaned_logs.csv`

Colonnes prises en charge :
- `datetime` (timestamp)
- `ipsrc` (IP source)
- `ipdst` (IP destination)
- `proto` (protocole)
- `dstport` (port destination)
- `action` (`DENY` ou `PERMIT`)
- `policyid` (règle)
- `interface`, `interface_out`

## Lancement en local (sans Docker)

### 1) Créer et activer un environnement virtuel

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Installer les dépendances

```powershell
pip install -r requirements.txt
```

### 3) Lancer l'application

```powershell
streamlit run main.py
```

Accès :
- http://localhost:8501

## Lancement avec Docker

### Option recommandée

```powershell
docker compose up --build -d
```

Accès :
- http://localhost:8501

Arrêt :

```powershell
docker compose down
```

### Option avec clé Mistral (facultatif)

Créer un fichier `.env` à la racine :

```dotenv
MISTRAL_API_KEY=VOTRE_CLE
```

Puis lancer :

```powershell
docker compose up --build -d
```

## Fonctionnalités principales

### Page Analyses

- KPIs globaux (`TOTAL`, `DENY`, `PERMIT`, IP uniques)
- Score SOC unifié par IP
- Timeline `DENY` vs `PERMIT`
- Top IP bloquées et top ports destination
- Section ports entrants avec filtres horaires
- Liste IP sur plage horaire avec `deny`, `permit`, `deny_%` et ports ciblés
- Courbe UDP/TCP appliquée à la plage horaire sélectionnée (y compris plage traversant minuit, ex. 22h->6h)
- Heatmap activité heure/jour
- Cartographie géographique
- Table filtrée exportable en CSV

### Page ML et interprétation

- Prétraitement automatique des variables
- ACP (projection 2D)
- KMeans (K paramétrable)
- Score de silhouette
- Profiling des clusters
- Interprétation assistée par Mistral (optionnelle)

## Exports

- Export CSV des données filtrées (page Analyses)
- Export CSV échantillon clusterisé (page ML)
- Exports TXT des analyses IA (si clé Mistral fournie)

## Notes importantes

- L'application fonctionne sans clé Mistral.
- La clé Mistral n'est nécessaire que pour les blocs d'interprétation IA.








## Auteur

Projet réalisé dans le cadre du Challenge OPSIE SISE (Master 2).
