# 🏭 JobShop-Planner

Système de planification d'atelier de fabrication avec optimisation par **Reinforcement Learning**.

## 🚀 Installation Rapide

```bash
# Cloner le repository
git clone https://github.com/ZougaMouhcine/JobShop-Planner.git
cd JobShop-Planner

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement (Windows)
.venv\Scripts\activate

# Installer les dépendances
pip install -e ./JSSEnv
pip install streamlit plotly stable-baselines3 gymnasium

# Lancer l'application
cd JSSEnv
streamlit run app.py
```

## 📦 Fonctionnalités

- ✅ **Interface Streamlit** intuitive pour la saisie des commandes
- ✅ **3 algorithmes RL** (PPO, A2C, DQN) comparés automatiquement
- ✅ **Sélection automatique** du meilleur algorithme
- ✅ **Diagramme de Gantt** interactif avec Plotly
- ✅ **3 types de pièces** (A, B, C) avec séquences d'opérations prédéfinies
- ✅ **5 machines** : CNC, Fraiseuse, Tour, Perceuse, Polisseuse
- ✅ **Export HTML** des plannings générés

## 🤖 Algorithmes de Reinforcement Learning

| Algorithme | Description |
|------------|-------------|
| **PPO** | Proximal Policy Optimization - Stable et performant |
| **A2C** | Advantage Actor-Critic - Rapide à entraîner |
| **DQN** | Deep Q-Network - Classique et robuste |

Les 3 algorithmes sont entraînés en parallèle et le meilleur (makespan le plus court) est automatiquement sélectionné.

## 📊 Structure du Projet

```
JobShop-Planner/
├── JSSEnv/                      # Package principal
│   ├── app.py                   # Interface Streamlit
│   ├── rl_agent.py              # Agents RL (PPO, A2C, DQN)
│   ├── instance_generator.py   # Générateur d'instances
│   ├── JSSEnv/                  # Module Gymnasium
│   │   ├── __init__.py          # Registration environnement
│   │   ├── utils.py
│   │   └── envs/
│   │       ├── jss_env.py       # Environnement Job Shop (jss-v1)
│   │       └── instances/       # Instances de test
│   └── results/                 # Plannings générés
└── README.md
```

## 🔧 Types de Pièces

### Pièce A - Support métallique (20 min)
CNC → Fraiseuse → Tour → Perceuse → Polisseuse

### Pièce B - Axe cylindrique (21 min)
Tour → CNC → Fraiseuse → Perceuse → Polisseuse

### Pièce C - Plaque percée (22 min)
Fraiseuse → Perceuse → CNC → Tour → Polisseuse

## 💡 Utilisation

1. Accéder à l'interface Streamlit
2. Saisir le nombre de pièces A, B et C souhaité
3. Configurer les timesteps d'entraînement (défaut: 10 000)
4. Cliquer sur "🚀 Générer le Planning"
5. Consulter les résultats :
   - Comparaison des 3 algorithmes RL
   - Makespan de chaque algorithme
   - Meilleur algorithme sélectionné automatiquement
   - Diagramme de Gantt interactif
6. Télécharger le planning en HTML

## 📈 Performances

L'entraînement des agents RL prend environ **1-5 minutes** selon :
- Le nombre de pièces
- Le nombre de timesteps
- La puissance de calcul disponible

## 🛠️ Technologies

- **Python 3.11+**
- **Streamlit** - Interface web
- **Stable-Baselines3** - Algorithmes RL (PPO, A2C, DQN)
- **Gymnasium** - Environnement de simulation
- **Plotly** - Visualisation des plannings
- **NumPy** - Calculs numériques

## 📝 Licence

MIT License

## 👤 Auteur

Mouhcine Zouga
