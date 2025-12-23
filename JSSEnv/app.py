"""
Application Streamlit pour la planification d'atelier de fabrication.
Utilise le Reinforcement Learning (PPO, A2C, DQN) pour optimiser le planning.
Compare automatiquement les 3 algorithmes et sélectionne le meilleur.
"""

import streamlit as st
import time
from pathlib import Path
import gymnasium as gym
import JSSEnv
from instance_generator import WorkshopInstanceGenerator
from rl_agent import RLScheduler, compare_rl_algorithms

# Configuration de la page
st.set_page_config(
    page_title="Planification d'Atelier - RL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalise
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .rl-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .best-algo {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    /* Style pour le bouton Réinitialiser */
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #ff6b6b;
        color: white;
        border: none;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #ee5a5a;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Application principale"""
    
    # Initialiser session_state pour conserver les résultats
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'instance_info' not in st.session_state:
        st.session_state.instance_info = None
    if 'output_path' not in st.session_state:
        st.session_state.output_path = None
    if 'instance_path' not in st.session_state:
        st.session_state.instance_path = None
    
    # Header
    st.markdown('<div class="main-header"> Planification d\'Atelier avec RL</div>', 
                unsafe_allow_html=True)
    
    # Info RL
    st.markdown("""
    <div class="rl-info">
        <h4> Reinforcement Learning - Comparaison Automatique</h4>
        <p>Cette application entraîne <b>3 agents RL</b> (PPO, A2C, DQN), compare leurs performances 
        et sélectionne automatiquement le <b>meilleur algorithme</b> pour minimiser le makespan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar pour la configuration RL
    with st.sidebar:
        st.header(" Configuration RL")
        
        st.subheader(" Temps d'Entraînement")
        timesteps = st.select_slider(
            "Timesteps par algorithme:",
            options=[5000, 10000, 25000, 50000, 100000],
            value=25000,
            help="Chaque algorithme sera entraîné avec ce nombre de timesteps"
        )
        
        # Estimation du temps (3 algorithmes) - environ 3 minutes pour 25000 timesteps
        estimated_minutes = (timesteps / 25000) * 3
        st.info(f" Temps estimé total: ~{estimated_minutes:.0f} minutes")
        st.caption("(3 algorithmes seront entraînés)")
        
        st.divider()
        
        st.subheader(" Algorithmes Comparés")
        st.markdown("""
        - **PPO** - Proximal Policy Optimization
        - **A2C** - Advantage Actor-Critic  
        - **DQN** - Deep Q-Network
        """)
        
        st.divider()
        
        st.subheader(" Machines Disponibles")
        machines = [
            "CNC (Commande numerique)",
            "Fraiseuse", 
            "Tour",
            "Perceuse",
            "Polisseuse"
        ]
        for machine in machines:
            st.text(f"• {machine}")
        
        st.divider()
        
        st.subheader(" Pieces")
        pieces_info = {
            "A": ("Support métallique", "CNC → Fraiseuse → Tour → Perceuse → Polisseuse"),
            "B": ("Axe cylindrique", "Tour → CNC → Fraiseuse → Perceuse → Polisseuse"),
            "C": ("Plaque percée", "Fraiseuse → Perceuse → CNC → Tour → Polisseuse")
        }
        
        for piece, (nom, sequence) in pieces_info.items():
            with st.expander(f"Piece {piece} - {nom}"):
                st.write(f"**Séquence:** {sequence}")
    
    # Tabs pour les différentes sections
    tab1, tab2, tab3 = st.tabs(["📦 Commande & Entraînement", "📊 Historique", "ℹ️ A propos"])
    
    with tab1:
        st.header(" Saisie de la Commande")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_a = st.number_input(
                "Nombre de pieces A (Support metallique)",
                min_value=0,
                max_value=100,
                value=5,
                step=1,
                help="Support metallique - Séquence: CNC → Fraiseuse → Tour → Perceuse → Polisseuse"
            )
        
        with col2:
            num_b = st.number_input(
                "Nombre de pieces B (Axe cylindrique)", 
                min_value=0,
                max_value=100,
                value=3,
                step=1,
                help="Axe cylindrique - Séquence: Tour → CNC → Fraiseuse → Perceuse → Polisseuse"
            )
        
        with col3:
            num_c = st.number_input(
                "Nombre de pieces C (Plaque percee)",
                min_value=0,
                max_value=100,
                value=4,
                step=1,
                help="Plaque percée - Séquence: Fraiseuse → Perceuse → CNC → Tour → Polisseuse"
            )
        
        total_jobs = num_a + num_b + num_c
        
        # Affichage du resume
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Jobs", total_jobs)
        with col2:
            st.metric("Pieces A", num_a)
        with col3:
            st.metric("Pieces B", num_b)
        with col4:
            st.metric("Pieces C", num_c)
        
        # Configuration affichée
        st.divider()
        st.info(f" **Timesteps par algorithme:** {timesteps:,} | **Total:** {timesteps * 3:,} (3 algorithmes)")
        
        # Validation et generation
        if total_jobs == 0:
            st.warning("⚠️ Veuillez commander au moins une piece!")
        else:
            st.divider()
            
            if st.button(" Entraîner les 3 Agents RL et Comparer", type="primary", use_container_width=True):
                # Lancer l'entraînement et stocker les résultats dans session_state
                results = generate_planning_rl(num_a, num_b, num_c, timesteps)
                if results:
                    st.session_state.results = results
                    st.session_state.instance_info = results['instance_info']
                    st.session_state.output_path = results['output_path']
                    st.session_state.instance_path = results['instance_path']
                    st.rerun()
            
            # Afficher les résultats s'ils existent dans session_state
            if st.session_state.results:
                display_results(st.session_state.results)
    
    with tab2:
        st.header(" Historique des Plannings")
        
        # Lister les plannings generes
        results_dir = Path("results")
        if results_dir.exists():
            gantt_files = sorted(results_dir.glob("gantt_rl_*.html"), 
                               key=lambda x: x.stat().st_mtime, reverse=True)
            
            if gantt_files:
                st.write(f"**{len(gantt_files)} planning(s) RL genere(s)**")
                
                for i, gantt_file in enumerate(gantt_files[:10]):
                    filename = gantt_file.stem
                    parts = filename.replace("gantt_rl_", "").replace(".html", "")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{i+1}.** {parts}")
                    with col2:
                        if st.button("Ouvrir", key=f"open_{i}"):
                            with open(gantt_file, 'r', encoding='utf-8') as f:
                                st.components.v1.html(f.read(), height=600, scrolling=True)
            else:
                st.info("Aucun planning RL généré pour le moment.")
        else:
            st.info("Aucun planning généré pour le moment.")
    
    with tab3:
        st.header("ℹ️ A propos du Reinforcement Learning")
        
        st.subheader(" Qu'est-ce que le Reinforcement Learning?")
        st.write("""
        Le **Reinforcement Learning** (apprentissage par renforcement) est une branche du Machine Learning 
        où un agent apprend à prendre des décisions en interagissant avec un environnement.
        
        Dans notre cas:
        - **L'agent** = Le planificateur
        - **L'environnement** = L'atelier avec ses machines et jobs
        - **Les actions** = Choisir quel job exécuter sur quelle machine
        - **La récompense** = Négative du makespan (on veut minimiser le temps total)
        """)
        
        st.subheader(" Algorithmes Disponibles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### PPO
            **Proximal Policy Optimization**
            
            ✅ Stable et fiable  
            ✅ Bon équilibre exploration/exploitation  
            ✅ Recommandé pour débuter  
            """)
        
        with col2:
            st.markdown("""
            ### A2C
            **Advantage Actor-Critic**
            
            ✅ Entraînement rapide  
            ✅ Bonne parallélisation  
            ⚠️ Peut être instable  
            """)
        
        with col3:
            st.markdown("""
            ### DQN
            **Deep Q-Network**
            
            ✅ Bon pour actions discrètes  
            ✅ Experience replay  
            ⚠️ Plus lent à converger  
            """)
        
        st.subheader(" Comment ça fonctionne?")
        st.write("""
        1. **Génération de l'instance**: Création du fichier décrivant les jobs et leurs opérations
        2. **Création de l'environnement**: L'atelier est simulé comme un environnement Gymnasium
        3. **Entraînement de l'agent**: L'agent RL essaie différentes stratégies et apprend
        4. **Génération du planning**: L'agent entraîné génère le planning optimal
        5. **Visualisation**: Affichage du diagramme de Gantt
        """)
        
        st.subheader("⚡ Conseils")
        st.write("""
        - **Peu de jobs (< 20)**: 10,000-25,000 timesteps suffisent
        - **Jobs moyens (20-50)**: 25,000-50,000 timesteps recommandés
        - **Beaucoup de jobs (> 50)**: 50,000-100,000 timesteps pour de bons résultats
        - L'algorithme **PPO** est généralement le plus fiable
        """)


def generate_planning_rl(num_a, num_b, num_c, timesteps):
    """Génère le planning en comparant les 3 algorithmes RL. Retourne les résultats."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Etape 1: Generation de l'instance
        status_text.text(" Étape 1/4: Génération de l'instance...")
        progress_bar.progress(5)
        
        generator = WorkshopInstanceGenerator()
        instance_path, instance_info = generator.generate_instance(num_a, num_b, num_c)
        
        time.sleep(0.3)
        
        # Etape 2: Entrainement des 3 algorithmes
        status_text.text(" Étape 2/4: Entraînement des 3 agents RL (PPO, A2C, DQN)...")
        progress_bar.progress(10)
        
        # Conteneur pour afficher la progression de chaque algo
        algo_status = st.empty()
        algo_progress = st.progress(0)
        
        def update_progress(p):
            algo_progress.progress(p)
            progress_bar.progress(10 + int(p * 70))  # 10% à 80%
            
        def update_status(s):
            algo_status.text(s)
        
        # Comparer les 3 algorithmes
        comparison_results = compare_rl_algorithms(
            instance_path=instance_path,
            timesteps=timesteps,
            progress_callback=update_progress,
            status_callback=update_status
        )
        
        algo_status.empty()
        algo_progress.empty()
        
        progress_bar.progress(85)
        
        # Etape 3: Sélection du meilleur
        status_text.text(" Étape 3/4: Sélection du meilleur algorithme...")
        
        best_algo = comparison_results['best_algorithm']
        best_makespan = comparison_results['best_makespan']
        best_fig = comparison_results['best_figure']
        all_results = comparison_results['all_results']
        operations_data = comparison_results.get('best_operations_data', [])
        
        # Mettre à jour le graphique du meilleur
        best_fig.update_layout(
            title=f"Planning RL ({best_algo}) - Makespan: {best_makespan} min",
            height=max(500, (num_a + num_b + num_c) * 25)
        )
        
        # Etape 4: Sauvegarde
        status_text.text(" Étape 4/4: Sauvegarde du planning...")
        progress_bar.progress(95)
        
        output_path = Path("results") / f"gantt_rl_{best_algo}_{num_a}A_{num_b}B_{num_c}C.html"
        output_path.parent.mkdir(exist_ok=True)
        best_fig.write_html(str(output_path))
        
        progress_bar.progress(100)
        status_text.text(" Comparaison terminée!")
        
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Retourner les résultats
        return {
            'best_algo': best_algo,
            'best_makespan': best_makespan,
            'best_fig': best_fig,
            'all_results': all_results,
            'operations_data': operations_data,
            'instance_info': instance_info,
            'output_path': output_path,
            'instance_path': instance_path,
            'num_a': num_a,
            'num_b': num_b,
            'num_c': num_c
        }
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'entraînement RL: {e}")
        import traceback
        st.code(traceback.format_exc())
        progress_bar.empty()
        status_text.empty()
        return None


def display_results(results):
    """Affiche les résultats de l'entraînement RL."""
    
    best_algo = results['best_algo']
    best_makespan = results['best_makespan']
    best_fig = results['best_fig']
    all_results = results['all_results']
    operations_data = results['operations_data']
    instance_info = results['instance_info']
    output_path = results['output_path']
    instance_path = results['instance_path']
    
    # Header avec bouton réinitialiser
    col_header1, col_header2 = st.columns([5, 1])
    with col_header1:
        st.success(f" Meilleur algorithme: **{best_algo}** avec un makespan de **{best_makespan} min**")
    with col_header2:
        if st.button(" Réinitialiser", type="secondary", use_container_width=True, key="reset_btn"):
            st.session_state.results = None
            st.session_state.instance_info = None
            st.session_state.output_path = None
            st.session_state.instance_path = None
            st.rerun()
    
    # Tableau comparatif des algorithmes
    st.subheader(" Comparaison des 3 Algorithmes RL")
    
    comparison_data = []
    for algo in ['PPO', 'A2C', 'DQN']:
        if all_results[algo]['success']:
            makespan = all_results[algo]['makespan']
            reward = all_results[algo]['total_reward']
            eff = (instance_info['estimated_min_makespan'] / makespan) * 100 if makespan > 0 else 0
            is_best = "🏆" if algo == best_algo else ""
            comparison_data.append({
                'Algorithme': f"{algo} {is_best}",
                'Makespan (min)': makespan,
                'Efficacité (%)': f"{eff:.1f}",
                'Reward': f"{reward:.2f}"
            })
        else:
            comparison_data.append({
                'Algorithme': algo,
                'Makespan (min)': "Échec",
                'Efficacité (%)': "-",
                'Reward': "-"
            })
    
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    
    # Metriques principales
    st.subheader(" Résultats du Meilleur Algorithme")
    
    efficiency = (instance_info['estimated_min_makespan'] / best_makespan) * 100 if best_makespan > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Makespan Obtenu",
            f"{best_makespan} min",
            help="Temps total pour compléter tous les jobs"
        )
    
    with col2:
        st.metric(
            "Makespan Théorique",
            f"{instance_info['estimated_min_makespan']} min",
            help="Temps minimum avec parallélisation parfaite"
        )
    
    with col3:
        st.metric(
            "Efficacité",
            f"{efficiency:.1f}%",
            help="Proximité par rapport au minimum théorique"
        )
    
    with col4:
        st.metric(
            "Meilleur Algorithme",
            best_algo,
            help=f"Sélectionné parmi PPO, A2C, DQN"
        )
    
    # Diagramme de Gantt
    st.subheader(" Diagramme de Gantt (Meilleur Résultat)")
    st.plotly_chart(best_fig, use_container_width=True)
    
    # Tableau récapitulatif des opérations
    st.subheader(" Tableau Récapitulatif des Opérations")
    
    if operations_data:
        import pandas as pd
        df_operations = pd.DataFrame(operations_data)
        
        # Options de tri
        col_sort1, col_sort2 = st.columns(2)
        with col_sort1:
            sort_by = st.selectbox(
                "Trier par:",
                options=['Début (min)', 'Job', 'Machine', 'Pièce'],
                index=0,
                key='sort_by_select'
            )
        with col_sort2:
            sort_order = st.radio(
                "Ordre:",
                options=['Croissant', 'Décroissant'],
                horizontal=True,
                key='sort_order_radio'
            )
        
        ascending = sort_order == 'Croissant'
        df_sorted = df_operations.sort_values(by=sort_by, ascending=ascending)
        
        # Filtres
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            filter_piece = st.multiselect(
                "Filtrer par pièce:",
                options=['A', 'B', 'C'],
                default=['A', 'B', 'C'],
                key='filter_piece_select'
            )
        with col_filter2:
            filter_machine = st.multiselect(
                "Filtrer par machine:",
                options=['CNC', 'Fraiseuse', 'Tour', 'Perceuse', 'Polisseuse'],
                default=['CNC', 'Fraiseuse', 'Tour', 'Perceuse', 'Polisseuse'],
                key='filter_machine_select'
            )
        
        # Appliquer les filtres
        df_filtered = df_sorted[
            (df_sorted['Pièce'].isin(filter_piece)) & 
            (df_sorted['Machine'].isin(filter_machine))
        ]
        
        # Afficher le tableau
        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Statistiques par machine
        st.subheader(" Statistiques par Machine")
        machine_stats = df_operations.groupby('Machine').agg({
            'Durée (min)': ['sum', 'mean', 'count']
        }).round(1)
        machine_stats.columns = ['Temps Total (min)', 'Durée Moyenne (min)', 'Nb Opérations']
        machine_stats['Taux Utilisation (%)'] = (machine_stats['Temps Total (min)'] / best_makespan * 100).round(1)
        st.dataframe(machine_stats, use_container_width=True)
    else:
        st.info("Aucune donnée d'opération disponible.")
    
    # Telechargement
    st.subheader(" Téléchargement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with open(output_path, 'rb') as f:
            st.download_button(
                label="📥 Télécharger le Gantt (HTML)",
                data=f,
                file_name=Path(output_path).name,
                mime="text/html",
                key='download_gantt'
            )
    
    with col2:
        with open(instance_path, 'rb') as f:
            st.download_button(
                label="📥 Télécharger l'Instance",
                data=f,
                file_name=Path(instance_path).name,
                mime="text/plain",
                key='download_instance'
            )


if __name__ == "__main__":
    main()
