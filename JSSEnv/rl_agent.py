"""
Agent de Reinforcement Learning pour l'optimisation du Job Shop Scheduling.
Utilise Stable-Baselines3 avec PPO, A2C ou DQN.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
import streamlit as st


class ActionMaskerWrapper(gym.Wrapper):
    """
    Wrapper qui force l'agent à choisir uniquement des actions légales.
    Convertit les actions illégales en actions légales aléatoires.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        # Récupérer le masque d'actions légales
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        legal_actions = base_env.legal_actions
        
        # Si l'action n'est pas légale, en choisir une légale
        if not legal_actions[action]:
            legal_indices = np.where(legal_actions)[0]
            if len(legal_indices) > 0:
                action = np.random.choice(legal_indices)
            else:
                # Si aucune action légale, utiliser l'action no-op (dernier index)
                action = len(legal_actions) - 1
        
        return self.env.step(action)


class TrainingProgressCallback(BaseCallback):
    """Callback pour afficher la progression de l'entraînement dans Streamlit."""
    
    def __init__(self, total_timesteps, progress_bar=None, status_text=None, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.best_reward = -np.inf
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # Mettre à jour la progression
        if self.progress_bar is not None:
            progress = min(self.num_timesteps / self.total_timesteps, 1.0)
            self.progress_bar.progress(progress)
            
        if self.status_text is not None:
            self.status_text.text(f"Entraînement RL: {self.num_timesteps}/{self.total_timesteps} timesteps")
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Récupérer les récompenses des épisodes
        if len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            if recent_rewards:
                avg_reward = np.mean(recent_rewards)
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward


class RLScheduler:
    """Agent RL pour l'ordonnancement Job Shop."""
    
    ALGORITHMS = {
        'PPO': PPO,
        'A2C': A2C,
        'DQN': DQN
    }
    
    def __init__(self, instance_path: str, algorithm: str = 'PPO'):
        """
        Initialise l'agent RL.
        
        Args:
            instance_path: Chemin vers le fichier d'instance
            algorithm: Algorithme à utiliser ('PPO', 'A2C', 'DQN')
        """
        self.instance_path = instance_path
        self.algorithm_name = algorithm
        self.env = None
        self.model = None
        
        # Créer l'environnement
        self._create_env()
        
    def _create_env(self):
        """Crée l'environnement Gymnasium avec le wrapper de masquage d'actions."""
        import JSSEnv  # Assure l'enregistrement
        base_env = gym.make('jss-v1', env_config={'instance_path': self.instance_path})
        # Appliquer le wrapper pour forcer les actions légales
        self.env = ActionMaskerWrapper(base_env)
        
    def _get_base_env(self):
        """Récupère l'environnement de base (sans wrappers)."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env
    
    def train(self, total_timesteps: int = 50000, progress_bar=None, status_text=None):
        """
        Entraîne l'agent RL.
        
        Args:
            total_timesteps: Nombre de timesteps d'entraînement
            progress_bar: Barre de progression Streamlit (optionnel)
            status_text: Texte de statut Streamlit (optionnel)
            
        Returns:
            dict: Statistiques d'entraînement
        """
        # Sélectionner l'algorithme
        AlgorithmClass = self.ALGORITHMS.get(self.algorithm_name, PPO)
        
        # Configuration selon l'algorithme
        if self.algorithm_name == 'DQN':
            # DQN nécessite un espace d'action discret (déjà le cas)
            self.model = AlgorithmClass(
                'MultiInputPolicy',
                self.env,
                learning_rate=1e-3,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                exploration_fraction=0.3,
                exploration_final_eps=0.05,
                verbose=0
            )
        elif self.algorithm_name == 'A2C':
            # A2C - pas de batch_size, utilise n_steps directement
            self.model = AlgorithmClass(
                'MultiInputPolicy',
                self.env,
                learning_rate=3e-4,
                n_steps=128,
                gamma=0.99,
                verbose=0
            )
        else:
            # PPO
            self.model = AlgorithmClass(
                'MultiInputPolicy',
                self.env,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                gamma=0.99,
                verbose=0
            )
        
        # Callback pour la progression
        callback = TrainingProgressCallback(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            status_text=status_text
        )
        
        # Entraînement
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )
        
        return {
            'algorithm': self.algorithm_name,
            'timesteps': total_timesteps,
            'best_reward': callback.best_reward
        }
    
    def generate_schedule(self):
        """
        Génère un planning en utilisant l'agent entraîné.
        
        Returns:
            dict: Résultats du planning (makespan, figure Gantt, etc.)
        """
        if self.model is None:
            raise ValueError("L'agent doit être entraîné avant de générer un planning.")
        
        # Reset de l'environnement
        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        base_env = self._get_base_env()
        
        # Exécution de l'agent
        while not done:
            # Utiliser le masque d'actions légales
            action_masks = obs.get('action_mask', None)
            
            # Prédiction de l'action
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Si l'action n'est pas légale, choisir une action légale aléatoire
            if action_masks is not None and not action_masks[action]:
                legal_actions = np.where(action_masks)[0]
                if len(legal_actions) > 0:
                    action = np.random.choice(legal_actions)
            
            # Exécuter l'action
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            
            if truncated:
                done = True
        
        # Générer le diagramme de Gantt
        fig = base_env.render()
        makespan = base_env.current_time_step
        
        # Extraire les données des opérations pour le tableau récapitulatif
        operations_data = self._extract_operations_data(base_env)
        
        return {
            'makespan': makespan,
            'total_reward': total_reward,
            'steps': steps,
            'figure': fig,
            'operations_data': operations_data
        }
    
    def _extract_operations_data(self, base_env):
        """Extrait les données des opérations pour le tableau récapitulatif."""
        operations = []
        
        machine_names = {
            0: "CNC",
            1: "Fraiseuse", 
            2: "Tour",
            3: "Perceuse",
            4: "Polisseuse"
        }
        
        # Détecter le type de pièce
        def detect_part_type(job_id):
            first_machine = base_env.instance_matrix[job_id][0][0]
            second_machine = base_env.instance_matrix[job_id][1][0]
            if first_machine == 0 and second_machine == 1:
                return 'A'
            elif first_machine == 2 and second_machine == 0:
                return 'B'
            elif first_machine == 1 and second_machine == 3:
                return 'C'
            else:
                return '?'
        
        for job in range(base_env.jobs):
            part_type = detect_part_type(job)
            op_num = 0
            while op_num < base_env.machines and base_env.solution[job][op_num] != -1:
                start = base_env.solution[job][op_num]
                machine_id = base_env.instance_matrix[job][op_num][0]
                duration = base_env.instance_matrix[job][op_num][1]
                finish = start + duration
                
                operations.append({
                    'Job': job,
                    'Pièce': part_type,
                    'Opération': op_num + 1,
                    'Machine': machine_names.get(machine_id, f"M{machine_id}"),
                    'Début (min)': start,
                    'Fin (min)': finish,
                    'Durée (min)': duration
                })
                op_num += 1
        
        return operations
    
    def save_model(self, path: str):
        """Sauvegarde le modèle entraîné."""
        if self.model is not None:
            self.model.save(path)
            
    def load_model(self, path: str):
        """Charge un modèle sauvegardé."""
        AlgorithmClass = self.ALGORITHMS.get(self.algorithm_name, PPO)
        self.model = AlgorithmClass.load(path, env=self.env)


def train_and_schedule(instance_path: str, algorithm: str = 'PPO', 
                       timesteps: int = 50000, progress_bar=None, 
                       status_text=None):
    """
    Fonction utilitaire pour entraîner et générer un planning.
    
    Args:
        instance_path: Chemin vers l'instance
        algorithm: Algorithme RL ('PPO', 'A2C', 'DQN')
        timesteps: Nombre de timesteps d'entraînement
        progress_bar: Barre de progression Streamlit
        status_text: Texte de statut Streamlit
        
    Returns:
        tuple: (scheduler, training_stats, schedule_results)
    """
    scheduler = RLScheduler(instance_path, algorithm)
    
    # Entraînement
    training_stats = scheduler.train(
        total_timesteps=timesteps,
        progress_bar=progress_bar,
        status_text=status_text
    )
    
    # Génération du planning
    schedule_results = scheduler.generate_schedule()
    
    return scheduler, training_stats, schedule_results


def compare_rl_algorithms(instance_path: str, timesteps: int = 25000,
                          progress_callback=None, status_callback=None):
    """
    Entraîne et compare les 3 algorithmes RL (PPO, A2C, DQN).
    
    Args:
        instance_path: Chemin vers l'instance
        timesteps: Nombre de timesteps d'entraînement par algorithme
        progress_callback: Fonction callback pour la progression globale
        status_callback: Fonction callback pour le statut
        
    Returns:
        dict: Résultats de comparaison avec le meilleur algorithme
    """
    algorithms = ['PPO', 'A2C', 'DQN']
    results = {}
    
    for i, algo in enumerate(algorithms):
        if status_callback:
            status_callback(f"🤖 Entraînement {algo} ({i+1}/3)...")
        
        try:
            # Créer et entraîner l'agent
            scheduler = RLScheduler(instance_path, algo)
            
            training_stats = scheduler.train(
                total_timesteps=timesteps,
                progress_bar=None,
                status_text=None
            )
            
            # Générer le planning
            schedule_results = scheduler.generate_schedule()
            
            results[algo] = {
                'scheduler': scheduler,
                'training_stats': training_stats,
                'schedule_results': schedule_results,
                'makespan': schedule_results['makespan'],
                'total_reward': schedule_results['total_reward'],
                'figure': schedule_results['figure'],
                'operations_data': schedule_results.get('operations_data', []),
                'success': True
            }
            
        except Exception as e:
            results[algo] = {
                'success': False,
                'error': str(e),
                'makespan': float('inf')
            }
        
        # Mettre à jour la progression globale
        if progress_callback:
            progress_callback((i + 1) / len(algorithms))
    
    # Trouver le meilleur algorithme (makespan le plus bas)
    best_algo = min(
        [algo for algo in algorithms if results[algo]['success']],
        key=lambda x: results[x]['makespan'],
        default=None
    )
    
    return {
        'all_results': results,
        'best_algorithm': best_algo,
        'best_makespan': results[best_algo]['makespan'] if best_algo else None,
        'best_figure': results[best_algo]['figure'] if best_algo else None,
        'best_scheduler': results[best_algo]['scheduler'] if best_algo else None,
        'best_operations_data': results[best_algo]['operations_data'] if best_algo else []
    }
