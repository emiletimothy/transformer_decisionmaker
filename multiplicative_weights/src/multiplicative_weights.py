"""
Multiplicative Weights Algorithm for online learning and decision making.

The algorithm maintains a probability distribution over experts/actions and
updates it based on observed losses using multiplicative updates.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiplicativeWeights:
    """
    Multiplicative Weights Algorithm for online learning and decision making.
    
    The algorithm maintains a probability distribution over experts/actions and
    updates it based on observed losses using multiplicative updates.
    """
    
    def __init__(self, num_experts: int, learning_rate: float = 0.1):
        """
        Initialize the multiplicative weights algorithm.
        
        Args:
            num_experts: Number of experts/actions to track
            learning_rate: Learning rate (eta) for weight updates, typically in (0, 1)
        """
        self.num_experts = num_experts
        self.learning_rate = learning_rate
        
        # Initialize uniform weights
        self.weights = np.ones(num_experts) / num_experts
        self.cumulative_losses = np.zeros(num_experts)
        self.round_count = 0
        
        # History tracking
        self.weight_history = [self.weights.copy()]
        self.loss_history = []
        self.regret_history = []
        
    def get_probabilities(self) -> np.ndarray:
        """Get current probability distribution over experts."""
        return self.weights.copy()
    
    def select_expert(self, method: str = 'sample') -> int:
        """
        Select an expert based on current weights.
        
        Args:
            method: Selection method ('sample', 'greedy', or 'proportional')
            
        Returns:
            Index of selected expert
        """
        if method == 'sample':
            return np.random.choice(self.num_experts, p=self.weights)
        elif method == 'greedy':
            return np.argmax(self.weights)
        elif method == 'proportional':
            # For continuous decisions, return the full distribution
            return self.weights
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def update_weights(self, losses: np.ndarray):
        """
        Update weights based on observed losses.
        
        Args:
            losses: Array of losses for each expert (lower is better)
        """
        if len(losses) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} losses, got {len(losses)}")
        
        # Multiplicative update: w_i = w_i * exp(-eta * loss_i)
        self.weights *= np.exp(-self.learning_rate * losses)
        
        # Normalize to maintain probability distribution
        self.weights /= np.sum(self.weights)
        
        # Update tracking variables
        self.cumulative_losses += losses
        self.round_count += 1
        
        # Store history
        self.weight_history.append(self.weights.copy())
        self.loss_history.append(losses.copy())
        
        # Calculate regret
        best_expert_loss = np.min(self.cumulative_losses)
        algorithm_loss = np.sum([np.dot(w, l) for w, l in zip(self.weight_history[:-1], self.loss_history)])
        regret = algorithm_loss - best_expert_loss
        self.regret_history.append(regret)
        
        logger.info(f"Round {self.round_count}: Updated weights, current regret: {regret:.4f}")
    
    def get_regret(self) -> float:
        """Get current cumulative regret."""
        if not self.regret_history:
            return 0.0
        return self.regret_history[-1]
    
    def get_average_regret(self) -> float:
        """Get average regret per round."""
        if self.round_count == 0:
            return 0.0
        return self.get_regret() / self.round_count
    
    def reset(self):
        """Reset the algorithm to initial state."""
        self.weights = np.ones(self.num_experts) / self.num_experts
        self.cumulative_losses = np.zeros(self.num_experts)
        self.round_count = 0
        self.weight_history = [self.weights.copy()]
        self.loss_history = []
        self.regret_history = []
    
    def plot_weights(self, title: str = "Weight Evolution", save_path: str = None):
        """Plot weight evolution over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        weights_array = np.array(self.weight_history)
        for i in range(self.num_experts):
            ax.plot(weights_array[:, i], label=f'Expert {i}', marker='o', markersize=3)
        ax.set_xlabel('Round')
        ax.set_ylabel('Weight')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_regret(self, title: str = "Cumulative Regret", save_path: str = None):
        """Plot regret over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.regret_history, 'b-', linewidth=2)
        ax.set_xlabel('Round')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig


def generate_bandit_losses(n_experts: int, n_rounds: int,
                           best_expert: int = 0,
                           noise: float = 0.1) -> List[np.ndarray]:
    """Generate synthetic bandit losses where one expert is consistently best."""
    losses = []
    for _ in range(n_rounds):
        round_losses = np.random.uniform(0.3, 0.7, n_experts)
        round_losses[best_expert] = np.random.uniform(0.0, 0.2 + noise)
        losses.append(round_losses)
    return losses


def generate_adversarial_losses(n_experts: int, n_rounds: int,
                                 shift_period: int = 10) -> List[np.ndarray]:
    """Generate adversarial losses where the best expert shifts over time."""
    losses = []
    for t in range(n_rounds):
        best = (t // shift_period) % n_experts
        round_losses = np.random.uniform(0.4, 0.8, n_experts)
        round_losses[best] = np.random.uniform(0.0, 0.2)
        losses.append(round_losses)
    return losses
