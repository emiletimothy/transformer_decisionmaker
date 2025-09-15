"""
Multiplicative Weights Algorithm Implementation

This module implements the multiplicative weights algorithm for online decision making.
The algorithm maintains weights over a set of experts/actions and updates them based
on observed losses, giving higher weight to better-performing experts over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Callable
import logging

logging.basicConfig(level=logging.INFO)
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
    
    def get_regret_bound(self) -> float:
        """
        Calculate theoretical regret bound for the algorithm.
        
        Returns:
            Theoretical upper bound on regret
        """
        if self.round_count == 0:
            return 0.0
        
        # Theoretical bound: (ln(n) + eta * sum(losses^2)) / eta
        max_loss_per_round = 1.0  # Assuming losses are in [0, 1]
        return (np.log(self.num_experts) + self.learning_rate * self.round_count * max_loss_per_round) / self.learning_rate
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics about the algorithm's performance."""
        if self.round_count == 0:
            return {"message": "No rounds completed yet"}
        
        best_expert = np.argmin(self.cumulative_losses)
        current_regret = self.regret_history[-1] if self.regret_history else 0
        
        return {
            "rounds_completed": self.round_count,
            "current_weights": self.weights,
            "cumulative_losses": self.cumulative_losses,
            "best_expert": best_expert,
            "best_expert_loss": self.cumulative_losses[best_expert],
            "current_regret": current_regret,
            "regret_bound": self.get_regret_bound(),
            "learning_rate": self.learning_rate
        }
    
    def plot_performance(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot algorithm performance including weights evolution and regret.
        
        Args:
            figsize: Figure size for the plot
        """
        if self.round_count == 0:
            print("No data to plot yet")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Weight evolution
        rounds = range(len(self.weight_history))
        for i in range(self.num_experts):
            weights_over_time = [w[i] for w in self.weight_history]
            axes[0, 0].plot(rounds, weights_over_time, label=f'Expert {i}', marker='o', markersize=3)
        
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Weight')
        axes[0, 0].set_title('Weight Evolution Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative losses
        axes[0, 1].bar(range(self.num_experts), self.cumulative_losses, 
                       color=['red' if i == np.argmin(self.cumulative_losses) else 'blue' 
                              for i in range(self.num_experts)])
        axes[0, 1].set_xlabel('Expert')
        axes[0, 1].set_ylabel('Cumulative Loss')
        axes[0, 1].set_title('Cumulative Losses by Expert')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Regret over time
        if self.regret_history:
            axes[1, 0].plot(range(1, len(self.regret_history) + 1), self.regret_history, 
                           'g-', label='Actual Regret', linewidth=2)
            regret_bounds = [self.get_regret_bound() for _ in range(len(self.regret_history))]
            axes[1, 0].plot(range(1, len(regret_bounds) + 1), regret_bounds, 
                           'r--', label='Theoretical Bound', linewidth=2)
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Regret')
            axes[1, 0].set_title('Regret vs Theoretical Bound')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss distribution per round
        if self.loss_history:
            loss_matrix = np.array(self.loss_history).T
            im = axes[1, 1].imshow(loss_matrix, aspect='auto', cmap='YlOrRd')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Expert')
            axes[1, 1].set_title('Loss Heatmap (Red = Higher Loss)')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()


def bandit_loss_function(expert_choice: int, true_best: int, noise_level: float = 0.1) -> float:
    """
    Generate loss for bandit-style problems.
    
    Args:
        expert_choice: The expert's choice/action
        true_best: The true best choice for this round
        noise_level: Amount of random noise to add
        
    Returns:
        Loss value (0 = perfect, higher = worse)
    """
    base_loss = abs(expert_choice - true_best) / max(expert_choice, true_best, 1)
    noise = np.random.normal(0, noise_level)
    return max(0, min(1, base_loss + noise))


def adversarial_loss_function(round_num: int, expert_idx: int, num_experts: int) -> float:
    """
    Generate adversarial losses designed to test the algorithm.
    
    Args:
        round_num: Current round number
        expert_idx: Expert index
        num_experts: Total number of experts
        
    Returns:
        Loss value for the expert in this round
    """
    # Create shifting best expert to test adaptation
    best_expert = (round_num // 10) % num_experts
    if expert_idx == best_expert:
        return 0.1 + 0.1 * np.random.random()  # Low loss for best expert
    else:
        return 0.5 + 0.4 * np.random.random()  # Higher loss for others
