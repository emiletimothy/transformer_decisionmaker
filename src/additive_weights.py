"""
Additive Multiplicative Weights Algorithm Implementation

This module implements the additive variant of multiplicative weights with softmax normalization.
Update rule: w_i = w_i + η * reward_i (where reward = 1 - loss)
Probabilities: p_i = softmax(w_i) = exp(w_i) / Σ exp(w_j)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdditiveMultiplicativeWeights:
    """
    Additive Multiplicative Weights Algorithm with softmax normalization.
    
    Uses additive updates: w_i = w_i + η * 1{expert i was good}
    And softmax for probabilities: p_i = exp(w_i) / Σ exp(w_j)
    """
    
    def __init__(self, num_experts: int, learning_rate: float = 0.1, temperature: float = 1.0):
        """
        Initialize the additive multiplicative weights algorithm.
        
        Args:
            num_experts: Number of experts/actions to track
            learning_rate: Learning rate (eta) for weight updates
            temperature: Temperature parameter for softmax (higher = more exploration)
        """
        self.num_experts = num_experts
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        # Initialize weights to zero (uniform after softmax)
        self.weights = np.zeros(num_experts)
        self.cumulative_rewards = np.zeros(num_experts)
        self.round_count = 0
        
        # History tracking
        self.weight_history = [self.weights.copy()]
        self.reward_history = []
        self.probability_history = [self.get_probabilities()]
        self.regret_history = []
        
    def softmax(self, weights: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities with temperature scaling."""
        # Numerical stability: subtract max
        stable_weights = (weights - np.max(weights)) / self.temperature
        exp_weights = np.exp(stable_weights)
        return exp_weights / np.sum(exp_weights)
    
    def get_probabilities(self) -> np.ndarray:
        """Get current probability distribution over experts via softmax."""
        return self.softmax(self.weights)
    
    def select_expert(self, method: str = 'sample') -> int:
        """
        Select an expert based on current probabilities.
        
        Args:
            method: Selection method ('sample', 'greedy', or 'proportional')
            
        Returns:
            Index of selected expert
        """
        probs = self.get_probabilities()
        
        if method == 'sample':
            return np.random.choice(self.num_experts, p=probs)
        elif method == 'greedy':
            return np.argmax(probs)
        elif method == 'proportional':
            return probs  # Return full distribution
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def update_weights_from_losses(self, losses: np.ndarray):
        """
        Update weights based on observed losses.
        Converts losses to rewards: reward = 1 - loss
        
        Args:
            losses: Array of losses for each expert (lower is better)
        """
        if len(losses) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} losses, got {len(losses)}")
        
        # Convert losses to rewards (assuming losses in [0,1])
        rewards = 1.0 - np.clip(losses, 0, 1)
        self.update_weights_from_rewards(rewards)
    
    def update_weights_from_rewards(self, rewards: np.ndarray):
        """
        Update weights based on observed rewards using additive rule.
        
        Args:
            rewards: Array of rewards for each expert (higher is better)
        """
        if len(rewards) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} rewards, got {len(rewards)}")
        
        # Additive update: w_i = w_i + η * reward_i
        self.weights += self.learning_rate * rewards
        
        # Update tracking variables
        self.cumulative_rewards += rewards
        self.round_count += 1
        
        # Store history
        self.weight_history.append(self.weights.copy())
        self.reward_history.append(rewards.copy())
        self.probability_history.append(self.get_probabilities())
        
        # Calculate regret (in terms of rewards - higher is better)
        best_expert_reward = np.max(self.cumulative_rewards)
        probs_history = self.probability_history[:-1]  # Exclude current round
        if probs_history and self.reward_history:
            algorithm_reward = np.sum([np.dot(p, r) for p, r in zip(probs_history, self.reward_history)])
            regret = best_expert_reward - algorithm_reward  # Reward regret (lower is better)
            self.regret_history.append(regret)
        
        logger.info(f"Round {self.round_count}: Updated weights, current regret: {regret if self.regret_history else 0:.4f}")
    
    def update_weights_indicator(self, good_experts: List[int], reward_value: float = 1.0):
        """
        Update weights using indicator function: w_i = w_i + η * 1{i ∈ good_experts}
        
        Args:
            good_experts: List of expert indices that performed well
            reward_value: Reward value to give to good experts
        """
        # Create indicator rewards
        indicator_rewards = np.zeros(self.num_experts)
        for expert_idx in good_experts:
            if 0 <= expert_idx < self.num_experts:
                indicator_rewards[expert_idx] = reward_value
        
        self.update_weights_from_rewards(indicator_rewards)
    
    def get_regret_bound(self) -> float:
        """
        Calculate theoretical regret bound for additive MW algorithm.
        
        Returns:
            Theoretical upper bound on regret
        """
        if self.round_count == 0:
            return 0.0
        
        # For additive MW with rewards in [0,1]: O(√(T log n))
        return np.sqrt(2 * self.round_count * np.log(self.num_experts))
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics about the algorithm's performance."""
        if self.round_count == 0:
            return {"message": "No rounds completed yet"}
        
        best_expert = np.argmax(self.cumulative_rewards)
        current_regret = self.regret_history[-1] if self.regret_history else 0
        
        return {
            "rounds_completed": self.round_count,
            "current_weights": self.weights,
            "current_probabilities": self.get_probabilities(),
            "cumulative_rewards": self.cumulative_rewards,
            "best_expert": best_expert,
            "best_expert_reward": self.cumulative_rewards[best_expert],
            "current_regret": current_regret,
            "regret_bound": self.get_regret_bound(),
            "learning_rate": self.learning_rate,
            "temperature": self.temperature
        }
    
    def plot_performance(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot algorithm performance including weights and probabilities evolution.
        
        Args:
            figsize: Figure size for the plot
        """
        if self.round_count == 0:
            print("No data to plot yet")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Raw weight evolution
        rounds = range(len(self.weight_history))
        for i in range(self.num_experts):
            weights_over_time = [w[i] for w in self.weight_history]
            axes[0, 0].plot(rounds, weights_over_time, label=f'Expert {i}', marker='o', markersize=2)
        
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Raw Weight')
        axes[0, 0].set_title('Raw Weight Evolution (Before Softmax)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Probability evolution (after softmax)
        for i in range(self.num_experts):
            probs_over_time = [p[i] for p in self.probability_history]
            axes[0, 1].plot(rounds, probs_over_time, label=f'Expert {i}', marker='o', markersize=2)
        
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].set_title('Probability Evolution (After Softmax)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative rewards
        axes[0, 2].bar(range(self.num_experts), self.cumulative_rewards, 
                       color=['green' if i == np.argmax(self.cumulative_rewards) else 'blue' 
                              for i in range(self.num_experts)])
        axes[0, 2].set_xlabel('Expert')
        axes[0, 2].set_ylabel('Cumulative Reward')
        axes[0, 2].set_title('Cumulative Rewards by Expert')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Regret over time
        if self.regret_history:
            axes[1, 0].plot(range(1, len(self.regret_history) + 1), self.regret_history, 
                           'r-', label='Actual Regret', linewidth=2)
            regret_bounds = [self.get_regret_bound() for _ in range(len(self.regret_history))]
            axes[1, 0].plot(range(1, len(regret_bounds) + 1), regret_bounds, 
                           'b--', label='Theoretical Bound', linewidth=2)
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Regret')
            axes[1, 0].set_title('Regret vs Theoretical Bound')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Reward distribution per round
        if self.reward_history:
            reward_matrix = np.array(self.reward_history).T
            im = axes[1, 1].imshow(reward_matrix, aspect='auto', cmap='RdYlGn')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Expert')
            axes[1, 1].set_title('Reward Heatmap (Green = Higher Reward)')
            plt.colorbar(im, ax=axes[1, 1])
        
        # Plot 6: Softmax temperature effect
        x = np.linspace(-2, 2, 100)
        for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
            y = np.exp(x / temp) / np.sum(np.exp(x / temp))
            axes[1, 2].plot(x, y, label=f'T={temp}')
        axes[1, 2].set_xlabel('Weight Difference')
        axes[1, 2].set_ylabel('Probability')
        axes[1, 2].set_title(f'Softmax Effect (Current T={self.temperature})')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def compare_algorithms_demo():
    """Demonstrate difference between multiplicative and additive variants."""
    print("=== Comparing Multiplicative vs Additive Weights ===\n")
    
    # Import the original algorithm
    from multiplicative_weights import MultiplicativeWeights
    
    # Same setup for both algorithms
    num_experts = 4
    num_rounds = 100
    
    # Create both algorithms
    mw_mult = MultiplicativeWeights(num_experts=num_experts, learning_rate=0.1)
    mw_add = AdditiveMultiplicativeWeights(num_experts=num_experts, learning_rate=0.1)
    
    # Run same sequence of losses
    np.random.seed(42)  # For reproducible comparison
    
    for round_num in range(num_rounds):
        # Generate losses (expert 1 is consistently better)
        losses = np.random.random(num_experts)
        losses[1] *= 0.3  # Make expert 1 much better
        
        # Update both algorithms
        mw_mult.update_weights(losses)
        mw_add.update_weights_from_losses(losses)
        
        if round_num % 25 == 24:
            print(f"Round {round_num + 1}:")
            print(f"  Multiplicative weights: {mw_mult.get_probabilities()}")
            print(f"  Additive weights:       {mw_add.get_probabilities()}")
            print()
    
    print("Key Differences:")
    print("• Multiplicative: Exponential decay, faster convergence")
    print("• Additive: Linear updates, more exploration")
    print("• Softmax: Smooth probabilities, temperature controls exploration")


if __name__ == "__main__":
    compare_algorithms_demo()
