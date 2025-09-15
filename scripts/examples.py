"""
Examples and demonstrations of the Multiplicative Weights Algorithm

This module contains various examples showing how to use the multiplicative weights
algorithm for different types of online decision making problems.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.multiplicative_weights import MultiplicativeWeights, bandit_loss_function, adversarial_loss_function
import matplotlib.pyplot as plt


def example_basic_usage():
    """Basic example showing how to use the multiplicative weights algorithm."""
    print("=== Basic Usage Example ===")
    
    # Create algorithm with 4 experts
    mw = MultiplicativeWeights(num_experts=4, learning_rate=0.1)
    
    # Simulate 50 rounds
    for round_num in range(50):
        # Select an expert
        selected_expert = mw.select_expert(method='sample')
        
        # Generate losses for all experts (in practice, you might only observe loss for selected expert)
        losses = np.random.random(4)  # Random losses between 0 and 1
        losses[1] *= 0.5  # Make expert 1 consistently better
        
        # Update weights based on losses
        mw.update_weights(losses)
        
        if round_num % 10 == 9:
            print(f"Round {round_num + 1}: Current weights: {mw.get_probabilities()}")
    
    # Print final statistics
    stats = mw.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"Best expert: {stats['best_expert']} (loss: {stats['best_expert_loss']:.3f})")
    print(f"Algorithm regret: {stats['current_regret']:.3f}")
    print(f"Regret bound: {stats['regret_bound']:.3f}")
    
    return mw


def example_online_portfolio():
    """Example of using multiplicative weights for online portfolio selection."""
    print("\n=== Online Portfolio Selection Example ===")
    
    # Simulate 5 stocks
    num_stocks = 5
    mw = MultiplicativeWeights(num_experts=num_stocks, learning_rate=0.05)
    
    # Simulate stock returns over 100 days
    portfolio_values = []
    stock_prices = np.ones(num_stocks) * 100  # Start all stocks at $100
    
    for day in range(100):
        # Get current portfolio weights
        weights = mw.get_probabilities()
        
        # Simulate daily stock returns (some stocks are better than others)
        returns = np.random.normal([0.001, 0.002, 0.0005, 0.0015, 0.0008], 0.02, num_stocks)
        
        # Update stock prices
        stock_prices *= (1 + returns)
        
        # Calculate portfolio value
        portfolio_value = np.dot(weights, stock_prices)
        portfolio_values.append(portfolio_value)
        
        # Convert returns to losses (negative returns are losses)
        losses = -returns  # Higher loss for negative returns
        losses = (losses - np.min(losses)) / (np.max(losses) - np.min(losses))  # Normalize to [0,1]
        
        # Update weights
        mw.update_weights(losses)
        
        if day % 20 == 19:
            print(f"Day {day + 1}: Portfolio value: ${portfolio_value:.2f}, Best stock: {np.argmax(stock_prices)}")
    
    # Plot portfolio performance
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(portfolio_values, 'b-', linewidth=2, label='MW Portfolio')
    plt.plot(stock_prices, alpha=0.7, linestyle='--', label=[f'Stock {i}' for i in range(num_stocks)])
    plt.xlabel('Day')
    plt.ylabel('Value ($)')
    plt.title('Portfolio vs Individual Stocks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    final_weights = mw.get_probabilities()
    plt.pie(final_weights, labels=[f'Stock {i}' for i in range(num_stocks)], autopct='%1.1f%%')
    plt.title('Final Portfolio Allocation')
    plt.tight_layout()
    plt.show()
    
    return mw


def example_expert_advice():
    """Example of aggregating advice from multiple experts."""
    print("\n=== Expert Advice Aggregation Example ===")
    
    # 6 weather prediction experts
    num_experts = 6
    mw = MultiplicativeWeights(num_experts=num_experts, learning_rate=0.2)
    
    # Expert characteristics (some are better at different times)
    expert_skills = [0.7, 0.8, 0.6, 0.75, 0.65, 0.9]  # Base accuracy
    
    correct_predictions = np.zeros(num_experts)
    total_predictions = 0
    
    # Simulate 200 days of weather prediction
    for day in range(200):
        # True weather (0=sunny, 1=rainy)
        true_weather = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Get expert predictions
        expert_predictions = []
        losses = np.zeros(num_experts)
        
        for i in range(num_experts):
            # Expert makes prediction based on their skill
            skill = expert_skills[i]
            if np.random.random() < skill:
                prediction = true_weather  # Correct prediction
                losses[i] = 0.0
                correct_predictions[i] += 1
            else:
                prediction = 1 - true_weather  # Wrong prediction
                losses[i] = 1.0
            expert_predictions.append(prediction)
        
        total_predictions += 1
        
        # Update weights
        mw.update_weights(losses)
        
        # Get weighted prediction
        weights = mw.get_probabilities()
        weighted_prediction = np.dot(weights, expert_predictions)
        algorithm_prediction = 1 if weighted_prediction > 0.5 else 0
        
        if day % 50 == 49:
            accuracies = correct_predictions / total_predictions
            print(f"Day {day + 1}:")
            print(f"  Expert accuracies: {accuracies}")
            print(f"  Current weights: {weights}")
            print(f"  Algorithm accuracy: {np.sum(weights * accuracies):.3f}")
    
    return mw


def example_adversarial_setting():
    """Example showing performance against adversarial losses."""
    print("\n=== Adversarial Setting Example ===")
    
    num_experts = 8
    mw = MultiplicativeWeights(num_experts=num_experts, learning_rate=0.1)
    
    # Run against adversarial losses
    for round_num in range(150):
        losses = np.array([adversarial_loss_function(round_num, i, num_experts) 
                          for i in range(num_experts)])
        mw.update_weights(losses)
    
    # Plot the results
    mw.plot_performance()
    
    stats = mw.get_statistics()
    print(f"Against adversarial losses:")
    print(f"Algorithm regret: {stats['current_regret']:.3f}")
    print(f"Regret bound: {stats['regret_bound']:.3f}")
    print(f"Regret ratio: {stats['current_regret'] / stats['regret_bound']:.3f}")
    
    return mw


def example_bandit_setting():
    """Example of multiplicative weights in a bandit setting."""
    print("\n=== Multi-Armed Bandit Example ===")
    
    num_arms = 5
    mw = MultiplicativeWeights(num_experts=num_arms, learning_rate=0.15)
    
    # Define true arm qualities (arm 2 is best)
    true_qualities = [0.3, 0.5, 0.8, 0.4, 0.6]
    
    total_reward = 0
    
    for round_num in range(300):
        # Select arm based on current weights
        selected_arm = mw.select_expert(method='sample')
        
        # Observe reward for selected arm (in practice, we only see this arm's reward)
        reward = np.random.random() < true_qualities[selected_arm]
        total_reward += reward
        
        # Convert reward to loss
        losses = np.ones(num_arms)  # We don't know losses for unselected arms
        losses[selected_arm] = 1 - reward  # Loss is 1 - reward
        
        # In bandit setting, we use loss estimates
        # Simple approach: assume other arms have average loss
        avg_loss = 0.5
        for i in range(num_arms):
            if i != selected_arm:
                losses[i] = avg_loss
        
        mw.update_weights(losses)
        
        if round_num % 75 == 74:
            weights = mw.get_probabilities()
            print(f"Round {round_num + 1}:")
            print(f"  Arm weights: {weights}")
            print(f"  Cumulative reward: {total_reward}")
            print(f"  Best arm probability: {weights[np.argmax(true_qualities)]:.3f}")
    
    print(f"\nFinal Results:")
    print(f"True best arm: {np.argmax(true_qualities)} (quality: {max(true_qualities)})")
    print(f"Algorithm's preferred arm: {np.argmax(mw.get_probabilities())}")
    print(f"Total reward rate: {total_reward / 300:.3f}")
    
    return mw


def run_all_examples():
    """Run all examples to demonstrate the algorithm."""
    print("🎯 Multiplicative Weights Algorithm Examples\n")
    
    # Run all examples
    mw1 = example_basic_usage()
    mw2 = example_online_portfolio()
    mw3 = example_expert_advice()
    mw4 = example_adversarial_setting()
    mw5 = example_bandit_setting()
    
    print("\n" + "="*60)
    print("✅ All examples completed successfully!")
    print("The multiplicative weights algorithm adapts to different scenarios:")
    print("• Basic online learning with multiple experts")
    print("• Online portfolio selection and investment")
    print("• Aggregating predictions from multiple sources")
    print("• Robust performance against adversarial losses")
    print("• Multi-armed bandit problems")
    print("="*60)
    
    return [mw1, mw2, mw3, mw4, mw5]


if __name__ == "__main__":
    run_all_examples()
