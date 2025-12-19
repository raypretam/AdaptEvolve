import random
import numpy as np
from typing import List, Optional
from queue import deque

class RandomSampling:
    """Uniform random sampling over arms."""
    def __init__(self, n_models: int, weights: List[float] = None, explore: int = 0):
        self.n_models = n_models
        total = sum(weights) if weights else n_models
        self.weights = [(w / total) for w in (weights or [1]*n_models)]
        self.random_state = random.Random()

    def sample(self) -> int:
        return self.random_state.choices(range(self.n_models), weights=self.weights, k=1)[0]

    def update(self, model_idx: int, reward: float):
        # no‐op for pure random
        pass


class ThompsonSampling:
    """Bernoulli‐Thompson (Beta prior) bandit sampler."""
    def __init__(self, n_models: int, explore: int = 10):
        self.n_models  = n_models
        self.alpha     = np.ones(n_models)   # success counts
        self.beta      = np.ones(n_models)   # failure counts
        self.iteration = 0
        self.explore   = explore
        self.random    = random.Random()

    def sample(self) -> int:
        # pure explore for the first N iters
        if self.iteration < self.explore:
            return self.random.randrange(self.n_models)
        thetas = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(thetas))

    def update(self, model_idx: int, reward: float):
        """
        reward ∈ [0,1]; we binarize success=r>0.
        """
        self.iteration += 1
        success = 1 if reward > 0.0 else 0
        if success:
            self.alpha[model_idx] += 1
        else:
            self.beta[model_idx]  += 1




class GaussianThompsonSampling:
    """
    Gaussian‐Thompson sampler with known noise variance.
    Posterior: N(mu, 1/λ)
    """
    def __init__(
        self,
        n_models:  int,
        prior_mean: float = 0.5,
        prior_var:  float = 0.1,
        noise_var:  float = 0.05,
        explore:    int   = 10
    ):
        self.n_models  = n_models
        self.mu        = np.full(n_models, prior_mean, dtype=float)
        self.lmbda     = np.full(n_models, 1.0/prior_var, dtype=float)
        self.noise_var = noise_var
        self.iteration = 0
        self.explore   = explore
        self.random    = random.Random()

    def sample(self) -> int:
        if self.iteration < self.explore:
            return self.random.randrange(self.n_models)
        sigma = np.sqrt(1.0 / self.lmbda)
        draws = np.random.normal(self.mu, sigma)
        return int(np.argmax(draws))

    def update(self, model_idx: int, reward: float):
        self.iteration += 1
        r = float(reward)
        # Bayesian update for known‐σ Gaussian:
        self.lmbda[model_idx] += 1.0 / self.noise_var
        self.mu[model_idx] = (
            (self.lmbda[model_idx] - 1.0/self.noise_var) * self.mu[model_idx]
            + r/self.noise_var
        ) / self.lmbda[model_idx]


class UCB1:
    """Upper Confidence Bound (UCB1) bandit sampler."""
    def __init__(self, n_models: int, explore: float = 2.0):
        self.n_models = n_models
        self.explore = explore  # Exploration parameter
        self.iteration = 0
        self.counts = np.zeros(n_models, dtype=int)
        self.values = np.zeros(n_models, dtype=float)
        print(f'Number of models : {self.n_models}')

    def sample(self) -> int:
        """Selects the arm with the highest UCB score."""
        # First, ensure each arm is tried at least once
        untried_arms = np.where(self.counts == 0)[0]
        if len(untried_arms) > 0:
            return untried_arms[0]

        # Calculate the UCB score for all arms
        # Score = Average Reward + Exploration Bonus
        exploration_bonus = np.sqrt(np.log(self.iteration) / self.counts)
        ucb_scores = self.values + self.explore * exploration_bonus
        
        return int(np.argmax(ucb_scores))

    def update(self, model_idx: int, reward: float):
        """Updates counts and average values for the chosen arm."""
        self.iteration += 1
        self.counts[model_idx] += 1
        n = self.counts[model_idx]
        
        # Update the average reward for the arm incrementally
        # This is equivalent to Q_n+1 = Q_n + (R_n+1 - Q_n) / (n+1)
        self.values[model_idx] += (reward - self.values[model_idx]) / n


class GradientBeliefNetwork:
    """
    Belief network that maintains explicit beliefs about:
    1. Current performance level
    2. Performance gradient (rate of change)
    3. Gradient stability (how consistent the trend is)
    
    Uses global best-so-far approach to reduce noise in gradient calculations.
    Tracks belief scores per iteration.
    Does not update beliefs during exploration phase.
    Uses safe gradient updates to prevent NaN values.
    """
    
    def __init__(
        self,
        n_models: int,
        window: int = 4,
        explore: int = 5,
        gradient_threshold: float = 0.01,  
        stability_weight: float = 0.1,     
        gradient_weight: float = 0.5,      
        performance_weight: float = 0.4    
    ):
        self.n_models = n_models
        self.explore = explore
        self.window = window
        self.gradient_threshold = gradient_threshold
        self.iteration = 0
        self.random = random.Random()
        
        # Weights for belief combination
        self.stability_weight = stability_weight
        self.gradient_weight = gradient_weight
        self.performance_weight = performance_weight
        
        # Belief states for each model
        self.performance_belief = np.full(n_models, 0.5)  # Current performance
        self.gradient_belief = np.zeros(n_models)         # Current gradient
        self.stability_belief = np.zeros(n_models)        # Gradient stability
        
        # History tracking
        self.reward_history = [deque(maxlen=window) for _ in range(n_models)]
        self.gradient_history = [deque(maxlen=window) for _ in range(n_models)]
        
        # Track belief scores per iteration
        self.belief_scores_history = []
        
        # Track global best performance so far
        self.global_best_so_far = -np.inf
        self.global_best_iteration = 0
        self.global_best_model = 0
        

        self.belief_confidence = np.ones(n_models)
        
        initial_scores = np.full(n_models, 0.5)
        self.belief_scores_history.append(initial_scores.copy())
        
    def _update_performance_belief(self, model_idx: int, reward: float):
        """Update belief about current performance level."""
        # Check for valid reward
        if not np.isfinite(reward):
            return
            
        # Update global best so far
        if reward > self.global_best_so_far:
            self.global_best_so_far = reward
            self.global_best_iteration = self.iteration
            self.global_best_model = model_idx
        
        # Use global best so far for performance belief
        alpha = 0.3
        if np.isfinite(self.global_best_so_far):
            new_belief = (1 - alpha) * self.performance_belief[model_idx] + alpha * self.global_best_so_far
            if np.isfinite(new_belief):
                self.performance_belief[model_idx] = new_belief
    
    def _update_gradient_belief(self, model_idx: int):
        """Update belief about performance gradient using global best-so-far approach."""
        if len(self.reward_history[model_idx]) < 2:
            return
            
        rewards = np.array(self.reward_history[model_idx])
        
        # Check if rewards contain valid values
        if not np.all(np.isfinite(rewards)):
            return
            
        current_reward = rewards[-1]
        best_reward = self.global_best_so_far
        best_iter = self.global_best_iteration
        current_iter = self.iteration
        
        gradient = 0.0
        
        # Calculate gradient between global best-so-far and current
        if current_iter > best_iter and current_iter != best_iter and np.isfinite(best_reward):
            time_diff = current_iter - best_iter
            if time_diff > 0:
                reward_diff = current_reward - best_reward
                gradient = reward_diff / time_diff
        else:
            # If current is the best or very early, calculate gradient from start of this model's history
            model_history_length = len(rewards)
            if model_history_length > 1:
                gradient = (current_reward - rewards[0]) / (model_history_length - 1)
        
        # Check if gradient is finite
        if not np.isfinite(gradient):
            gradient = 0.0
        
        # Smooth gradient belief
        alpha = 0.3
        new_gradient_belief = (1 - alpha) * self.gradient_belief[model_idx] + alpha * gradient
        
        if np.isfinite(new_gradient_belief):
            self.gradient_belief[model_idx] = new_gradient_belief
            self.gradient_history[model_idx].append(gradient)
    
    def _update_stability_belief(self, model_idx: int):
        """Update belief about gradient stability (consistency)."""
        if len(self.gradient_history[model_idx]) < 2:
            return
            
        gradients = np.array(self.gradient_history[model_idx])
        
        # Check if gradients contain valid values
        if not np.all(np.isfinite(gradients)) or len(gradients) == 0:
            return
        
        # Calculate variance safely
        gradient_var = np.var(gradients)
        if not np.isfinite(gradient_var):
            gradient_var = 0.0
            
        stability = 1.0 / (1.0 + gradient_var)  # Normalize to [0,1]
        
        # Check if stability is finite
        if not np.isfinite(stability):
            stability = 0.0
        
        # Smooth stability belief
        alpha = 0.1
        new_stability_belief = (1 - alpha) * self.stability_belief[model_idx] + alpha * stability
        
        if np.isfinite(new_stability_belief):
            self.stability_belief[model_idx] = new_stability_belief
    
    def _calculate_belief_score(self, model_idx: int) -> float:
        """Calculate overall belief score combining all factors."""
        perf = self.performance_belief[model_idx]
        grad = self.gradient_belief[model_idx]
        stab = self.stability_belief[model_idx]
        conf = self.belief_confidence[model_idx]
        
        # Check for NaN/inf values and replace with safe defaults
        if not np.isfinite(grad):
            grad = 0.0
        if not np.isfinite(stab):
            stab = 0.0
        
        # Gradient contribution: positive gradients boost score
        gradient_contrib = max(0, grad) if grad > self.gradient_threshold else grad * 0.5
        
        # Combined score
        score = (
            self.performance_weight * perf +
            self.gradient_weight * gradient_contrib +
            self.stability_weight * stab
        )
        
        # Weight by confidence
        final_score = score * conf
            
        return final_score
    
    def _calculate_all_belief_scores(self) -> np.ndarray:
        """Calculate belief scores for all models."""
        scores = np.array([self._calculate_belief_score(i) for i in range(self.n_models)])
        
        # Ensure no NaN values in the final scores
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
        
        return scores
    
    def sample(self) -> int:
        """Sample model based on belief network."""
        if self.iteration < self.explore:
            return self.random.randrange(self.n_models)
        
        # Calculate belief scores
        belief_scores = self._calculate_all_belief_scores()
        
        # Add exploration noise
        exploration_noise = np.random.normal(0, 1, self.n_models)
        final_scores = belief_scores + exploration_noise
        
        # Ensure final_scores are finite
        # final_scores = np.nan_to_num(final_scores, nan=0.5, posinf=1.0, neginf=0.0)
        
        return int(np.argmax(final_scores))
    
    def update(self, model_idx: int, reward: float):
        """Update all belief components."""
        self.iteration += 1
        
        # Validate reward before storing
        if not np.isfinite(reward):
            print(f"Warning: Invalid reward {reward} received for model {model_idx}")
            reward = 0.0  # Use safe default
        
        # Always store reward for tracking purposes
        self.reward_history[model_idx].append(reward)
        
        # Only update beliefs if we're past the exploration phase
        if self.iteration > self.explore:
            # Update all belief components
            self._update_performance_belief(model_idx, reward)
            self._update_gradient_belief(model_idx)
            self._update_stability_belief(model_idx)
            
            # Update confidence (more samples = higher confidence)
            new_confidence = min(len(self.reward_history[model_idx]) / self.window, 1.0)
            if np.isfinite(new_confidence):
                self.belief_confidence[model_idx] = new_confidence
            
            # Calculate and store current belief scores for all models
            current_belief_scores = self._calculate_all_belief_scores()
            self.belief_scores_history.append(current_belief_scores.copy())
        else:
            # During exploration, track global best but don't update beliefs
            if reward > self.global_best_so_far:
                self.global_best_so_far = reward
                self.global_best_iteration = self.iteration
                self.global_best_model = model_idx
    
    def get_belief_state(self) -> dict:
        """Get complete belief state for analysis."""
        return {
            'performance_beliefs': self.performance_belief.copy(),
            'gradient_beliefs': self.gradient_belief.copy(),
            'stability_beliefs': self.stability_belief.copy(),
            'belief_confidence': self.belief_confidence.copy(),
            'global_best_so_far': self.global_best_so_far,
            'global_best_iteration': self.global_best_iteration,
            'global_best_model': self.global_best_model,
            # 'belief_scores_history': np.array(self.belief_scores_history),
            'current_belief_scores': self._calculate_all_belief_scores(),
            'is_exploring': self.iteration <= self.explore,
            'current_gradients': [
                self._calculate_gradient_simple(i) for i in range(self.n_models)
            ]
        }
    
    def get_belief_scores_at_iteration(self, iteration: int) -> np.ndarray:
        """Get belief scores at a specific iteration."""
        if 0 <= iteration < len(self.belief_scores_history):
            return self.belief_scores_history[iteration].copy()
        else:
            raise IndexError(f"Iteration {iteration} not found. Available range: 0-{len(self.belief_scores_history)-1}")
    
    def get_latest_belief_scores(self) -> np.ndarray:
        """Get the most recent belief scores."""
        if self.belief_scores_history:
            return self.belief_scores_history[-1].copy()
        else:
            return np.full(self.n_models, 0.5)
    
    def _calculate_gradient_simple(self, model_idx: int) -> float:
        """Simple gradient calculation using global best-so-far approach."""
        if len(self.reward_history[model_idx]) < 2:
            return 0.0
            
        rewards = list(self.reward_history[model_idx])
        
        # Check for valid rewards
        if not all(np.isfinite(r) for r in rewards):
            return 0.0
            
        current_reward = rewards[-1]
        best_reward = self.global_best_so_far
        best_iter = self.global_best_iteration
        current_iter = self.iteration
        
        gradient = 0.0
        
        if current_iter > best_iter and current_iter != best_iter and np.isfinite(best_reward):
            time_diff = current_iter - best_iter
            if time_diff > 0:
                gradient = (current_reward - best_reward) / time_diff
        else:
            # If current is from recent iterations, calculate from start of this model's history
            model_history_length = len(rewards)
            if model_history_length > 1:
                gradient = (current_reward - rewards[0]) / (model_history_length - 1)
        
        # Return 0.0 if gradient is not finite
        return gradient if np.isfinite(gradient) else 0.0



class StochasticLearningAutomaton:
    """
    Linear Reward-Inaction (LRI) style stochastic learning automaton.

    Args:
        n_models: number of arms
        beta: learning rate (SLA learning rate in the paper; e.g., 0.5 used by authors)
        explore: number of initial pure-random exploration steps
        init_probs: optional initial probability vector (defaults to uniform)
        random_seed: optional seed for reproducibility
        reward_threshold: when treating reward as binary we compare > reward_threshold as success; default 0.0
    """
    def __init__(
        self,
        n_models: int,
        beta: float = 0.5,
        explore: int = 10,
        init_probs: Optional[List[float]] = None,
        random_seed: Optional[int] = None,
        reward_threshold: float = 0.0,
    ):
        if n_models < 2:
            raise ValueError("SLA requires at least 2 arms")
        self.n_models = n_models
        if init_probs is None:
            self.p = np.full(n_models, 1.0 / n_models, dtype=float)
        else:
            arr = np.array(init_probs, dtype=float)
            if arr.size != n_models:
                raise ValueError("init_probs length mismatch")
            arr = np.clip(arr, 1e-12, None)
            arr = arr / arr.sum()
            self.p = arr
        self.beta = float(beta)
        self.iteration = 0
        self.explore = int(explore)
        self.random = random.Random(random_seed)
        self.reward_threshold = float(reward_threshold)

    def sample(self) -> int:
        """
        Return chosen arm index.
        During initial explore phase, use uniform random; afterwards sample using current p.
        """
        if self.iteration < self.explore:
            return self.random.randrange(self.n_models)
        # sample according to probability vector self.p
        return self.random.choices(range(self.n_models), weights=self.p.tolist(), k=1)[0]

    def update(self, model_idx: int, reward: float):
        """
        Update action probabilities using LRI rule blended for continuous reward in [0,1).

        reward: float, expected in [0,1). If evaluator returns unbounded metrics, normalize before calling.
        """
        # clamp reward
        r = float(reward)
        r = 0.0 if np.isnan(r) else r
        r = min(max(r, 0.0), 1.0)

        self.iteration += 1

        # Save old probs for calculation
        p_old = self.p.copy()
        i = int(model_idx)
        n = self.n_models
        b = self.beta

        # Success update (LRI): p_i <- p_i + b*(1 - p_i), p_j <- (1 - b)*p_j
        p_success = (1 - b) * p_old
        p_success[i] = p_success[i] + b * (1.0 - p_old[i])

        # Penalty update (LR-P): p_i <- (1 - b) * p_i, distribute b among others
        p_penalty = p_old.copy()
        p_penalty[i] = (1 - b) * p_old[i]
        # distribute mass b * p_old[i] proportionally to others or equally:
        # We add equal share of b * p_old[i] to other arms:
        if n > 1:
            add = b * p_old[i] / (n - 1)
            for j in range(n):
                if j != i:
                    p_penalty[j] = p_penalty[j] + add

        # Blend using continuous reward r: final = r * success + (1-r) * penalty
        p_new = r * p_success + (1.0 - r) * p_penalty

        # Numerical safety: clip and renormalize
        p_new = np.clip(p_new, 1e-12, None)
        p_new = p_new / p_new.sum()

        self.p = p_new

    def get_probs(self) -> np.ndarray:
        return self.p.copy()



class CostAwareAsymmetricTS:
    """Thompson Sampling with both cost and performance asymmetry"""
    
    def __init__(self, n_models=2, costs=[1.0, 10.0],  # 3B first, 32B second
                 amplify_cheap_success=1.0, penalize_expensive_failure=0.5, explore=4): # have to re-factor this
        self.iteration = 0
        self.explore = explore
        self.n_models = n_models
        self.costs = np.array(costs)
        self.alpha = np.ones(n_models)  # Start at 1 (minimum for Beta distribution)
        self.beta = np.ones(n_models)   # Start at 1 (minimum for Beta distribution)
        self.random = random.Random()
        
        # Asymmetric factors
        self.amplify_cheap = amplify_cheap_success
        self.penalize_expensive = penalize_expensive_failure
        
        # Track performance metrics
        self.total_cost = 0
        self.performance_history = []
        
        # Model indices - make it clear which is which
        self.CHEAP_MODEL = 0  # 3B model
        self.EXPENSIVE_MODEL = 1  # 32B model
        
    def sample(self) -> int:
        """Sample with cost-adjusted probabilities"""
        # Exploration phase - alternate between models
        if self.iteration < self.explore:
            return self.iteration % self.n_models
        
        # Ensure alpha and beta are valid for Beta distribution
        safe_alpha = np.maximum(self.alpha, 1e-6)
        safe_beta = np.maximum(self.beta, 1e-6)
        
        # Sample from posteriors
        thetas = [np.random.beta(safe_alpha[i], safe_beta[i]) 
                  for i in range(self.n_models)]
        
        # Adjust for cost AND expected asymmetric rewards
        adjusted_values = []
        for i, theta in enumerate(thetas):
            # Expected value considering asymmetric rewards
            if i == self.CHEAP_MODEL:  # 3B model
                # Account for amplification on success
                expected_value = theta * self.amplify_cheap + (1-theta) * 1.0
            else:  # 32B model
                # Account for penalty on failure
                expected_value = theta * 1.0 + (1-theta) * self.penalize_expensive
            
            # Divide by cost
            adjusted_values.append(expected_value / self.costs[i])
        
        selected = np.argmax(adjusted_values)
        self.total_cost += self.costs[selected]
        
        return selected
    
    def update(self, model_idx: int, reward: float):
        """Update with cost-weighted asymmetric rewards"""
        self.iteration += 1
        
        # Store original reward for logging
        original_reward = reward
        adjusted_reward = reward
        
        # Apply asymmetric modifications based on model and outcome
        if model_idx == self.CHEAP_MODEL and reward > 0.5:  # 3B success
            # Amplify based on cost savings
            adjusted_reward = min(reward * self.amplify_cheap, 1.0)
            print(f"3B Success: {reward:.3f} -> {adjusted_reward:.3f} (amplify={self.amplify_cheap})")
            
        elif model_idx == self.EXPENSIVE_MODEL and reward == 0.0:  # 32B failure
            # Penalize based on wasted cost
            adjusted_reward = max(reward * self.penalize_expensive, 0.0)
            print(f"32B Failure: {reward:.3f} -> {adjusted_reward:.3f} (penalty={self.penalize_expensive})")
        
        # For negative rewards, treat them as failures
        if original_reward < 0:
            adjusted_reward = 0.0  # Treat negative rewards as complete failures
            
        # Update Beta parameters with adjusted reward
        self._update_beta_params(model_idx, adjusted_reward)
        
        # Track performance
        self.performance_history.append({
            'model': model_idx,
            'raw_reward': original_reward,
            'adjusted_reward': adjusted_reward,
            'cumulative_cost': self.total_cost
        })
    
    def _update_beta_params(self, model_idx: int, adjusted_reward: float):
        """Update Beta parameters with continuous rewards"""
        # Ensure adjusted_reward is in valid range [0, 1]
        adjusted_reward = np.clip(adjusted_reward, 0.0, 1.0)
        
        if adjusted_reward > 0.5:
            # Success-like outcome
            increment = adjusted_reward
            self.alpha[model_idx] += increment
        else:
            # Failure-like outcome
            increment = 1 - adjusted_reward
            self.beta[model_idx] += increment
        
        # Ensure minimum values for Beta distribution
        self.alpha[model_idx] = max(self.alpha[model_idx], 0.01)
        self.beta[model_idx] = max(self.beta[model_idx], 0.01)


class LinUCB:
    """Linear Upper Confidence Bound (LinUCB) bandit sampler."""
    
    def __init__(self, n_models: int, feature_dim: int = 100, alpha: float = 1.0, 
                 lambda_reg: float = 1.0, explore: int = 0):
        self.n_models = n_models
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.iteration = 0
        self.explore = explore
        self.random = random.Random()
        
        # Initialize LinUCB parameters for each arm
        self.A = [np.identity(feature_dim) * lambda_reg for _ in range(n_models)]
        self.b = [np.zeros(feature_dim) for _ in range(n_models)]
        
        # Store the last context for update
        self.last_context = None
        
    def set_context(self, context: np.ndarray):
        """Set the context (feature vector) for the current decision."""
        if context.shape[0] != self.feature_dim:
            raise ValueError(f"Context dimension {context.shape[0]} != expected {self.feature_dim}")
        self.last_context = context
        
    def sample(self) -> int:
        """Select arm using LinUCB algorithm with current context."""
        if self.last_context is None:
            raise ValueError("Context not set. Call set_context() before sampling.")
            
        # Pure exploration phase
        if self.iteration < self.explore:
            return self.random.randrange(self.n_models)
        
        # Calculate UCB scores for all arms
        ucb_scores = []
        
        for i in range(self.n_models):
            try:
                # Solve for theta: A^{-1}b
                theta = np.linalg.solve(self.A[i], self.b[i])
                
                # Calculate confidence bound
                # UCB = x^T theta + alpha * sqrt(x^T A^{-1} x)
                A_inv = np.linalg.inv(self.A[i])
                confidence = self.alpha * np.sqrt(
                    self.last_context.T @ A_inv @ self.last_context
                )
                
                expected_reward = self.last_context.T @ theta
                ucb_score = expected_reward + confidence
                
            except np.linalg.LinAlgError:
                # If matrix is singular, use a default score
                ucb_score = 0.0
                
            ucb_scores.append(ucb_score)
        
        return int(np.argmax(ucb_scores))
    
    def update(self, model_idx: int, reward: float):
        """Update LinUCB parameters for the chosen arm."""
        if self.last_context is None:
            raise ValueError("Context not set. Cannot update without context.")
            
        self.iteration += 1
        
        # Update A and b for the chosen arm
        # A_a = A_a + x_t x_t^T
        # b_a = b_a + r_t x_t
        self.A[model_idx] += np.outer(self.last_context, self.last_context)
        self.b[model_idx] += self.last_context * reward


class HybridLinUCB:
    """Hybrid LinUCB with shared and per-arm features."""
    
    def __init__(self, n_models: int, shared_feature_dim: int = 5, 
                 per_arm_feature_dim: int = 5, alpha: float = 1.0, 
                 lambda_reg: float = 1.0, explore: int = 0):
        self.n_models = n_models
        self.shared_dim = shared_feature_dim
        self.per_arm_dim = per_arm_feature_dim
        self.feature_dim = shared_dim + per_arm_feature_dim  # k + d
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.iteration = 0
        self.explore = explore
        self.random = random.Random()
        
        # Shared parameters
        self.A0 = np.identity(shared_dim) * lambda_reg
        self.b0 = np.zeros(shared_dim)
        
        # Per-arm parameters
        self.A = [np.identity(per_arm_dim) * lambda_reg for _ in range(n_models)]
        self.B = [np.zeros((per_arm_dim, shared_dim)) for _ in range(n_models)]
        self.b = [np.zeros(per_arm_dim) for _ in range(n_models)]
        
        # Store contexts
        self.last_shared_context = None
        self.last_per_arm_contexts = None
        
    def set_context(self, shared_context: np.ndarray, 
                   per_arm_contexts: List[np.ndarray] = None):
        """
        Set contexts for hybrid LinUCB.
        
        Args:
            shared_context: Shared features across all arms
            per_arm_contexts: List of per-arm specific features (optional)
        """
        if shared_context.shape[0] != self.shared_dim:
            raise ValueError(f"Shared context dimension mismatch")
            
        self.last_shared_context = shared_context
        
        if per_arm_contexts is None:
            # If no per-arm features provided, use zeros
            self.last_per_arm_contexts = [
                np.zeros(self.per_arm_dim) for _ in range(self.n_models)
            ]
        else:
            if len(per_arm_contexts) != self.n_models:
                raise ValueError("Number of per-arm contexts must match n_models")
            for i, ctx in enumerate(per_arm_contexts):
                if ctx.shape[0] != self.per_arm_dim:
                    raise ValueError(f"Per-arm context {i} dimension mismatch")
            self.last_per_arm_contexts = per_arm_contexts
    
    def sample(self) -> int:
        """Select arm using Hybrid LinUCB algorithm."""
        if self.last_shared_context is None:
            raise ValueError("Context not set. Call set_context() before sampling.")
            
        # Pure exploration phase
        if self.iteration < self.explore:
            return self.random.randrange(self.n_models)
        
        # Calculate UCB scores for all arms
        ucb_scores = []
        
        # Precompute shared component
        try:
            A0_inv = np.linalg.inv(self.A0)
            beta = A0_inv @ self.b0
        except np.linalg.LinAlgError:
            A0_inv = np.identity(self.shared_dim)
            beta = np.zeros(self.shared_dim)
        
        for i in range(self.n_models):
            try:
                z_i = self.last_shared_context
                x_i = self.last_per_arm_contexts[i]
                
                # Compute theta for arm i
                A_i_inv = np.linalg.inv(self.A[i])
                B_i_T_A_i_inv = self.B[i].T @ A_i_inv
                
                # theta = A_i^{-1}(b_i - B_i^T beta)
                theta_i = A_i_inv @ (self.b[i] - self.B[i].T @ beta)
                
                # Compute confidence bound
                s_i = z_i.T @ A0_inv @ z_i
                s_i -= 2 * z_i.T @ A0_inv @ B_i_T_A_i_inv @ x_i
                s_i += x_i.T @ A_i_inv @ x_i
                s_i += x_i.T @ A_i_inv @ self.B[i] @ A0_inv @ B_i_T_A_i_inv @ x_i
                
                # Expected reward
                expected = z_i.T @ beta + x_i.T @ theta_i
                
                # UCB score
                confidence = self.alpha * np.sqrt(max(s_i, 0))  # Ensure non-negative
                ucb_score = expected + confidence
                
            except np.linalg.LinAlgError:
                ucb_score = 0.0
                
            ucb_scores.append(ucb_score)
        
        return int(np.argmax(ucb_scores))
    
    def update(self, model_idx: int, reward: float):
        """Update Hybrid LinUCB parameters."""
        if self.last_shared_context is None:
            raise ValueError("Context not set. Cannot update without context.")
            
        self.iteration += 1
        
        z = self.last_shared_context
        x = self.last_per_arm_contexts[model_idx]
        
        # Update per-arm parameters
        self.A[model_idx] += np.outer(x, x)
        self.B[model_idx] += np.outer(x, z)
        self.b[model_idx] += x * reward
        
        # Update shared parameters
        try:
            A_inv = np.linalg.inv(self.A[model_idx])
            self.A0 += self.B[model_idx].T @ A_inv @ self.B[model_idx]
            self.b0 += self.B[model_idx].T @ A_inv @ self.b[model_idx]
        except np.linalg.LinAlgError:
            # Skip update if matrix is singular
            pass


class SimpleFeatureLinUCB(LinUCB):
    """
    LinUCB with simple automatic feature extraction.
    This version doesn't require external context setting.
    """
    
    def __init__(self, n_models: int, alpha: float = 1.0, 
                 lambda_reg: float = 1.0, explore: int = 0):
        # Simple features: [1, iteration_normalized, recent_rewards_per_model]
        feature_dim = 1 + 1 + n_models
        super().__init__(n_models, feature_dim, alpha, lambda_reg, explore)
        
        # Track recent rewards for feature construction
        self.recent_rewards = [[] for _ in range(n_models)]
        self.max_history = 10
        
    def sample(self) -> int:
        """Generate features automatically and sample."""
        # Construct simple context features
        features = []
        
        # Bias term
        features.append(1.0)
        
        # Normalized iteration count
        features.append(min(self.iteration / 100.0, 1.0))
        
        # Average recent rewards per model
        for i in range(self.n_models):
            if self.recent_rewards[i]:
                avg_reward = np.mean(self.recent_rewards[i][-self.max_history:])
            else:
                avg_reward = 0.5  # Prior
            features.append(avg_reward)
        
        self.set_context(np.array(features))
        return super().sample()
    
    def update(self, model_idx: int, reward: float):
        """Update with automatic feature generation."""
        # Store reward for future feature construction
        self.recent_rewards[model_idx].append(reward)
        
        # Call parent update
        super().update(model_idx, reward)



# Registry and factory
SAMPLING_FUNCTIONS = {
    "random":            RandomSampling,
    "thompson":          ThompsonSampling,
    "gaussian_thompson": GaussianThompsonSampling,
    "ucb1":              UCB1,
    "cost_aware_asymmetric_ts" : CostAwareAsymmetricTS,
    "sla": StochasticLearningAutomaton,
    "gbfn" : GradientBeliefNetwork,
}

def get_sampling_function(fn_name: str, n_models: int, **kwargs):
    if fn_name not in SAMPLING_FUNCTIONS:
        raise ValueError(f"Unknown sampling fn '{fn_name}'. Valid: {list(SAMPLING_FUNCTIONS)}")
    return SAMPLING_FUNCTIONS[fn_name](n_models, **kwargs)
