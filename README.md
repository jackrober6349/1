import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Callable
import networkx as nx

class CausalEnvironment:
    """Simulated environment with causal structure."""
    def __init__(self, causal_graph: nx.DiGraph):
        self.causal_graph = causal_graph
        self.state = np.zeros(len(causal_graph.nodes))
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = np.random.normal(0, 1, len(self.causal_graph.nodes))
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take an action and return next state, reward, done, and info."""
        # Apply action's direct effect
        action_node = f"A{action}"
        for effect_node in self.causal_graph.successors(action_node):
            node_idx = int(effect_node[1:])  # Get node index from name like "S1"
            self.state[node_idx] += np.random.normal(0.5, 0.1)  # Direct effect
            
        # Propagate indirect effects through causal graph
        for _ in range(2):  # Allow effects to propagate through 2 hops
            state_copy = self.state.copy()
            for node in self.causal_graph.nodes:
                if node.startswith("S"):  # State nodes
                    node_idx = int(node[1:])
                    predecessors = list(self.causal_graph.predecessors(node))
                    if predecessors:
                        effect = sum(state_copy[int(p[1:])] * 0.3 for p in predecessors if p.startswith("S"))
                        self.state[node_idx] += effect
        
        # Calculate reward based on causal effects
        reward = self._calculate_reward(action)
        
        # Check if done (simplified)
        done = False
        
        return self.state.copy(), reward, done, {}
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward using causal effects (including counterfactuals)."""
        # In a real implementation, this would use a causal model
        # Here we simulate by summing relevant state variables
        action_node = f"A{action}"
        relevant_states = []
        
        # Get direct and indirect effects of action
        for effect_node in nx.descendants(self.causal_graph, action_node):
            if effect_node.startswith("S"):
                relevant_states.append(int(effect_node[1:]))
        
        # Calculate reward as sum of relevant states
        reward = np.sum(self.state[relevant_states])
        
        # Add counterfactual component (simulated)
        counterfactual_penalty = 0.0
        for a in range(3):  # Assume 3 possible actions
            if a != action:
                # Simulate penalty for not taking other actions
                counterfactual_penalty += np.random.normal(0, 0.1)
                
        return reward + counterfactual_penalty

class CausalRLAgent:
    """Reinforcement learning agent that uses causal inference."""
    def __init__(self, state_dim: int, action_dim: int, causal_graph: nx.DiGraph):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.causal_graph = causal_graph
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Causal effect estimator
        self.causal_estimator = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)  # Predicts state changes
        )
        
        self.optimizer = optim.Adam(list(self.policy.parameters()) + 
                                    list(self.causal_estimator.parameters()), lr=0.001)
        self.gamma = 0.99  # Discount factor
        
    def select_action(self, state: np.ndarray) -> int:
        """Select action based on policy."""
        state_tensor = torch.FloatTensor(state)
        action_probs = self.policy(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        return action
    
    def train(self, states: List[np.ndarray], actions: List[int], 
              rewards: List[float], next_states: List[np.ndarray]):
        """Train the agent using causal reinforcement learning."""
        # Convert to tensors
        state_tensor = torch.FloatTensor(states)
        action_tensor = torch.LongTensor(actions)
        reward_tensor = torch.FloatTensor(rewards)
        next_state_tensor = torch.FloatTensor(next_states)
        
        # One-hot encode actions
        action_onehot = torch.zeros(len(actions), self.action_dim)
        action_onehot.scatter_(1, action_tensor.unsqueeze(1), 1)
        
        # Calculate policy loss
        log_probs = torch.log(self.policy(state_tensor))
        action_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze()
        
        # Calculate causal advantage using counterfactual reasoning
        causal_advantage = self._calculate_causal_advantage(
            state_tensor, action_onehot, reward_tensor, next_state_tensor
        )
        
        # Policy gradient loss
        policy_loss = -(action_log_probs * causal_advantage.detach()).mean()
        
        # Causal estimator loss
        state_action = torch.cat([state_tensor, action_onehot], dim=1)
        predicted_state_changes = self.causal_estimator(state_action)
        actual_state_changes = next_state_tensor - state_tensor
        causal_loss = nn.MSELoss()(predicted_state_changes, actual_state_changes)
        
        # Total loss
        loss = policy_loss + causal_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _calculate_causal_advantage(self, states: torch.Tensor, actions: torch.Tensor, 
                                   rewards: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """Calculate advantage using causal effects and counterfactuals."""
        # Get predicted state changes from causal estimator
        state_action = torch.cat([states, actions], dim=1)
        predicted_changes = self.causal_estimator(state_action)
        
        # Calculate causal effect-based reward
        causal_rewards = []
        for i in range(len(states)):
            state_change = predicted_changes[i]
            # Simulate causal effect propagation through graph
            causal_reward = self._simulate_causal_reward(states[i], state_change, actions[i])
            causal_rewards.append(causal_reward)
        
        causal_rewards = torch.FloatTensor(causal_rewards)
        
        # Combine actual and causal rewards
        advantage = rewards + 0.5 * causal_rewards
        return advantage
    
    def _simulate_causal_reward(self, state: torch.Tensor, state_change: torch.Tensor, 
                               action: torch.Tensor) -> float:
        """Simulate causal reward propagation through the causal graph."""
        # In a real implementation, this would use the causal graph structure
        # Here we simplify by summing weighted state changes
        causal_reward = torch.sum(state_change * state).item()
        
        # Add counterfactual component
        for a_idx in range(self.action_dim):
            if action[a_idx] == 0:  # Counterfactual action
                counterfactual_action = torch.zeros_like(action)
                counterfactual_action[a_idx] = 1
                state_action = torch.cat([state, counterfactual_action])
                counterfactual_change = self.causal_estimator(state_action.unsqueeze(0)).squeeze()
                # Negative penalty for not taking this action
                causal_reward -= 0.1 * torch.sum(counterfactual_change * state).item()
        
        return causal_reward

# Example usage
if __name__ == "__main__":
    # Create causal graph (S0-S3 are state nodes, A0-A2 are actions)
    G = nx.DiGraph()
    # Add state nodes
    for i in range(4):
        G.add_node(f"S{i}")
    
    # Add action nodes
    for i in range(3):
        G.add_node(f"A{i}")
    
    # Add causal edges (actions affecting states)
    G.add_edge("A0", "S0")
    G.add_edge("A0", "S1")
    G.add_edge("A1", "S1")
    G.add_edge("A1", "S2")
    G.add_edge("A2", "S2")
    G.add_edge("A2", "S3")
    
    # Add state interactions
    G.add_edge("S0", "S1")
    G.add_edge("S1", "S2")
    G.add_edge("S2", "S3")
    
    # Initialize environment and agent
    env = CausalEnvironment(G)
    agent = CausalRLAgent(state_dim=4, action_dim=3, causal_graph=G)
    
    # Training loop
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards, next_states = [], [], [], []
        
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            
            state = next_state
            
            if done:
                break
        
        # Train agent
        loss = agent.train(states, actions, rewards, next_states)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Reward: {np.mean(rewards):.4f}, Loss: {loss:.4f}")
