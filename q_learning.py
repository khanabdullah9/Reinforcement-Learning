import numpy as np
import random

# --- 1. Define your custom environment ---
class SimpleGrid:
    def __init__(self, size=5):
        self.size = size
        self.start = 0
        self.goal = size - 1
        self.state = self.start

    def reset(self):
        """Reset the environment to start position."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Action: 0 = left, 1 = right
        Returns: next_state, reward, done
        """
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(self.size - 1, self.state + 1)

        # Define rewards
        if self.state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1  # small penalty per step
            done = False

        return self.state, reward, done

    def render(self):
        """Visualize the current state."""
        grid = ["⬜"] * self.size
        grid[self.goal] = "🏁"
        grid[self.state] = "🤖"
        print(" ".join(grid))


# --- 2. Define the Q-learning Agent ---
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        """Standard Q-learning update rule."""
        best_next = np.max(self.Q[next_state])
        target = reward if done else reward + self.gamma * best_next
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def decay_epsilon(self):
        """Reduce exploration gradually."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# --- 3. Train the Agent ---
def train(env, agent, episodes=200, max_steps=20):
    all_rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        all_rewards.append(total_reward)
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{episodes} — Total Reward: {total_reward} — Epsilon: {agent.epsilon:.3f}")

    print("Training complete!")
    return all_rewards


# --- 4. Test the Learned Policy ---
def test(env, agent, max_steps=10):
    print(f"Comencing tests!")
    state = env.reset()
    done = False
    total_reward = 0
    print("\nTesting learned policy:\n")
    for _ in range(max_steps):
        env.render()
        action = np.argmax(agent.Q[state])
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            env.render()
            break
    print(f"Total reward in test: {total_reward}\n")


# --- 5. Run Everything ---
if __name__ == "__main__":
    env = SimpleGrid(size=5)
    agent = QLearningAgent(n_states=env.size, n_actions=4, alpha=0.5, gamma=0.9, epsilon=1.0)

    rewards = train(env, agent, episodes=200)
    test(env, agent)

    print("\nLearned Q-table:")
    print(agent.Q)
