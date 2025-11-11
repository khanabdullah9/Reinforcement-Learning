import numpy as np
import random

class GridWorld:
    def __init__(self, rows=5, cols=5, goal=(4, 4), walls=[]):
        self.rows = rows
        self.cols = cols
        self.goal = goal
        self.walls = set(walls)
        self.reset()

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        r, c = self.state
        if action == 0:    # up
            r = max(r - 1, 0)
        elif action == 1:  # down
            r = min(r + 1, self.rows - 1)
        elif action == 2:  # left
            c = max(c - 1, 0)
        elif action == 3:  # right
            c = min(c + 1, self.cols - 1)

        next_state = (r, c)
        reward = -1
        done = False

        if next_state in self.walls:
            reward = -5
            next_state = self.state
        elif next_state == self.goal:
            reward = 10
            done = True

        self.state = next_state
        return next_state, reward, done

    def get_state_space(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.walls]

    def render(self):
        grid = np.full((self.rows, self.cols), "_", dtype=str)
        for (r, c) in self.walls:
            grid[r, c] = "#"
        r, c = self.goal
        grid[r, c] = "G"
        r, c = self.state
        grid[r, c] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print()


# -----------------------------
# SARSA Agent
# -----------------------------
class SARSAAgent:
    def __init__(self, rows, cols, actions=[0,1,2,3], alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((rows, cols, len(actions)))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        r, c = state
        return np.argmax(self.Q[r, c])

    def update(self, state, action, reward, next_state, next_action):
        r, c = state
        nr, nc = next_state
        predict = self.Q[r, c, action]
        target = reward + self.gamma * self.Q[nr, nc, next_action]
        self.Q[r, c, action] += self.alpha * (target - predict)

    def expected_update(self, state, action, reward, next_state):
        """
        Expected SARSA:
            Assigns (weights) probabilities to the Q-Value for next state-action pair
            Probability:
                - Exploration: ε / n_actions
                - Exploitation: (1 - ε) / n_actions
        """
        r, c = state
        nr, nc = next_state
    
        q_next = self.Q[nr, nc]
        best_action = np.argmax(q_next)
        n_actions = len(self.actions)
    
        # Compute expected Q under ε-greedy policy
        expected_q = 0
        for a in range(n_actions):
            if a == best_action:
                prob = 1 - self.epsilon + (self.epsilon / n_actions)
            else:
                prob = self.epsilon / n_actions
            expected_q += prob * q_next[a]
    
        # SARSA expected target
        predict = self.Q[r, c, action]
        target = reward + self.gamma * expected_q
        self.Q[r, c, action] += self.alpha * (target - predict)


    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# -----------------------------
# Training
# -----------------------------
def train(env, agent, episodes=1000, max_steps=100):
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0

        for step in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            # agent.update(state, action, reward, next_state, next_action)
            agent.update(state, action, reward, next_state, next_action) # Using Expected SARSA algorithm

            state, action = next_state, next_action
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards[-100:])
            print(f"Episode {ep+1}/{episodes} | Avg Reward (last 100): {avg_r:.2f} | Epsilon: {agent.epsilon:.3f}")

    return rewards


# -----------------------------
# Testing
# -----------------------------
def test(env, agent, max_steps=50):
    state = env.reset()
    total_reward = 0
    for step in range(max_steps):
        env.render()
        action = np.argmax(agent.Q[state[0], state[1]])
        print(f"Selected action: {action}")
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            print("Goal reached!")
            break
    print(f"Total Reward: {total_reward}")
