import numpy as np
import random

# ----------------------------
# 1️⃣ Define a 2D Grid Environment
# ----------------------------
class GridWorld:
    def __init__(self, rows=5, cols=5, goal=(4, 4), walls=None):
        self.rows = rows
        self.cols = cols
        self.goal = goal
        self.walls = walls if walls else []
        self.action_space = [0, 1, 2, 3]  # Up, Down, Left, Right
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        return self.agent_pos

    def step(self, action):
        row, col = self.agent_pos

        # Move depending on action
        if action == 0:    # Up
            row = max(row - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, self.rows - 1)
        elif action == 2:  # Left
            col = max(col - 1, 0)
        elif action == 3:  # Right
            col = min(col + 1, self.cols - 1)

        next_pos = (row, col)

        # Hit a wall
        if next_pos in self.walls:
            next_pos = self.agent_pos  # stay in place

        self.agent_pos = next_pos

        # Reward logic
        if self.agent_pos == self.goal:
            reward = 10
            done = True
        else:
            reward = -1  # small penalty for each move
            done = False

        return self.agent_pos, reward, done

    def render(self):
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        r, c = self.goal
        grid[r][c] = "G"
        for (wr, wc) in self.walls:
            grid[wr][wc] = "#"
        ar, ac = self.agent_pos
        grid[ar][ac] = "A"
        for row in grid:
            print(" ".join(row))
        print()


# ----------------------------
# 2️⃣ Q-Learning Agent
# ----------------------------
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.Q = {}  # key: (state), value: [Q for each action]
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_Q(self, state):
        if state not in self.Q:
            self.Q[state] = [0.0 for _ in self.actions]
        return self.Q[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        q_values = self.get_Q(state)
        return int(np.argmax(q_values))

    #using bellman
    def update(self, state, action, reward, next_state):
        q_values = self.get_Q(state)
        next_q_values = self.get_Q(next_state)
        best_next = max(next_q_values)
        q_values[action] += self.alpha * (reward + self.gamma * best_next - q_values[action])


# ----------------------------
# 3️⃣ Training
# ----------------------------
def train(env, agent, episodes=500, max_steps=50):
    for ep in range(episodes):
        state = env.reset()
        done = False

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

            if done:
                break

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{episodes} completed.")


# ----------------------------
# 4️⃣ Testing
# ----------------------------
def test(env, agent, max_steps=20):
    state = env.reset()
    done = False
    total_reward = 0
    print("\nTesting learned policy:\n")

    for step in range(max_steps):
        env.render()
        action = np.argmax(agent.get_Q(state))
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            env.render()
            print(f"Goal reached in {step+1} steps!")
            break

    print(f"Total reward in test: {total_reward}\n")


# ----------------------------
# 5️⃣ Run Everything
# ----------------------------
if __name__ == "__main__":
    walls = [(1, 1), (2, 2), (3, 1)]
    env = GridWorld(rows=5, cols=5, goal=(2,3), walls=walls)
    agent = QLearningAgent(actions=[0, 1, 2, 3])

    train(env, agent, episodes=1_000)
    test(env, agent)
