# Assignment 11: Reinforcement Learning (Q-Learning for Grid World)

import random
import numpy as np
import matplotlib.pyplot as plt

# 1. Environment definition (4x4 Grid World)
ROWS, COLS = 4, 4
START, GOAL, HAZARD = (0, 0), (3, 3), (2, 2)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
ACTION_NAMES = ['^', 'v', '<', '>']

def to_state(r, c):
    return r * COLS + c

def step(pos, action_idx):
    dr, dc = ACTIONS[action_idx]
    nr = min(max(pos[0] + dr, 0), ROWS - 1)
    nc = min(max(pos[1] + dc, 0), COLS - 1)
    nxt = (nr, nc)
    
    if nxt == GOAL:
        return nxt, 10.0, True
    if nxt == HAZARD:
        return nxt, -10.0, True
    return nxt, -1.0, False

# 2. Q-Learning parameters
alpha = 0.1          # Learning rate
gamma = 0.95         # Discount factor
episodes = 1000      # Training episodes
eps_start = 1.0      # Initial exploration rate
eps_end = 0.05       # Minimum exploration rate

Q = np.zeros((ROWS * COLS, len(ACTIONS)))
rewards_history = []

random.seed(42)
np.random.seed(42)

# 3. Training agent with epsilon-greedy strategy
for ep in range(episodes):
    pos = START
    eps = eps_end + (eps_start - eps_end) * np.exp(-0.005 * ep)
    total_reward = 0
    
    for _ in range(50):
        s = to_state(*pos)
        
        # Exploration vs Exploitation
        if random.random() < eps:
            action = random.randrange(len(ACTIONS))
        else:
            action = int(np.argmax(Q[s]))
            
        nxt, reward, done = step(pos, action)
        ns = to_state(*nxt)
        
        # Q-Learning update rule
        target = reward if done else reward + gamma * np.max(Q[ns])
        Q[s, action] += alpha * (target - Q[s, action])
        
        total_reward += reward
        pos = nxt
        if done:
            break
            
    rewards_history.append(total_reward)

# 4. Display learned optimal policy
print("Learned Optimal Policy Grid (^: Up, v: Down, <: Left, >: Right, G: Goal, X: Hazard):")
for r in range(ROWS):
    row_symbols = []
    for c in range(COLS):
        pos = (r, c)
        if pos == GOAL:
            row_symbols.append(" G ")
        elif pos == HAZARD:
            row_symbols.append(" X ")
        else:
            best_act = int(np.argmax(Q[to_state(r, c)]))
            row_symbols.append(f" {ACTION_NAMES[best_act]} ")
    print(" ".join(row_symbols))

# 5. Evaluate greedy policy execution
pos = START
path = [pos]
test_reward = 0
for _ in range(20):
    action = int(np.argmax(Q[to_state(*pos)]))
    pos, reward, done = step(pos, action)
    path.append(pos)
    test_reward += reward
    if done:
        break

print(f"\nGreedy Path from Start: {' -> '.join(str(p) for p in path)}")
print(f"Total Test Reward: {test_reward}")

# 6. Plot learning curve (moving average reward)
window_size = 50
moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(8, 4.5))
plt.plot(moving_avg, color='darkorange', linewidth=2)
plt.title("Q-Learning Convergence: 50-Episode Moving Average Reward")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
