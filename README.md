# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

```
Environment: FrozenLake-v1

The project utilizes the FrozenLake-v1 environment from Gymnasium, a classic grid-world reinforcement learning problem. In this setup, the agent navigates a 4x4 grid. The objective is to move from a starting position ('S') to a goal ('G') while avoiding frozen holes ('H'). For simplicity and to focus on the core Monte Carlo Control algorithm, the is_slippery parameter is set to False, meaning the agent's actions deterministically move it in the chosen direction without random slips. This provides a clear, discrete state and action space suitable for demonstrating value-based learning methods.
```



## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm
```

1.Initialize Environment & Q-table: Set up the FrozenLake environment, initialize the Q-table (state-action values) to zeros, and define hyperparameters like num_episodes, gamma (discount factor), alpha (learning rate), and epsilon for exploration.
2.Episode Loop: Iterate for a predefined number of num_episodes to train the agent.
3.Epsilon Decay: At the start of each episode, decay epsilon gradually to reduce exploration over time, ensuring it doesn't go below epsilon_min.
4.Generate Episode: Simulate an episode from start to terminal state using the current epsilon-greedy policy. Store each (state, action, reward) tuple encountered.
5.Calculate Returns: After an episode ends, iterate backward through the episode to calculate the discounted return (G) for each time step.
6.Update Q-values: For each first occurrence of a (state, action) pair in the episode, update its Q-value using the calculated return G and the learning rate alpha. This is the core of the Monte Carlo update.
7.Extract Optimal Policy: After all episodes, derive the optimal_policy by selecting the action with the maximum Q-value for each state (argmax(Q[state, :])).
8.Evaluate Performance: Calculate and report the average reward over the last set of episodes to gauge the policy's success.
9.Visualize Learning: Plot the moving average of episode rewards over time to visualize the learning curve and convergence of the policy
```



## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for i_episode in range(num_episodes):
    episode = generate_episode(epsilon)
    
    # Total reward for the episode
    total_reward = sum([reward for state, action, reward in episode])
    episode_rewards.append(total_reward)

    # Update Q-table
    G = 0  
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = reward + gamma * G

        first_occurrence = next(i for i, (s, a, r) in enumerate(episode) if s == state and a == action)
        if first_occurrence == t:

            Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (i_episode + 1) % 1000 == 0:
        print(f"Episode {i_episode + 1}/{num_episodes}, Epsilon: {epsilon:.4f}, Avg Reward: {np.mean(episode_rewards[-1000:]):.2f}")


```

---

## Output
### num_episodes = 20000
<img width="436" height="653" alt="Screenshot 2026-09-08 213910" src="https://github.com/user-attachments/assets/9e744b9f-c9d3-472d-adb3-e495f03d071c" />




<img width="738" height="498" alt="Screenshot 2026-08-20 085417" src="https://github.com/user-attachments/assets/40926938-7f91-483d-9c38-4ee8570cc5d2" />


---

## Result
```text
The Monte Carlo Control successfully trained an agent, achieving a high average reward of approximately 93% over the final 1000 episodes. The learning curve clearly shows rapid convergence to an optimal policy that efficiently navigates the FrozenLake environment.


```
---

## Inference
```text

The Monte Carlo Control experiment successfully trained an agent to navigate the FrozenLake environment. Over 20,000 episodes, the agent's average reward steadily increased, achieving a high success rate of approximately 91.9% in the last 1000 episodes. This demonstrates that off-policy Monte Carlo Control, using an epsilon-greedy exploration strategy, effectively learned an optimal greedy policy to guide the agent to the goal, avoiding holes, and achieving consistent positive rewards.

```





---
