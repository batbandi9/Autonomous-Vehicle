# Autonomous Vehicle Learning Framework

A comprehensive framework for training autonomous driving agents using **Hierarchical Reinforcement Learning (HRL)** and **Model Predictive Control (MPC)**. This project combines end-to-end learning with safety-critical control for parking, lane changing, highway driving, and multi-agent scenarios.

## 🎯 Features

- **End-to-End (E2E) Reinforcement Learning**: Train agents using state-of-the-art RL algorithms (PPO, DQN, SAC, A2C, TD3)
- **Hierarchical RL + MPC Integration**: Combine high-level RL decision-making with low-level MPC control for safety and stability
- **Multiple Environments**:
  - Parking environments
  - Lane changing scenarios
  - Highway driving
  - Multi-agent coordination
- **Built on Industry Standards**:
  - OpenAI Gymnasium for environments
  - highway-env for realistic driving scenarios
  - Stable Baselines3 for RL algorithms
  - CVXPY for MPC optimization

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Environments](#environments)
- [Agents](#agents)
- [Examples](#examples)
- [Training](#training)
- [Research Motivation](#research-motivation)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
git clone https://github.com/batbandi9/Autonomous-Vehicle.git
cd Autonomous-Vehicle
pip install -r requirements.txt
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## 🏃 Quick Start

### 1. E2E Learning - Parking

Train an agent for autonomous parking:

```python
from autonomous_vehicle.environments import ParkingEnv
from autonomous_vehicle.agents import E2EAgent

# Create environment
env = ParkingEnv()

# Create and train agent
agent = E2EAgent(env, algorithm="PPO", total_timesteps=100000)
agent.train()
agent.save("./models/parking_agent")
```

Run the example:
```bash
python examples/train_parking_e2e.py
```

### 2. HRL + MPC Integration

Combine hierarchical RL with MPC for improved safety:

```python
from autonomous_vehicle.environments import ParkingEnv
from autonomous_vehicle.agents import HRLAgent

# Create HRL agent with MPC
env = ParkingEnv()
agent = HRLAgent(env, algorithm="PPO", mpc_horizon=10)
agent.train()

# Evaluate with MPC enabled
metrics = agent.evaluate(env, use_mpc=True)
```

Run the example:
```bash
python examples/train_parking_hrl.py
```

### 3. Multi-Agent Scenarios

Train multiple agents to cooperate:

```python
from autonomous_vehicle.multi_agent import MultiParkingEnv

# Create multi-agent environment
env = MultiParkingEnv(n_agents=2)
observations, info = env.reset()

# Step with multiple agents
actions = {agent: env.action_space.sample() for agent in env.agents}
observations, rewards, dones, truncateds, info = env.step(actions)
```

Run the example:
```bash
python examples/multi_agent_parking.py
```

## 🏗️ Architecture

The framework follows a modular architecture:

```
autonomous_vehicle/
├── environments/       # Gymnasium-based environments
│   ├── parking_env.py
│   ├── lane_change_env.py
│   └── highway_env.py
├── agents/            # RL agents
│   ├── e2e_agent.py   # End-to-end learning
│   └── hrl_agent.py   # Hierarchical RL + MPC
├── controllers/       # Low-level controllers
│   └── mpc_controller.py
├── multi_agent/       # Multi-agent environments
│   ├── multi_parking_env.py
│   └── multi_highway_env.py
└── utils/             # Utilities and visualization
    ├── visualization.py
    └── callbacks.py
```

## 🌍 Environments

### Parking Environment

Simulates a parking lot where the agent must navigate and park in a designated spot.

**Features**:
- Continuous action space (steering, acceleration)
- Kinematic goal observations
- Collision detection
- Success/failure rewards

### Lane Changing Environment

Highway scenario focusing on safe lane changes.

**Features**:
- Discrete or continuous actions
- Multi-vehicle interaction
- Speed and safety rewards
- Traffic simulation

### Highway Driving Environment

General highway driving with multiple vehicles.

**Features**:
- 4-lane highway
- Dense traffic (up to 50 vehicles)
- High-speed decision making
- Right-lane preference reward

### Multi-Agent Environments

Cooperative scenarios with multiple controlled vehicles.

**Features**:
- Shared or individual rewards
- Collision avoidance between agents
- Coordinated parking or highway driving

## 🤖 Agents

### E2E Agent

End-to-end reinforcement learning agent supporting multiple algorithms:

- **PPO** (Proximal Policy Optimization) - Best for continuous control
- **DQN** (Deep Q-Network) - For discrete actions
- **SAC** (Soft Actor-Critic) - Advanced continuous control
- **A2C** (Advantage Actor-Critic) - Fast training
- **TD3** (Twin Delayed DDPG) - Robust continuous control

### HRL Agent

Hierarchical agent combining:
- **High-level RL**: Strategic decision making (lane choice, speed targets)
- **Low-level MPC**: Safe trajectory execution and control stability

**Benefits**:
- Improved safety through MPC constraints
- Sample efficiency from hierarchical structure
- Better generalization to new scenarios

## 📚 Examples

All examples are in the `examples/` directory:

1. **E2E Learning**:
   - `train_parking_e2e.py` - Parking with PPO
   - `train_lane_change_e2e.py` - Lane changing with DQN
   - `train_highway_e2e.py` - Highway driving with PPO

2. **HRL + MPC**:
   - `train_parking_hrl.py` - Hierarchical parking

3. **Multi-Agent**:
   - `multi_agent_parking.py` - Cooperative parking
   - `multi_agent_highway.py` - Cooperative highway driving

## 🎓 Training

### Basic Training

```python
from autonomous_vehicle.environments import HighwayEnv
from autonomous_vehicle.agents import E2EAgent

env = HighwayEnv()
agent = E2EAgent(
    env=env,
    algorithm="PPO",
    learning_rate=3e-4,
    total_timesteps=200000,
    log_dir="./logs",
    model_dir="./models"
)

agent.train(
    eval_freq=10000,
    checkpoint_freq=50000,
    n_eval_episodes=5
)
```

### Advanced Training with Custom Config

```python
# Custom environment configuration
config = {
    "lanes_count": 4,
    "vehicles_count": 30,
    "duration": 50,
    "collision_reward": -1,
    "high_speed_reward": 0.5,
}

env = HighwayEnv(config=config)

# Custom agent parameters
agent = E2EAgent(
    env=env,
    algorithm="PPO",
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
)
```

### Monitor Training

Training logs are saved for TensorBoard visualization:

```bash
tensorboard --logdir=./logs
```

## 🔬 Research Motivation

This framework addresses key challenges in autonomous driving:

1. **Sample Efficiency**: HRL reduces the sample complexity by decomposing the problem
2. **Safety**: MPC provides formal safety guarantees through constraints
3. **Generalization**: Hierarchical structure improves transfer to new scenarios
4. **Multi-Agent Coordination**: Supports cooperative driving scenarios

**Research Pipeline**:
```
Step 1: E2E Learning (baseline) → 
Step 2: MPC Integration (safety) → 
Step 3: Multi-Agent Scenarios (coordination)
```

## 📊 Evaluation

Evaluate trained agents:

```python
# Load and evaluate
agent = E2EAgent(env, algorithm="PPO")
agent.load("./models/my_model")

metrics = agent.evaluate(
    env,
    n_episodes=10,
    deterministic=True,
    render=False
)

print(f"Mean Reward: {metrics['mean_reward']:.2f}")
print(f"Mean Episode Length: {metrics['mean_length']:.2f}")
```

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black autonomous_vehicle/
flake8 autonomous_vehicle/
```

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{autonomous_vehicle_2024,
  title={Autonomous Vehicle Learning Framework},
  author={Autonomous Vehicle Team},
  year={2024},
  url={https://github.com/batbandi9/Autonomous-Vehicle}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [highway-env](https://github.com/Farama-Foundation/HighwayEnv) for realistic driving environments
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [OpenAI Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for the RL interface

## 📞 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Autonomous Driving! 🚗💨**
