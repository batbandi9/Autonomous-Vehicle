# Quick Start Guide

This guide will get you started with the Autonomous Vehicle Learning Framework in 5 minutes.

## 1. Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/batbandi9/Autonomous-Vehicle.git
cd Autonomous-Vehicle

# Install dependencies
pip install gymnasium highway-env stable-baselines3 numpy matplotlib

# Install package
pip install -e .
```

For full installation including MPC support, see [INSTALL.md](INSTALL.md).

## 2. Verify Installation (30 seconds)

```bash
python tests/test_integration.py
```

You should see:
```
✓ All basic integration tests passed!
```

## 3. Run Your First Example (2 minutes)

### Option A: E2E Learning for Parking

```bash
python examples/train_parking_e2e.py
```

This will:
- Create a parking environment
- Train a PPO agent for 100,000 steps
- Evaluate the trained agent
- Save the model to `./models/parking_e2e/`

### Option B: Multi-Agent Parking Demo

```bash
python examples/multi_agent_parking.py
```

This demonstrates cooperative parking with multiple vehicles.

## 4. Monitor Training

While training, open another terminal and run:

```bash
tensorboard --logdir=./logs
```

Then open http://localhost:6006 in your browser to see:
- Episode rewards over time
- Training loss
- Other metrics

## 5. Next Steps

### Try Different Environments

**Lane Changing:**
```bash
python examples/train_lane_change_e2e.py
```

**Highway Driving:**
```bash
python examples/train_highway_e2e.py
```

### Try HRL + MPC

For improved safety and sample efficiency:

```bash
# First install MPC dependencies
pip install cvxpy scipy

# Then run HRL example
python examples/train_parking_hrl.py
```

### Customize Your Training

Edit the example scripts to customize:

```python
# Change the algorithm
agent = E2EAgent(
    env=env,
    algorithm="SAC",  # Try: PPO, DQN, SAC, A2C, TD3
    learning_rate=3e-4,
    total_timesteps=200000,  # More training
)

# Modify environment config
config = {
    "lanes_count": 4,
    "vehicles_count": 30,
    "duration": 50,
}
env = HighwayEnv(config=config)
```

## Common Use Cases

### 1. Train an Agent

```python
from autonomous_vehicle.environments import ParkingEnv
from autonomous_vehicle.agents import E2EAgent

env = ParkingEnv()
agent = E2EAgent(env, algorithm="PPO")
agent.train()
agent.save("my_model")
```

### 2. Load and Evaluate

```python
agent = E2EAgent(env, algorithm="PPO")
agent.load("my_model")
metrics = agent.evaluate(env, n_episodes=10)
print(f"Mean reward: {metrics['mean_reward']}")
```

### 3. Use MPC for Safety

```python
from autonomous_vehicle.agents import HRLAgent

agent = HRLAgent(env, mpc_horizon=10)
agent.train()
metrics = agent.evaluate(env, use_mpc=True)
```

### 4. Multi-Agent Scenario

```python
from autonomous_vehicle.multi_agent import MultiParkingEnv

env = MultiParkingEnv(n_agents=3)
obs, info = env.reset()

for step in range(100):
    actions = {agent: env.action_space.sample() for agent in env.agents}
    obs, rewards, dones, truncated, info = env.step(actions)
```

## Troubleshooting

### "No module named 'highway_env'"

Install highway-env:
```bash
pip install highway-env
```

### "No module named 'stable_baselines3'"

Install stable-baselines3:
```bash
pip install stable-baselines3
```

### Slow training

Enable GPU acceleration:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Resources

- 📖 [Full Documentation](README.md)
- 🔧 [Installation Guide](INSTALL.md)
- 💻 [Examples](examples/README.md)
- 🧪 [Tests](tests/)

## Research Pipeline

Follow this progression for research:

1. **Week 1-2:** E2E learning baseline
   - Train agents for all three environments
   - Establish baseline performance

2. **Week 3-4:** MPC integration
   - Implement HRL + MPC
   - Compare with E2E baseline
   - Measure safety improvements

3. **Week 5-6:** Multi-agent scenarios
   - Implement cooperative policies
   - Test in multi-agent environments
   - Measure coordination efficiency

## Getting Help

If you encounter issues:
1. Check the [Installation Guide](INSTALL.md)
2. Review [Examples](examples/)
3. Open an issue on GitHub

---

**Happy Training! 🚗💨**
