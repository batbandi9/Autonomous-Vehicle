# Examples

This directory contains example scripts demonstrating how to use the autonomous vehicle framework.

## End-to-End Learning Examples

### 1. Parking (`train_parking_e2e.py`)
Train a PPO agent to perform autonomous parking.

```bash
python examples/train_parking_e2e.py
```

**What it does**:
- Creates a parking environment
- Initializes a PPO agent
- Trains for 100,000 timesteps
- Evaluates the trained agent
- Saves the model

### 2. Lane Changing (`train_lane_change_e2e.py`)
Train a DQN agent for lane changing on highways.

```bash
python examples/train_lane_change_e2e.py
```

**What it does**:
- Uses discrete actions for lane changes
- Trains with DQN algorithm
- Handles multi-vehicle scenarios

### 3. Highway Driving (`train_highway_e2e.py`)
Train a PPO agent for general highway driving.

```bash
python examples/train_highway_e2e.py
```

**What it does**:
- Simulates dense traffic (50 vehicles)
- 4-lane highway environment
- High-speed decision making

## Hierarchical RL + MPC Example

### 4. HRL Parking (`train_parking_hrl.py`)
Train a hierarchical agent with MPC for parking.

```bash
python examples/train_parking_hrl.py
```

**What it does**:
- High-level RL for strategic decisions
- Low-level MPC for safe control
- Compares performance with/without MPC

## Multi-Agent Examples

### 5. Multi-Agent Parking (`multi_agent_parking.py`)
Demonstrate cooperative parking with multiple agents.

```bash
python examples/multi_agent_parking.py
```

**What it does**:
- Creates 2 agents in shared parking space
- Shows multi-agent observation/action spaces
- Demonstrates collision avoidance

### 6. Multi-Agent Highway (`multi_agent_highway.py`)
Demonstrate cooperative highway driving.

```bash
python examples/multi_agent_highway.py
```

**What it does**:
- 3 controlled agents on highway
- Coordination in dense traffic
- Shows agent state information

## Customization

All examples can be customized by modifying parameters:

```python
# Custom environment config
config = {
    "lanes_count": 4,
    "vehicles_count": 30,
    "duration": 50,
}
env = HighwayEnv(config=config)

# Custom agent parameters
agent = E2EAgent(
    env=env,
    learning_rate=5e-4,
    total_timesteps=200000,
)
```

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir=./logs
```

## Next Steps

After running examples:
1. Experiment with different algorithms
2. Tune hyperparameters
3. Try custom reward functions
4. Integrate multi-agent RL frameworks (e.g., RLlib)
