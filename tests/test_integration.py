"""
Simple integration test to verify project structure.

This test doesn't require external packages and verifies:
- Module imports work
- Basic class structure is correct
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_package_structure():
    """Test that package structure is correct."""
    import autonomous_vehicle
    assert hasattr(autonomous_vehicle, '__version__')
    print(f"✓ Package version: {autonomous_vehicle.__version__}")


def test_module_imports():
    """Test that all modules can be imported."""
    # Test environment module structure
    from autonomous_vehicle import environments
    assert hasattr(environments, '__all__')
    print(f"✓ Environment modules: {environments.__all__}")
    
    # Test agent module structure  
    from autonomous_vehicle import agents
    assert hasattr(agents, '__all__')
    print(f"✓ Agent modules: {agents.__all__}")
    
    # Test controller module structure
    from autonomous_vehicle import controllers
    assert hasattr(controllers, '__all__')
    print(f"✓ Controller modules: {controllers.__all__}")
    
    # Test multi-agent module structure
    from autonomous_vehicle import multi_agent
    assert hasattr(multi_agent, '__all__')
    print(f"✓ Multi-agent modules: {multi_agent.__all__}")
    
    # Test utils module structure
    from autonomous_vehicle import utils
    assert hasattr(utils, '__all__')
    print(f"✓ Utility modules: {utils.__all__}")


def test_class_definitions():
    """Test that classes are defined correctly."""
    # These should not raise errors even if dependencies aren't installed
    try:
        from autonomous_vehicle.environments.parking_env import ParkingEnv
        print("✓ ParkingEnv class defined")
    except ImportError as e:
        print(f"⚠ ParkingEnv import requires: {e}")
    
    try:
        from autonomous_vehicle.agents.e2e_agent import E2EAgent
        print("✓ E2EAgent class defined")
    except ImportError as e:
        print(f"⚠ E2EAgent import requires: {e}")
    
    try:
        from autonomous_vehicle.agents.hrl_agent import HRLAgent
        print("✓ HRLAgent class defined")
    except ImportError as e:
        print(f"⚠ HRLAgent import requires: {e}")
    
    try:
        from autonomous_vehicle.controllers.mpc_controller import MPCController
        print("✓ MPCController class defined")
    except ImportError as e:
        print(f"⚠ MPCController import requires: {e}")


def test_example_scripts_exist():
    """Test that example scripts exist."""
    examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    expected_examples = [
        'train_parking_e2e.py',
        'train_lane_change_e2e.py',
        'train_highway_e2e.py',
        'train_parking_hrl.py',
        'multi_agent_parking.py',
        'multi_agent_highway.py',
    ]
    
    for example in expected_examples:
        path = os.path.join(examples_dir, example)
        assert os.path.exists(path), f"Example {example} not found"
        print(f"✓ Example script exists: {example}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Integration Tests")
    print("=" * 60)
    
    try:
        test_package_structure()
        print()
        test_module_imports()
        print()
        test_class_definitions()
        print()
        test_example_scripts_exist()
        print()
        print("=" * 60)
        print("All basic integration tests passed! ✓")
        print("=" * 60)
        print("\nNote: Full functionality requires installing dependencies:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
