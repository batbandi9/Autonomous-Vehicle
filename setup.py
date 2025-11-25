from setuptools import setup, find_packages

setup(
    name="autonomous-vehicle",
    version="0.1.0",
    description="Autonomous driving agents using Hierarchical RL and MPC",
    author="Autonomous Vehicle Team",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "highway-env>=1.8.0",
        "stable-baselines3>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "cvxpy>=1.4.0",
        "scipy>=1.11.0",
        "pettingzoo>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    python_requires=">=3.8",
)
