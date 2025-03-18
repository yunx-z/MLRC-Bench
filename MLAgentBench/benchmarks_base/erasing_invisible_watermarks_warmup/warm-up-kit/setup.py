from setuptools import setup, find_packages

# Reading requirements from 'requirements.txt'
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="erasinginvisible",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": ["erasinginvisible=erasinginvisible.cli:cli"]
    },
    # Other metadata
)
