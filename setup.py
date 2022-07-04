from setuptools import setup, find_packages

setup(
    name='gym_droneRace',
    version='1.0.1',
    packages=find_packages(),
    install_requires=['gym',
                      'transforms3d',
                      'matplotlib',
                      ],
    description='a 3d drone racing reinforcement learning environment',
)
