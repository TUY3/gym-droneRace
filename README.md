# DroneRace: a 3d drone racing reinforcement learning environment

DroneRace is a reinforcement learning environment for training fixed-wing aircraft control task, it has approximate 
real dynamics.

## Dependencies
* gym, numpy, transform3d
* python3.8
* linux or win

## Installation
gym-droneRace is pip installable using its GitHub:

```
pip install git+https://github.com/TUY3/gym-droneRace
```
or
```angular2html
git clone https://github.com/TUY3/gym-droneRace
cd gym-droneRace
pip install -e .
```

## Environment
### observation space
    relative position(x,y,z)
    linear speed
    linear acceleration
    health level
    cap(yaw),pitch,roll
    thrust_level
### action space
continuous action space, including throttle,elevator,aileron,rudder

## Example
```
env = gym.make('DroneRace-v0')
obs_space = env.observation_space
action_space = env.action_space
init_obs = env.reset()
while True:
    action = env.action_space.sample()
    next_obs, r, done, info = env.step(action)
    if done:
        print(info, env.current_step)
        break
```

### Reference
* harfang3d/dogfight-sandbox-hg1

