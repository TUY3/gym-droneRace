import time, random
from typing import Optional, Union
import gym
from gym import spaces
import numpy as np
from gym_dogfight3d.envs.uav import UAV
from gym_dogfight3d.envs.utils import *
import math, transforms3d
import matplotlib.pyplot as plt
from matplotlib import cm

class DroneRaceEnv(gym.Env):
    """A 3D drone racing environment for OpenAI gym"""

    def __init__(self):
        super(DroneRaceEnv, self).__init__()
        self.current_step = 0
        self.uav = UAV()
        self.continuous = True
        self.dt = 0.0625
        self.ax = None
        # self.info = {"attack": 0, "be_attacked": 0, "fallInWater": 0} # fallInWater,0:don't fall, 1:uav1 fall, 2: uav2 fall
        self.goal = None
        self.end = False
        self.info = {}
        self.uav_replay = []
        self.init_distance = 0
        if self.continuous:
            # thrust, pitch, roll, yaw
            # throttle(油门), elevator(升降), aileron(副翼), rudder(方向舵)
            self.action_space = spaces.Box(-1.0, 1.0, np.array([4, ]), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(-1.0, 1.0, np.array([20, ]), dtype=np.float64)

    def _next_obs(self, uav, goal):
        obs = np.zeros(self.observation_space.shape[0])
        obs[0] = (uav.position[0] - goal[0]) / 1000
        obs[1] = (uav.position[2] - goal[2]) / 1000
        obs[2] = (uav.position[1] - goal[1]) / 1000
        obs[3] = uav.linear_speed / 300
        obs[4] = uav.linear_acceleration / 10
        obs[5] = uav.health_level
        obs[6] = uav.cap / 360
        obs[7] = uav.pitch_attitude / 90
        obs[8] = uav.roll_attitude / 90
        obs[9] = uav.thrust_level
        return obs

    def _take_action(self, uav, action):
        throttle, elevator, aileron, rudder = action
        uav.set_thrust_level(uav.thrust_level + throttle * 0.01)
        uav.set_pitch_level(elevator)
        uav.set_roll_level(aileron)
        uav.set_yaw_level(rudder)


    def get_reward(self):
        reward = 0
        target_distance = np.linalg.norm(self.uav.position - self.goal)
        if target_distance < 50:
            reward += (2000 - self.current_step) / 2000
            self.end = True
        reward += -target_distance * 1e-7
        return reward

    def reset(self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.end = False
        self.ax = None
        self.uav_replay = []
        init_thrust_level = 0.7
        init_linear_speed = 800 / 3.6
        init_pos = [1000, 500, 1000]
        init_postion = np.array([random.randint(-init_pos[0], init_pos[0]),
                                  random.randint(-init_pos[1], init_pos[1]),
                                  random.randint(-init_pos[2], init_pos[2])], dtype=np.float32)
        init_rotation = np.array([math.radians(random.randint(-180, 180)), 0, 0], dtype=np.float32)

        self.uav.reset(init_thrust_level, init_linear_speed, init_postion, init_rotation)
        self.goal = np.array([random.randint(-init_pos[0], init_pos[0]),
                              random.randint(-init_pos[1], init_pos[1]),
                              random.randint(-init_pos[2], init_pos[2])], dtype=np.float32)
        self.init_distance = np.linalg.norm(init_postion - self.goal)
        self.current_step = 0
        self.info = {"steps": 0}
        if not return_info:
            return self._next_obs(self.uav, self.goal)
        else:
            return self._next_obs(self.uav, self.goal), {}

    def step(self, action):
        self.current_step += 1
        self.info["steps"] += 1
        self._take_action(self.uav, action)
        self.uav.update_kinetics(self.dt)
        reward = self.get_reward()
        done = self.end or self.current_step >= 2000
        return self._next_obs(self.uav, self.goal), reward, done, self.info

    def _render(self):
        if self.ax is None:
            self.ax = plt.axes(projection='3d')
            self.ax.set_title('dogfight3d')
            # axes range
            self.ax.set_xlim([-3000, 3000])
            self.ax.set_ylim([-3000, 3000])
            self.ax.set_zlim([-3000, 3000])

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')

            self.ax.set_xticks(range(-3000, 3001, 1000))
            self.ax.set_zticks(range(-3000, 3001, 1000))
            self.ax.set_yticks(range(-3000, 3001, 1000))
            self.uav1_pos = [[], [], []]

            self.line1, = plt.plot([0,0,0], [0,0,0], 'r')

            plt.ion()
        self.uav1_pos[0].append(self.uav.position[0])
        self.uav1_pos[1].append(self.uav.position[1])
        self.uav1_pos[2].append(self.uav.position[2])
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }
        plt.legend(labels=[f"{self.uav.position[0]:.3f}\n{self.uav.position[1]:.3f}\n{self.uav.position[2]:.3f}"],
                   loc="upper left",
                   bbox_to_anchor=(-0.1, 0, 0, 1.1))
        # trajectory
        self.line1.set_xdata(self.uav1_pos[0])
        self.line1.set_ydata(self.uav1_pos[2])
        self.line1.set_3d_properties(self.uav1_pos[1])

        # # attack range
        # p = np.linspace(0, 2 * np.pi, 20)
        # r = np.linspace(0, 139, 20)
        # R, P = np.meshgrid(r, p)
        # x = R * np.cos(P)
        # y = R * np.sin(P)
        # z = np.sqrt(x ** 2 + y ** 2) / math.radians(10)
        # x, y, z = x.flatten(), y.flatten(), z.flatten()
        # # rotation
        # mat1 = transforms3d.euler.euler2mat(self.uav.rotation[0], self.uav.rotation[1],  self.uav.rotation[2], 'ryxz')
        # # print(self.uav1.rotation[0], self.uav1.rotation[1], self.uav1.pitch_attitude)
        # x1, z1, y1 = np.dot(mat1, np.stack([x, y, z], 0)) + self.uav.position.reshape(3, 1)
        # x1, y1, z1 = x1.reshape((20, -1)), y1.reshape((20, -1)), z1.reshape((20, -1))
        # cone1 = self.ax.plot_surface(x1, y1, z1, color="crimson")

        # uav position
        r = 50
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v)) * r + self.uav.position[0]
        y = np.outer(np.sin(u), np.sin(v)) * r + self.uav.position[2]
        z = np.outer(np.ones(np.size(u)), np.cos(v)) * r + self.uav.position[1]
        cone = self.ax.plot_surface(x, y, z, linewidth=0.0)

        # goal
        r = 100
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v)) * r + self.goal[0]
        y = np.outer(np.sin(u), np.sin(v)) * r + self.goal[2]
        z = np.outer(np.ones(np.size(u)), np.cos(v)) * r + self.goal[1]
        cone1 = self.ax.plot_surface(x, y, z, linewidth=0.0, color="crimson")

        self.ax.set_xlim([-3000, 3000])
        self.ax.set_zlim([-3000, 3000])
        self.ax.set_ylim([-3000, 3000])
        plt.pause(0.00001)
        cone1.remove()
        cone.remove()


    def render(self, mode='live'):
        assert mode in ["live", "replay"], "Invalid mode, must be either \"live\" or \"replay\""
        if mode == 'replay':
            self.uav_replay.append({"uav1_pos": self.uav.position,
                                    "uav1_rot": self.uav.rotation,})
        elif mode == 'live':
            self._render()

    def close(self):
        pass


def test():
    env = DroneRaceEnv()
    # obs = env.reset()
    obs = env.reset()
    start = time.time()
    steps = 0
    # actions = []
    # with open('action.txt', 'r') as f:
    #     for line in f:
    #         actions.append(list(map(float, line.strip().split())))
    while True:
        env.render('live')
        action = env.action_space.sample()
        # action = [action, action]
        # action = actions[steps]
        # print(action)
        steps += 1
        next_obs, r, done, info = env.step(action)
        print(r)
        # print(env.current_step)
        if done or steps > 5000:
            print(info, env.current_step)
            break
    plt.close()
    print(time.time() - start)

if __name__ == '__main__':
    test()