from gym.envs.registration import register


register(
		id='Drone-Race-v0',
		entry_point='gym_droneRace.envs:DroneRaceEnv',
)
