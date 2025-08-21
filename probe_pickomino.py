import gymnasium as gym
import pickomino_env   # wichtig: löst das register() aus

env = gym.make("Pickomino-v0")
obs, info = env.reset()
print(obs)
terminated = False
action = None
dice_sum = 0
while True:
	try:
		# chose = env.action_space.sample()
		# chose = argmax(obs[1])  # choose the first player
		chose = max(range(len(obs['dice_rolled'])), key=lambda i: obs['dice_rolled'][i])  # choose the dice with the highest value
		action = [chose, 0]
		if dice_sum>20:
			action[1] = 1
		print("Action:", action)
		obs, reward, terminated, truncated, info = env.step(action)
		dice_sum = info['self._sum']
		print(dice_sum)
		if terminated or truncated:
			print("Game ended.")
			break
		print(obs)
		print(reward)
	except Exception as e:
		print("Action was: ", action)
		print("Observation:", obs)
		print("Error:", e)
		break
