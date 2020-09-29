import numpy as np
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv


class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv):
    def __init__(self, n_tasks=100, randomize_tasks=True):
        super(WalkerRandParamsWrappedEnv, self).__init__()
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 0.0
        reward = ((posafter - posbefore) / self.dt)
        dist = abs(reward - 1.5)
        if dist > 0.5:
            reward = 0
        else:
            reward = 0.8 - dist
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        super().reset()

    def sample_goals(self, num_goals):
        return np.random.choice(np.array(range(100)), num_goals)

    def reset(self, init_state=None, reset_args=None, **kwargs):
        self._task = self.tasks[reset_args]
        self._goal = reset_args
        self.set_task(self._task)
        return super().reset()
