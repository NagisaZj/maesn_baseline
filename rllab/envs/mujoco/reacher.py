import numpy as np


from gym.envs.mujoco.reacher import ReacherEnv as ReacherEnv_
import numpy as np
from rllab import spaces
from gym import Env
from rllab.core.serializable import Serializable
from rllab.envs.base import Step

# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py

class ReacherGoalEnv_sparse(ReacherEnv_,Serializable):
    def __init__(self, task={}, n_tasks=100, randomize_tasks=True, **kwargs):
        self.goals = self.sample_tasks(n_tasks)
        self.goal_radius = 0.09
        self._goal = [0,0,0.01]
        super(ReacherGoalEnv_sparse, self).__init__()
        self.reset_task(0)
        Serializable.__init__(self)
        self.observation_space = spaces.Box(low=self.observation_space.low,high=self.observation_space.high)
        self.action_space = spaces.Box(low=self.action_space.low,high=self.action_space.high)

    def sample_goals(self, num_goals):
        return np.random.choice(np.array(range(100)), num_goals)

    def get_all_task_idx(self):
        #print(self.action_space.low,self.action_space.high)
        print(len(self.goals))
        return range(len(self.goals))

    def step(self, action):
        #action = np.clip(action, -0.1, 0.1)
        tmp_finger = self.get_body_com("fingertip")
        vec = self.get_body_com("fingertip") - self._goal

        reward_dist = - np.linalg.norm(vec)
        #print(vec,reward_dist)
        reward_ctrl = - np.square(action).sum()
        sparse_reward = self.sparsify_rewards(reward_dist)
        reward = reward_dist + reward_ctrl
        sparse_reward = sparse_reward + reward_ctrl
        reward = sparse_reward
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = False
        env_infos = dict(finger=tmp_finger.tolist(),reward_dist=reward_dist, reward_ctrl=reward_ctrl,sparse_reward=sparse_reward,goal=self._goal)
        #print(env_infos['finger'])
        return Step(ob, reward, done)

    def reset(self, init_state=None, reset_args=None, **kwargs):
        #print(reset_args)
        #return self.reset_model()
        self._goal = self.goals[reset_args]
        return super().reset()

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        if r<-self.goal_radius:
            sparse_r = 0
        else:
            sparse_r = r + 0.2
        return sparse_r

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        radius = np.random.uniform(0.2,0.25)
        angles = np.linspace(0, np.pi, num=num_tasks)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        heights = np.ones((num_tasks,), dtype=np.float32) * 0.01
        #print(xs.shape,heights.shape)
        goals = np.stack([xs, ys,heights], axis=1)

        #goals = np.stack([goals, heights], axis=1)
        np.random.shuffle(goals)
        goals = goals.tolist()
        return goals

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        return super().reset()

    def log_diagnostics(self, paths, prefix=''):
        return