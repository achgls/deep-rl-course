import datetime
import json
import pickle
from pathlib import Path

import gym
import imageio
import numpy as np
from gym import Env
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from tqdm.notebook import tqdm


class QLearningAgent:
    def __init__(
            self,
            env: Env,
            lr=0.1,
            gamma=0.99,
            max_timesteps=100,
            jitter_sigma=0.005,
            eps_range=None,
            eps_schedule="exponential",  # "linear"
            eps_decay_rate=0.0005,
            init="zeros",  # "normal"
            seed=None,
            init_sigma=None
    ):
        if eps_range is None:
            eps_range = [1.0, 0.005]
        self.max_eps = eps_range[0]
        self.min_eps = eps_range[1]

        assert isinstance(env.observation_space, gym.spaces.discrete.Discrete)
        assert isinstance(env.action_space, gym.spaces.discrete.Discrete)
        self.env = env
        self.nA = self.env.action_space.n
        self.nS = self.env.observation_space.n

        assert init in ("zeros", "random")
        if init == "random":
            assert init_sigma is not None

        self.lr = lr
        self.max_timesteps = max_timesteps

        self.gamma = gamma
        if jitter_sigma is None:
            self._jitter_Q_table = lambda: None
        self.jitter_sigma = jitter_sigma

        assert eps_schedule in ("exponential", "linear")
        self.eps_decay_schedule = eps_schedule
        if eps_schedule == "exponential":
            self.update_eps = self._exponential_schedule
        else:
            self.update_eps = self._linear_schedule
        self.epsilon = self.max_eps
        self.eps_decay_rate = eps_decay_rate

        self.init_mode = init
        self.init_sigma = init_sigma

        self.total_timesteps = 0
        self.training_episodes = 0
        self.Q_table = self._initialize_Q_table()
        self.checkpoints = dict()  # Q-table checkpoints

    def _exponential_schedule(self):
        self.epsilon = max(self.min_eps, self.epsilon * (1 - self.eps_decay_rate))

    def _linear_schedule(self):
        self.epsilon = max(self.min_eps, self.epsilon - self.eps_decay_rate)

    def _initialize_Q_table(self):
        q_table_shape = (self.nS, self.nA)
        if self.init_mode == "zeros":
            return np.zeros(q_table_shape, dtype=np.float32)
        elif self.init_mode == "random":
            return np.random.randn(q_table_shape) * self.init_sigma

    def _greedy_action(self, state):
        return np.argmax(self.Q_table[state])

    def _random_action(self, state):
        return np.random.randint(self.nA)

    def _epsilon_greedy_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self._random_action(state)
        else:
            action = self._greedy_action(state)
        return action

    def _update_Q_table(self, state, action, reward, new_state):
        a = self._greedy_action(new_state)
        td_target = reward + self.gamma * self.Q_table[new_state, a]

        # Perform update
        self.Q_table[state, action] += self.lr * (td_target - self.Q_table[state, action])

    def _train_one_episode(self, seed=None):
        eps_probs = np.random.uniform(0, 1, size=self.max_timesteps)
        policies = [self._greedy_action if p > self.epsilon else self._random_action for p in eps_probs]
        obs = self.env.reset(seed=seed)[0]

        action = policies[0](obs)
        new_obs, reward, end, _, _ = self.env.step(action)  # Care to
        self._update_Q_table(obs, action, reward, new_obs)
        obs = new_obs
        timesteps = 1

        while not end and timesteps < self.max_timesteps:
            action = policies[timesteps](obs)
            new_obs, reward, end, _, _ = self.env.step(action)
            self._update_Q_table(obs, action, reward, new_obs)
            obs = new_obs
            timesteps += 1

        return timesteps

    def _jitter_Q_table(self):
        self.Q_table += np.random.randn(*self.Q_table.shape) * self.jitter_sigma

    def get_Q_table(self):
        return self.Q_table

    def get_checkpoints_keys(self):
        return list(self.checkpoints.keys())

    def train(self, num_episodes, episodes_per_checkpoint=100, resume=False):
        if not resume:
            self.total_timesteps = 0
            self.Q_table = self._initialize_Q_table()
            self.checkpoints = {}
            self.training_episodes = 0

        for ep in tqdm(range(num_episodes)):
            self._jitter_Q_table()
            self.update_eps()
            self.total_timesteps += self._train_one_episode()
            self.training_episodes += 1

            if ep and ep % episodes_per_checkpoint == 0:
                # save checkpoint and print metrics
                # metrics ?
                tqdm.write("Epsilon: {:.5f}".format(self.epsilon))
                cp_name = "cp_{}".format(self.total_timesteps)
                tqdm.write(f"Saving current Q-Table as checkpoint '{cp_name}'")
                self.checkpoints[cp_name] = self.Q_table.copy()
                mean_reward, std_reward = self.evaluate_agent(n_eval_episodes=100)
                tqdm.write("Current reward = {:.2f} +- {:.2f}".format(mean_reward, std_reward))

        print("Training is over.")
        cp_name = "cp_{}".format(self.total_timesteps)
        tqdm.write(f"Saving current Q-Table as checkpoint '{cp_name}'")
        self.checkpoints[cp_name] = self.Q_table.copy()
        mean_reward, std_reward = self.evaluate_agent(n_eval_episodes=100)
        tqdm.write("Current reward = {:.2f} +- {:.2f}".format(mean_reward, std_reward))

    def inference(self, obs):
        return self._greedy_action(obs)

    def _infer_one_episode(self, seed=None):
        if seed:
            obs = self.env.reset(seed=seed)[0]
        else:
            obs = self.env.reset()[0]

        reward = 0

        action = self._greedy_action(obs)
        obs, r, end, _, _ = self.env.step(action)
        reward += r
        timesteps = 1

        while not end and timesteps < self.max_timesteps:
            action = self._greedy_action(obs)
            obs, r, end, _, _ = self.env.step(action)
            reward += r
            timesteps += 1

        return reward

    def evaluate_agent(self, n_eval_episodes, cp=None, seed=None):
        ep_rewards = np.empty(n_eval_episodes)
        if cp:
            assert isinstance(cp, str), "'cp' must be a string"
            try:
                self.Q_table = self.checkpoints[cp]
            except KeyError:
                print(f"Checkpoint {cp} not found.")
        if seed is None:
            seed = [None] * n_eval_episodes

        for ep in range(n_eval_episodes):
            ep_rewards[ep] = self._infer_one_episode(seed=seed[ep])

        mean_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)

        return mean_reward, std_reward

    def get_model_dict(self):
        model = {
            "env_id": self.env.spec.id,
            "max_steps": self.max_timesteps,
            # n_eval_episodes": n_eval_episodes,  # to be added in push_to_hub
            # "eval_seed": eval_seed,  # to be added in push_to_hub

            "learning_rate": self.lr,
            "gamma": self.gamma,

            "jitter_sigma": self.jitter_sigma,
            "max_epsilon": self.max_eps,
            "min_epsilon": self.min_eps,
            "decay_rate": self.eps_decay_rate,
            "decay_schedule": self.eps_decay_schedule,
            "init_mode": self.init_mode,
            "init_sigma": self.init_sigma,

            "n_training_episodes": self.training_episodes,
            "qtable": self.Q_table
        }
        return model

    def record_video(self, out_dir, fps, cp=None):
        """
        Generate a replay video of the agent
        :param out_dir=
        :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
        :param cp: checkpoint to use the Q-table from, latest if None
        """
        if cp:
            assert isinstance(cp, str), "'cp' must be a string"
            try:
                self.Q_table = self.checkpoints[cp]
            except KeyError:
                print(f"Checkpoint {cp} not found.")
        images = []
        done = False
        state = self.env.reset(seed=np.random.randint(0, 500))[0]
        img = self.env.render()
        images.append(img)
        while not done:
            # Take the action (index) that have the maximum expected future reward given that state
            action = self._greedy_action(state)
            state, reward, done, _, info = self.env.step(
                action)  # We directly put next_state = state for recording logic
            img = self.env.render()
            images.append(img)
        imageio.mimsave(out_dir, [np.array(img) for i, img in enumerate(images)], fps=fps)

    def find_best_cp(self, n_eval_episodes, eval_seed):
        """
        Find best saved checkpoint based on a given evaluation protocol, with specific seed

        :param n_eval_episodes: number of evaluation episodes
        :param eval_seed: seed for evaluation reproductibility
        :return: the key for the best checkpoint in the checkpoints dict
        """
        best_score = -np.inf
        best_cp = ""
        for cp in self.checkpoints:
            mean_reward, std_reward = self.evaluate_agent(
                n_eval_episodes=n_eval_episodes,
                seed=eval_seed,
                cp=cp
            )
            score = mean_reward - std_reward
            if score > best_score:
                best_score = score
                best_cp = cp

        return best_cp, best_score

    def push_to_hub(
            self,
            repo_id,
            n_eval_episodes,
            eval_seed,
            video_fps=1,
            local_repo_path="hub",
            cp=None
    ):
        """
        Evaluate, Generate a video and Upload a model to Hugging Face Hub.
        This method does the complete pipeline:
        - It evaluates the model
        - It generates the model card
        - It generates a replay video of the agent
        - It pushes everything to the Hub

        :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
        :param n_eval_episodes: number of evaluation episodes
        :param eval_seed: evaluation seed for the evaluation episodes
        :param video_fps: how many frame per seconds to record our video replay
        (with taxi-v3 and frozenlake-v1 we use 1)
        :param local_repo_path: where the local repository is
        :param cp: checkpoint key, if none, fetches best cp
        """
        if cp == "best":
            self.Q_table = self.checkpoints[self.find_best_cp(n_eval_episodes=n_eval_episodes, eval_seed=eval_seed)[0]]
        elif cp is not None:
            assert isinstance(cp, str), "'cp' must be a string: 'best' or a key corresponding to a checkpoint"
            try:
                self.Q_table = self.checkpoints[cp]
            except KeyError:
                print(f"Checkpoint {cp} not found.")

        model = self.get_model_dict()
        _, repo_name = repo_id.split("/")

        eval_env = self.env
        api = HfApi()

        # Step 1: Create the repo
        repo_url = api.create_repo(
            repo_id=repo_id,
            exist_ok=True,
        )

        # Step 2: Download files
        repo_local_path = Path(snapshot_download(repo_id=repo_id))

        # Step 3: Save the model
        if self.env.spec.kwargs.get("map_name"):
            model["map_name"] = self.env.spec.kwargs.get("map_name")
            if not self.env.spec.kwargs.get("is_slippery", ""):
                model["slippery"] = False

        # Pickle the model
        with open(repo_local_path / "q-learning.pkl", "wb") as f:
            pickle.dump(model, f)

        # Step 4: Evaluate the model and build JSON with evaluation metrics
        mean_reward, std_reward = self.evaluate_agent(
            n_eval_episodes=n_eval_episodes, seed=eval_seed
        )

        evaluate_data = {
            "env_id": model["env_id"],
            "mean_reward": mean_reward,
            "n_eval_episodes": n_eval_episodes,
            "eval_seed": eval_seed,
            "eval_datetime": datetime.datetime.now().isoformat()
        }

        # Write a JSON file called "results.json" that will contain the
        # evaluation results
        with open(repo_local_path / "results.json", "w") as outfile:
            json.dump(evaluate_data, outfile)

        # Step 5: Create the model card
        env_name = model["env_id"]
        if self.env.spec.kwargs.get("map_name"):
            env_name += "-" + self.env.spec.kwargs.get("map_name")

        if "FrozenLake" in env_name:
            if not self.env.spec.kwargs.get("is_slippery", ""):
                env_name += "-" + "no_slippery"

        metadata = dict()
        metadata["tags"] = [env_name, "q-learning", "reinforcement-learning", "custom-implementation"]

        # Add metrics
        eval_results = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env_name,
            dataset_id=env_name,
        )

        # Merges both dictionaries
        metadata = {**metadata, **eval_results}

        model_card = f"""
      # **Q-Learning** Agent playing1 **{model["env_id"]}**
      This is a trained model of a **Q-Learning** agent playing **{model["env_id"]}** .

      ## Usage

      ```python

      model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

      # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
      env = gym.make(model["env_id"])
      ```
      """

        readme_path = repo_local_path / "README.md"
        readme = ""
        print(readme_path.exists())
        if readme_path.exists():
            with readme_path.open("r", encoding="utf8") as f:
                readme = f.read()
        else:
            readme = model_card

        with readme_path.open("w", encoding="utf-8") as f:
            f.write(readme)

        # Save our metrics to Readme metadata
        metadata_save(readme_path, metadata)

        # Step 6: Record a video
        video_path = repo_local_path / "replay.mp4"
        self.record_video(video_path, video_fps)

        # Step 7. Push everything to the Hub
        api.upload_folder(
            repo_id=repo_id,
            folder_path=repo_local_path,
            path_in_repo=".",
        )


class QLearningTaxi(QLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _greedy_valid_action(self, state, action_mask):
        # Overrides greedy action for valid greedy action
        valid_actions = np.where(action_mask == 1)[0]
        sorted_values = np.argsort(self.Q_table[state])[::-1]
        return sorted_values[np.isin(sorted_values, valid_actions)][0]

    def _random_valid_action(self, state, action_mask):
        # Overrides random action for valid actions
        valid_actions = np.where(action_mask == 1)[0]
        return np.random.choice(valid_actions)

    def _epsilon_greedy_valid_action(self, state, action_mask):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self._random_valid_action(state, action_mask)
        else:
            action = self._greedy_valid_action(state, action_mask)
        return action

    def _train_one_episode(self, seed=None):
        eps_probs = np.random.uniform(0, 1, size=self.max_timesteps)
        policies = [self._greedy_valid_action if p > self.epsilon else self._random_valid_action for p in eps_probs]
        obs, info = self.env.reset(seed=seed)
        action_mask = info["action_mask"]

        action = policies[0](obs, action_mask)
        new_obs, reward, end, _, info = self.env.step(action)
        action_mask = info["action_mask"]
        self._update_Q_table(obs, action, reward, new_obs)
        obs = new_obs
        timesteps = 1

        while not end and timesteps < self.max_timesteps:
            action = policies[timesteps](obs, action_mask)
            new_obs, reward, end, _, info = self.env.step(action)
            action_mask = info["action_mask"]
            self._update_Q_table(obs, action, reward, new_obs)
            obs = new_obs
            timesteps += 1

        return timesteps

    def _infer_one_episode(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        action_mask = info["action_mask"]

        reward = 0

        action = self._greedy_valid_action(obs, action_mask)
        obs, r, end, _, info = self.env.step(action)
        action_mask = info["action_mask"]
        reward += r
        timesteps = 1

        while not end and timesteps < self.max_timesteps:
            action = self._greedy_valid_action(obs, action_mask)
            obs, r, end, _, info = self.env.step(action)
            action_mask = info["action_mask"]
            reward += r
            timesteps += 1

        return reward


