# vec_env.py
import multiprocessing as mp
import sys
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.env import MarineEnv


class CloudpickleWrapper(object):
    """Wrap objects so they can be serialized with cloudpickle."""

    def __init__(self, x):
        """Initialize the instance."""
        self.x = x

    def __getstate__(self):
        """Run   getstate  ."""
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        """Run   setstate  ."""
        import pickle
        self.x = pickle.loads(ob)


class VecEnvCmd(Enum):
    """Define commands used by the vectorized environment workers."""
    STEP = 1
    RESET = 2
    CLOSE = 3
    GET_ATTR = 4


def worker(remote, parent_remote, env_fn_wrapper):
    """Run a vectorized environment worker loop."""
    parent_remote.close()


    env = env_fn_wrapper.x()

    # env: 'MarineEnv'

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == VecEnvCmd.STEP:


                p_obs, reward, done, info_list = env.step(data)


                e_obs, _ = env.get_evaders_observation()


                active_pursuers_count = len([p for p in env.pursuers if not p.deactivated])
                all_captured = env.check_all_evader_is_captured()


                for i, info_item in enumerate(info_list):

                    info_item['active_pursuers_count'] = active_pursuers_count
                    info_item['all_captured'] = all_captured


                    if i < len(env.pursuers):
                        p = env.pursuers[i]
                        info_item['x'] = p.x
                        info_item['y'] = p.y
                        info_item['deactivated'] = p.deactivated
                        info_item['collision'] = p.collision


                remote.send(((p_obs, e_obs), reward, done, info_list))

            elif cmd == VecEnvCmd.RESET:

                options = data
                p_obs, _, _ = env.reset(options=options)
                e_obs, _ = env.get_evaders_observation()


                remote.send((p_obs, e_obs))

            elif cmd == VecEnvCmd.CLOSE:
                remote.close()
                break

            elif cmd == VecEnvCmd.GET_ATTR:

                attr_name = data
                remote.send(getattr(env, attr_name))

    except Exception as e:
        import traceback
        print(f"❌ Worker process failed with error: {e}", file=sys.stderr)
        traceback.print_exc()
        remote.close()


class SubprocVecEnv:
    """Manage multiple environments in subprocesses."""
    def __init__(self, env_fns):
        """
        env_fns: list of functions that return env instances
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)


        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])


        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        """Run step async."""
        for remote, action in zip(self.remotes, actions):
            remote.send((VecEnvCmd.STEP, action))
        self.waiting = True

    def step_wait(self):
        """Run step wait."""
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def reset(self, options=None):
        """Run reset."""
        for remote in self.remotes:
            remote.send((VecEnvCmd.RESET, options))
        return [remote.recv() for remote in self.remotes]

    def reset_indices(self, indices, options=None):
        """Run reset indices."""

        for i in indices:
            self.remotes[i].send((VecEnvCmd.RESET, options))


        results = []
        for i in indices:
            results.append(self.remotes[i].recv())
        return results

    def get_attr(self, attr_name):
        """Run get attr."""
        self.remotes[0].send((VecEnvCmd.GET_ATTR, attr_name))
        return self.remotes[0].recv()

    def close(self):
        """Close worker processes and communication pipes."""
        if self.closed: return
        if self.waiting:
            for remote in self.remotes: remote.recv()
        for remote in self.remotes:
            remote.send((VecEnvCmd.CLOSE, None))
        for p in self.ps: p.join()
        self.closed = True