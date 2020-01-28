from gym import error

from pmbrl.envs.mountain_car import SparseMountainCarEnv
from pmbrl.envs.cartpole import SparseCartpoleSwingupEnv

try:
    from pmbrl.envs.half_cheetah import SparseHalfCheetahEnv
    from pmbrl.envs.ant import SparseAntEnv, rate_buffer
except Exception:

    class SparseHalfCheetahEnv(object):
        pass

    class SparseAntEnv(object):
        pass

    def rate_buffer(args):
        pass
