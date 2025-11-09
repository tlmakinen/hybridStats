
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import optax


from typing import Sequence, Any, Callable
Array = Any

import numpy as np
import flax.linen as nn
import matplotlib.pyplot as plt
import cloudpickle as pickle

import yaml,os,sys
#from orbax.checkpoint import CheckpointManager, checkpoint_utils
from jax import tree_util
import optax
from flax.serialization import msgpack_restore, to_bytes
import zlib
from typing import TypeVar,Dict,Mapping

TX = TypeVar("TX", bound=optax.OptState)



def save_obj(obj, name ):
    name = name.split('.pkl')[0] # get rid of pkl ext
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    



def restore_optimizer_state(opt_state: TX, restored: Mapping[str, ...]) -> TX:
    """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
    return tree_util.tree_unflatten(
        tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
    )



def save_as_msgpack(params, save_path: str, compression = None) -> None:
    msgpack_bytes: bytes = to_bytes(params)
    if compression == "GZIP":
        msgpack_bytes = zlib.compress(msgpack_bytes)
    with open(save_path, "wb+") as file:
        file.write(msgpack_bytes)


def load_from_msgpack(params, save_path: str, compression = None) -> Dict[str, Any]:
    bytes_data = file.read()
    if compression == "GZIP":
        bytes_data = zlib.decompress(bytes_data)

    params = msgpack_restore(bytes_data)
    return params


