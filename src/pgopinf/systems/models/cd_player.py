import os
import logging

import numpy as np

from scipy.io import loadmat

from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem

logger = logging.getLogger(__name__)


def cd_player():
    """
    From MOR-Wiki: https://modelreduction.org/morwiki/CD_Player
    This benchmark models the swing arm of a CD Player holding a lens which can be moved in the horizontal plain.
    nstates 120
    ninputs 2
    noutputs 2
    nparameters 0
    components A, B, C
    """
    # load matrices
    working_dir = os.path.dirname(os.path.realpath(__file__))
    data_name_mat = f"CDplayer"
    data_name_npz = (
        f"cd_player"  # not used, but could be used to load additional data if needed
    )
    path_to_models_data_files = os.path.join(
        working_dir,
        "..",
        "..",
        "..",
        "..",
        "models_data_files",
        "cd_player",
    )

    path_to_npz = os.path.join(
        path_to_models_data_files,
        f"{data_name_npz}.npz",
    )
    if os.path.exists(path_to_npz):
        # Load data from npz file
        logger.info(f"Loading CD player data from npz file {data_name_npz}.npz")
        data_dict = dict(np.load(path_to_npz, allow_pickle=True))
        A = data_dict["A"]
        B = data_dict["B"]
        C = data_dict["C"]
    else:
        path_to_mat = os.path.join(
            path_to_models_data_files,
            f"{data_name_mat}.mat",
        )

        logger.info(f"Loading CD player data from matlab file {data_name_mat}.mat")
        data_dict = loadmat(path_to_mat)

        A = data_dict["A"]
        B = data_dict["B"]
        C = data_dict["C"]

        # convert to dense
        A = np.asarray(A.todense())

        # save to npz for faster loading next time
        np.savez_compressed(path_to_npz, A=A, B=B, C=C, allow_pickle=True)

    system = LTISystem(A=A, B=B, C=C)

    return system


if __name__ == "__main__":
    # example call
    cd_player_system = cd_player()
