import os
import logging

import numpy as np

from scipy.io import loadmat

from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem

logger = logging.getLogger(__name__)


def earth_atmosphere():
    """
    From MOR-Wiki: https://modelreduction.org/morwiki/Earth_Atmosphere
    This benchmark models the track of an atmospheric storm track.
    nstates 598
    ninputs 1
    noutputs 1
    nparameters 0
    components A, B, C
    """
    # load matrices
    working_dir = os.path.dirname(os.path.realpath(__file__))
    data_name = f"eady"
    data_name_npz = f"earth_atmosphere"  # not used, but could be used to load additional data if needed
    path_to_models_data_files = os.path.join(
        working_dir,
        "..",
        "..",
        "..",
        "..",
        "models_data_files",
        "earth_atmosphere",
    )

    path_to_npz = os.path.join(
        path_to_models_data_files,
        f"{data_name_npz}.npz",
    )
    if os.path.exists(path_to_npz):
        # Load data from npz file
        logger.info(f"Loading earth atmosphere data from npz file {data_name_npz}.npz")
        data_dict = dict(np.load(path_to_npz, allow_pickle=True))
        A = data_dict["A"]
        B = data_dict["B"]
        C = data_dict["C"]

    else:
        logger.info(f"Loading earth atmosphere data from matlab file {data_name}.mat")
        path_to_mat = os.path.join(
            path_to_models_data_files,
            f"{data_name}.mat",
        )

        data_dict = loadmat(path_to_mat)

        A = data_dict["A"]
        B = data_dict["B"]
        C = data_dict["C"]

        # save to npz for faster loading next time
        np.savez_compressed(path_to_npz, A=A, B=B, C=C, allow_pickle=True)

    system = LTISystem(A=A, B=B, C=C)

    return system


if __name__ == "__main__":
    # example call
    earth_atmosphere_system = earth_atmosphere()
