import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial as spsp


def calculate_squared_distances(coord: np.array, diagonal_element=0.0):

    dist = spsp.distance.cdist(coord, coord, "euclidean")
    dist_sq_arr = dist * dist

    if diagonal_element != 0.0:
        np.fill_diagonal(dist_sq_arr, diagonal_element)

    return dist_sq_arr


def get_segtp_index(df_extsp, segtp_tar):

    bool_series = None
    for desc in segtp_tar.keys():
        if bool_series is None:
            bool_series = np.abs(df_extsp[desc] - segtp_tar[desc]) < 1e-14
        else:
            bool_series = bool_series & (
                np.abs(df_extsp[desc] - segtp_tar[desc]) < 1e-14
            )

    df_temp = df_extsp.loc[bool_series, :]

    if len(df_temp.index) != 1:
        index = None
        print("Did not find index. Inspect returned dataframe.")
    else:
        index = df_temp.index[0]

    return index, df_temp


class PycrsError(Exception):
    pass


if __name__ == "__main__":

    int_arr_dct = {}

    int_arr_dct["A_int"] = np.loadtxt(r"publication/test_A_int.csv", delimiter=",")
    int_arr_dct["A_mf"] = np.loadtxt(r"publication/test_A_mf.csv", delimiter=",")
    int_arr_dct["A_hb"] = np.loadtxt(r"publication/test_A_hb.csv", delimiter=",")
    int_arr_dct["tau"] = np.loadtxt(r"publication/test_tau.csv", delimiter=",")

    df_extsp = pd.read_csv(r"publication/clusters_mix.csv", index_col=0)

    X = np.loadtxt(r"publication/test_X.csv", delimiter=",")
    Gamma = np.loadtxt(r"publication/test_X.csv", delimiter=",")

    #####################################################
    segtp_tar_1 = {
        "sigma": -0.012,
        "sigma_orth": -0.01,
        "elmnt_nr": 106,
        "mol_charge": 0,
    }
    idx_1, df_segtp_1 = get_segtp_index(df_extsp, segtp_tar_1)

    segtp_tar_2 = {"sigma": 0.017, "sigma_orth": 0.014, "elmnt_nr": 8, "mol_charge": 0}
    idx_2, df_segtp_2 = get_segtp_index(df_extsp, segtp_tar_2)

    print("A_hb", idx_1, idx_2, int_arr_dct["A_hb"][idx_1, idx_2])

    ####################################################

    #####################################################
    segtp_tar_1 = {
        "sigma": -0.012,
        "sigma_orth": -0.01,
        "elmnt_nr": 106,
        "mol_charge": 0,
    }
    idx_1, df_segtp_1 = get_segtp_index(df_extsp, segtp_tar_1)

    segtp_tar_2 = {"sigma": 0.007, "sigma_orth": 0.005, "elmnt_nr": 8, "mol_charge": 0}
    idx_2, df_segtp_2 = get_segtp_index(df_extsp, segtp_tar_2)

    print("A_hb", idx_1, idx_2, int_arr_dct["A_hb"][idx_1, idx_2])

    ####################################################
