import itertools

import numpy as np
import pandas as pd


def connectivity_mat(edges_data: pd.DataFrame, from_col: str = 'from', to_col: str = 'to', direction='', param=''):
    n_edges = len(edges_data)
    n_nodes = pd.concat([edges_data[from_col], edges_data[to_col]]).nunique()

    mat = np.zeros((n_nodes, n_edges))
    mat[edges_data.loc[:, from_col], np.arange(n_edges)] = -1
    mat[edges_data.loc[:, to_col], np.arange(n_edges)] = 1

    if direction == 'in':
        mat[mat == -1] = 0
    if direction == 'out':
        mat[mat == 1] = 0

    if param:
        # row-wise multiplication
        mat = mat * edges_data[param].values
    return mat


def get_mat_for_type(data: pd.DataFrame, category_data: pd.DataFrame, inverse=False):
    """
    generate a matrix that can be multiplied by nodes / edges vector to get nodes / edges of certain type
    returns a NxN matrix that is based on an eye matrix where only nodes / edges from the requested type are 1
    N is the number of nodes / edges
    inverse - to return all nodes /edges beside the input type

    data: pd.DataFrame - probably one of: wds.nodes, wds.pipes, pds.bus, pds.lines
    """
    # idx = np.where(data['type'] == element_type, 1, 0)
    # print(idx)
    idx = np.where(data.index.isin(category_data.index.to_list()), 1, 0)

    if inverse:
        # to get a matrix of all types but the input one
        idx = np.logical_not(idx)

    mat = idx * np.eye(len(data))
    return mat


def get_dt_mat(n):
    """
    generate a matrix for representing difference between following time steps
    for example, change in tank volume: dv = v2 - v1
    if dv is positive (v2 > v1) water flow from the network into the tank and vice versa
    """
    mat = np.eye(n, k=0) - np.eye(n, k=1)
    return mat


def linear_coefficients_from_two_points(p1, p2):
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - a * p1[0]
    return a, b


def normalize_mat(mat):
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat))


def get_subsets_of_size(elements, subsets_size):
    return [list(combination) for combination in itertools.combinations(elements, subsets_size)]


def get_subsets_of_max_size(elements, max_subset_size, include_empty=False):
    """
    Generate all possible combinations of elements in a list,
    with each combination's size limited to max_subset_size or less.

    Parameters:
    elements (list): A list of elements for which the combinations are to be generated.
    max_subset_size (int): The maximum size of the subset combinations to be included in the output.
    include_empty (bool): Add an empty set or not

    Returns:
    list of lists: A list where each sublist is a unique combination of elements from the input list,
                   with each sublist's size being max_subset_size or smaller, including the empty combination.
    """
    if include_empty:
        all_combinations = [[]]
    else:
        all_combinations = []
    for r in range(1, min(len(elements), max_subset_size) + 1):
        all_combinations.extend(itertools.combinations(elements, r))
    return [list(comb) for comb in all_combinations]


GRB_STATUS = {
    1: 'LOADED',
    2: 'OPTIMAL',
    3: 'INFEASIBLE',
    4: 'INF_OR_UNBD',
    5: 'UNBOUNDED',
    6: 'CUTOFF',
    7: 'ITERATION_LIMIT',
    8: 'NODE_LIMIT',
    9: 'TIME_LIMIT',
    10: 'SOLUTION_LIMIT',
    11: 'INTERRUPTED',
    12: 'NUMERIC',
    13: 'SUBOPTIMAL',
    14: 'INPROGRESS',
    15: 'USER_OBJ_LIMIT'
}
