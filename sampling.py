import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union


def constant_correlation_mat(size, rho):
    mat = np.ones((size, size)) * rho
    diag = np.diag_indices(size)
    mat[diag] = 1.
    return mat


def analyze_time_series(data: pd.DataFrame, col: str):
    """
    param data:   pd.Series with datetime index
    param col:  str - the name of the column to be analyzed

    return:     pd.DataFrames: nominal (mean), cov, corr
    """

    data['date'] = data.index.date
    data['hour'] = data.index.hour

    df_pivot = data.pivot(index='date', columns='hour', values=col)
    df_pivot.dropna(inplace=True)

    mean = df_pivot.mean().values
    corr_matrix = df_pivot.corr().values
    cov_matrix = df_pivot.cov().values
    return mean, cov_matrix, corr_matrix


def construct_consumer_cov(std, corr, nominal=None):
    """
    param std:          float or array: if float std will be as percentage from nominal
    param corr:         array: correlation matrix
    param nominal:      array: optional if std should be as percentage from nominal
    return:             array: cov matrix
    """
    if isinstance(std, float) and nominal is not None:
        std = std * np.array(nominal)

    mat = np.zeros((len(std), len(std)))
    np.fill_diagonal(mat, std)
    cov = mat @ corr @ mat
    return cov


def network_cov(nominal: np.ndarray,
                std: Union[float, np.ndarray],
                temporal_corr: Union[float, np.ndarray],
                spatial_corr: Union[float, np.ndarray]):
    """

    param nominal:          array - nominal consumption rows are hour of the day, columns are consumers
    param std:              float or array - if float std is calculated as percentage from nominal
                                             if array should be with same shape as nominal, std per every consumer per
                                             every hour of the day
    param temporal_corr:    float or array - if float constant correlation between all time steps
                                             if array should be TxT shape with coefficients between time steps
    param spatial_corr:     float or array - if float than constant correlation between consumers at the same
                                             time steps only
                                             if array should be nxn shape with coefficients between consumers
    return:                 array - cov all network
    """
    t, n = nominal.shape
    mat = np.zeros((t * n, t * n))

    if isinstance(std, float):
        std = nominal * std

    if isinstance(temporal_corr, float):
        temporal_corr = constant_correlation_mat(t, temporal_corr)

    if isinstance(spatial_corr, float):
        spatial_corr = constant_correlation_mat(n, spatial_corr)

    for i in range(n):
        # construct cov per single consumer
        cov_i = construct_consumer_cov(std=std[:, i], corr=temporal_corr, nominal=nominal[:, i])
        std_i = cov_i.diagonal() ** 0.5
        mat[i * t: i * t + t, i * t: i * t + t] = cov_i
        for j in range(n):
            r = spatial_corr[i, j]
            cov_j = construct_consumer_cov(std=std[:, j], corr=temporal_corr, nominal=nominal[:, j])
            std_j = cov_j.diagonal() ** 0.5
            np.fill_diagonal(mat[i*t: i*t+t, j*t: j*t+t], np.multiply(std_i, std_j) * r)

    return mat


def multivariate_sample(nominal, cov, n):
    """
    Construct gaussian (multivariate normal) sample based on mean and cov
    :param nominal:    vector of nominal values (n_consumers * T, 1)
    :param cov:     cov matrix (n_consumers * T, n_consumers * T)
    :param n:       sample size
    :return:        sample matrix
    """
    if not is_pd(cov):
        print(f'Warning: COV matrix not positive defined')
        cov = nearest_positive_defined(cov)

    delta = np.linalg.cholesky(cov)
    z = np.random.normal(size=(nominal.shape[0] * nominal.shape[1], n))
    sample = (nominal.flatten(order='F') + (delta @ z).T).T

    split_sample = np.split(sample, nominal.shape[1])
    reshaped_sample = np.stack(split_sample, axis=1)

    return reshaped_sample


def nearest_positive_defined(mat):
    """
    source: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """
    b = (mat + mat.T) / 2
    _, s, v = np.linalg.svd(b)

    h = np.dot(v.T, np.dot(np.diag(s), v))
    mat2 = (b + h) / 2
    mat3 = (mat2 + mat2.T) / 2
    if is_pd(mat3):
        return mat3

    spacing = np.spacing(np.linalg.norm(mat))
    k = 1
    while not is_pd(mat3):
        mineig = np.min(np.real(np.linalg.eigvals(mat3)))
        mat3 += np.eye(mat.shape[0]) * (-mineig * k**2 + spacing)
        k += 1

    return mat3


def is_pd(mat):
    """
    Returns true when input is positive-definite, via Cholesky
    source: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """
    try:
        _ = np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False


if __name__ == "__main__":
    # typical corr matrix
    loads = pd.read_csv("data/loads/LoadR1RT.csv", index_col=0)
    loads.index = pd.to_datetime(loads.index)
    mean, cov, corr = analyze_time_series(loads, loads.columns[0])

    # water nominal
    from wds import WDS
    wds = WDS("data/wds_wells")

    # example for drawing random sample
    nominal = wds.demands.values
    cov = network_cov(nominal=nominal, std=0.05, temporal_corr=corr, spatial_corr=0.8)
    sample = multivariate_sample(nominal=nominal, cov=cov, n=100)

