import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import differential_evolution

from trial_data import load_home_trial_data, load_trial_data


def stable_seed(base_seed, *parts):
    joined = "|".join([str(base_seed), *[str(p) for p in parts]])
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def get_dbm_fit_inputs(data_dir="behavioural_data", context="lab", timeout_policy="incorrect"):

    if context == "lab":
        d = load_trial_data(data_dir=data_dir, timeout_policy=timeout_policy)
    elif context == "home":
        d = load_home_trial_data(data_dir=data_dir, timeout_policy=timeout_policy)
    else:
        raise ValueError("context must be 'lab' or 'home'")

    models = [
        nll_rand_guess,
        nll_bias_guess,
        nll_unix,
        nll_unix,
        nll_uniy,
        nll_uniy,
        nll_glc,
        nll_glc,
        nll_gcc_eq,
        nll_gcc_eq,
        nll_gcc_eq,
        nll_gcc_eq,
    ]
    side = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3]
    k = [0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    model_names = [
        "nll_rand_guess",
        "nll_bias_guess",
        "nll_unix_0",
        "nll_unix_1",
        "nll_uniy_0",
        "nll_uniy_1",
        "nll_glc_0",
        "nll_glc_1",
        "nll_gcc_eq_0",
        "nll_gcc_eq_1",
        "nll_gcc_eq_2",
        "nll_gcc_eq_3",
    ]

    return d, models, side, k, model_names


def fit_dbm_top(
    seed=462,
    optimizer_workers=1,
    out_path="dbm_fits/dbm_results_lab_blocks.csv",
    data_dir="behavioural_data",
):

    d, models, side, k, model_names = get_dbm_fit_inputs(data_dir=data_dir)

    dbm = (
        d.groupby(["subject", "session", "phase", "block", "probe_condition"])
        .apply(fit_dbm, models, side, k, model_names, seed, optimizer_workers)
        .reset_index()
    )

    dbm.to_csv(out_path)


def fit_dbm(
    d,
    model_func,
    side,
    k,
    model_name,
    base_seed=462,
    optimizer_workers=1,
    verbose=False,
):
    fit_args = {
        "obj_func": None,
        "bounds": None,
        "disp": False,
        "maxiter": 3000,
        "popsize": 20,
        "mutation": 0.7,
        "recombination": 0.5,
        "tol": 1e-3,
        "polish": False,
        "updating": "deferred",
        "workers": optimizer_workers,
    }

    obj_func = fit_args["obj_func"]
    bounds = fit_args["bounds"]
    maxiter = fit_args["maxiter"]
    disp = fit_args["disp"]
    tol = fit_args["tol"]
    polish = fit_args["polish"]
    updating = fit_args["updating"]
    workers = fit_args["workers"]
    popsize = fit_args["popsize"]
    mutation = fit_args["mutation"]
    recombination = fit_args["recombination"]

    drec = []
    group_parts = []
    for col in ["subject", "session", "phase", "block", "probe_condition"]:
        if col in d.columns:
            group_parts.append(d.iloc[0][col])

    for m, mod in enumerate(model_func):
        dd = d[["cat", "x", "y", "resp"]]
        n = len(dd)

        cat = dd.cat.to_numpy()
        x = dd.x.to_numpy()
        y = dd.y.to_numpy()
        resp = dd.resp.to_numpy()

        # rescale x and y to be [0, 100]
        range_x = np.max(x) - np.min(x)
        x = ((x - np.min(x)) / range_x) * 100
        range_y = np.max(y) - np.min(y)
        y = ((y - np.min(y)) / range_y) * 100

        # compute glc bnds
        yub = np.max(y) + 0.1 * range_y
        ylb = np.min(y) - 0.1 * range_y
        bub = 2 * np.max([yub, -ylb])
        blb = -bub
        nlb = 0.001
        nub = np.max([range_x, range_y]) / 2

        if "unix" in model_name[m]:
            bnd = ((0, 100), (nlb, nub))
        elif "uniy" in model_name[m]:
            bnd = ((0, 100), (nlb, nub))
        elif "glc" in model_name[m]:
            bnd = ((-1, 1), (blb, bub), (nlb, nub))
        elif "gcc" in model_name[m]:
            bnd = ((0, 100), (0, 100), (nlb, nub))
        elif "rand_guess" in model_name[m]:
            bnd = ((0, 1), )  # dummy param; NLL is constant, k=0 for BIC
        elif "bias_guess" in model_name[m]:
            bnd = ((0.001, 0.999), )

        z_limit = 3

        args = (z_limit, cat, x, y, resp, side[m])

        results = differential_evolution(
            func=mod,
            bounds=bnd,
            args=args,
            seed=stable_seed(base_seed, *group_parts, model_name[m]),
            disp=disp,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            polish=polish,
            updating=updating,
            workers=workers,
        )

        tmp = np.concatenate((results["x"], [results["fun"]]))
        tmp = np.reshape(tmp, (tmp.shape[0], 1))

        if verbose:
            print(model_name[m], results["x"], results["fun"])
            if "glc" in model_name[m]:
                # a1*x + a2*y + b = 0  /  y = -(a1*x + b) / a2
                a1 = results['x'][0]
                a2 = np.sqrt(1 - a1**2)
                b = results['x'][1]
                print(a1, a2, b)
            print(np.unique(resp))

        #        fig, ax = plt.subplots(1, 1, squeeze=False)
        #        ax[0, 0].scatter(x, y, c=resp)
        #        ax[0, 0].plot([0, 100], [-b / a2, -(100 * a1 + b) / a2], '--k')
        #        ax[0, 0].set_xlim(-5, 105)
        #        ax[0, 0].set_ylim(-5, 105)
        #        plt.show()

        resx = pd.DataFrame(results["x"])
        resx.columns = ["p"]
        resx["nll"] = results["fun"]
        resx["bic"] = k[m] * np.log(n) + 2 * results["fun"]
        # resx['aic'] = k[m] * 2 + 2 * results['fun']
        resx["model"] = model_name[m]
        drec.append(resx)

    drec = pd.concat(drec)
    return drec


def nll_unix(params, *args):
    """
    - returns the negative loglikelihood of the unidimensional X bound fit
    - params format:  [bias noise] (so x=bias is boundary)
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    xc = params[0]
    noise = params[1]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    zscoresX = (x - xc) / noise
    zscoresX = np.clip(zscoresX, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscoresX, 0.0, 1.0)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscoresX, 0.0, 1.0)
        prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + sum(log_B_probs))

    return nll


def nll_uniy(params, *args):
    """
    - returns the negative loglikelihood of the unidimensional Y bound fit
    - params format:  [bias noise] (so y=bias is boundary)
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    yc = params[0]
    noise = params[1]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    zscoresY = (y - yc) / noise
    zscoresY = np.clip(zscoresY, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscoresY, 0.0, 1.0)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscoresY, 0.0, 1.0)
        prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + sum(log_B_probs))

    return nll


def nll_glc(params, *args):
    """
    - returns the negative loglikelihood of the GLC
    - params format: [a1 b noise]
    -- a1*x+a2*y+b=0 is the linear bound
    -- assumes without loss of generality that:
    --- a2=sqrt(1-a1^2)
    --- a2 >= 0
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    a1 = params[0]
    a2 = np.sqrt(1 - params[0]**2)
    b = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    z_coefs = np.array([[a1, a2, b]]).T / params[2]
    data_info = np.array([x, y, np.ones(np.shape(x))]).T
    zscores = np.dot(data_info, z_coefs)
    zscores = np.clip(zscores, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscores)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscores)
        prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + np.sum(log_B_probs))

    return nll


def nll_gcc_eq(params, *args):
    """
    returns the negative loglikelihood of the 2d data for the General
    Conjunctive Classifier with equal variance in the two dimensions.

    Parameters:
    params format: [biasX biasY noise] (so x = biasX and
    y = biasY make boundary)
    data row format:  [subject_response x y correct_response]
    z_limit is the z-score value beyond which one should truncate
    """

    xc = params[0]
    yc = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    if side == 0:
        zscoresX = (x - xc) / noise
        zscoresY = (y - yc) / noise
    elif side == 1:
        zscoresX = (xc - x) / noise
        zscoresY = (y - yc) / noise
    elif side == 2:
        zscoresX = (x - xc) / noise
        zscoresY = (yc - y) / noise
    else:
        zscoresX = (xc - x) / noise
        zscoresY = (yc - y) / noise

    zscoresX = np.clip(zscoresX, -z_limit, z_limit)
    zscoresY = np.clip(zscoresY, -z_limit, z_limit)

    pXB = norm.cdf(zscoresX)
    pYB = norm.cdf(zscoresY)

    prB = pXB * pYB
    prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + np.sum(log_B_probs))

    return nll


def nll_rand_guess(params, *args):
    """
    Random guessing model: P(B) = 0.5 for all stimuli regardless of position.
    No true free parameters; accepts one dummy param to satisfy differential_evolution.
    Set k=0 in the BIC calculation when using this model.
    """
    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    nll = n * np.log(2)

    return nll


def nll_bias_guess(params, *args):
    """
    Biased guessing model: P(B) = b for all stimuli regardless of position.
    One free parameter: b, the constant probability of responding B.
    """
    b = params[0]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    prA = np.clip(1 - b, 1e-10, 1 - 1e-10)
    prB = np.clip(b, 1e-10, 1 - 1e-10)

    nll = -(len(A_indices[0]) * np.log(prA) + len(B_indices[0]) * np.log(prB))

    return nll


def val_rand_guess(params, *args):
    """
    Generates model responses for random guessing.
    P(B) = 0.5 for all stimuli.
    """
    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    prB = np.full(x.shape[0], 0.5, dtype=float)
    resp = np.random.uniform(size=prB.shape) < prB
    resp = resp.astype(int)

    return cat, x, y, resp


def val_bias_guess(params, *args):
    """
    Generates model responses for biased guessing.
    P(B) = b for all stimuli.
    """
    b = params[0]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    prB = np.full(x.shape[0], b, dtype=float)
    prB = np.clip(prB, 1e-10, 1 - 1e-10)
    resp = np.random.uniform(size=prB.shape) < prB
    resp = resp.astype(int)

    return cat, x, y, resp


def val_unix(params, *args):
    """
    Generates model responses for unidimensional X boundary model.
    params format: [bias noise]
    """
    xc = params[0]
    noise = params[1]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    zscoresX = (x - xc) / noise
    zscoresX = np.clip(zscoresX, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscoresX, 0.0, 1.0)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscoresX, 0.0, 1.0)
        prA = 1 - prB

    prB = np.clip(prB, 1e-10, 1 - 1e-10)
    resp = np.random.uniform(size=prB.shape) < prB
    resp = resp.astype(int)

    return cat, x, y, resp


def val_uniy(params, *args):
    """
    Generates model responses for unidimensional Y boundary model.
    params format: [bias noise]
    """
    yc = params[0]
    noise = params[1]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    zscoresY = (y - yc) / noise
    zscoresY = np.clip(zscoresY, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscoresY, 0.0, 1.0)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscoresY, 0.0, 1.0)
        prA = 1 - prB

    prB = np.clip(prB, 1e-10, 1 - 1e-10)
    resp = np.random.uniform(size=prB.shape) < prB
    resp = resp.astype(int)

    return cat, x, y, resp


def val_gcc_eq(params, *args):
    """
    Generates model responses for 2d data for the General Conjunctive
    Classifier with equal variance in the two dimensions.

    Parameters:
    params format: [biasX biasY noise] (so x = biasX and
    y = biasY make boundary)
    data row format:  [subject_response x y correct_response]
    z_limit is the z-score value beyond which one should truncate
    """

    xc = params[0]
    yc = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    if side == 0:
        zscoresX = (x - xc) / noise
        zscoresY = (y - yc) / noise
    elif side == 1:
        zscoresX = (xc - x) / noise
        zscoresY = (y - yc) / noise
    elif side == 2:
        zscoresX = (x - xc) / noise
        zscoresY = (yc - y) / noise
    else:
        zscoresX = (xc - x) / noise
        zscoresY = (yc - y) / noise

    zscoresX = np.clip(zscoresX, -z_limit, z_limit)
    zscoresY = np.clip(zscoresY, -z_limit, z_limit)

    pXB = norm.cdf(zscoresX)
    pYB = norm.cdf(zscoresY)

    prB = pXB * pYB
    prA = 1 - prB

    resp = np.random.uniform(size=prB.shape) < prB
    resp = resp.astype(int)

    return cat, x, y, resp


def val_glc(params, *args):
    """
    Generates model responses for 2d data in the GLC.
    - params format: [a1 b noise]
    -- a1*x+a2*y+b=0 is the linear bound
    -- assumes without loss of generality that:
    --- a2=sqrt(1-a1^2)
    --- a2 >= 0
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    a1 = params[0]
    a2 = np.sqrt(1 - params[0]**2)
    b = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    z_coefs = np.array([[a1, a2, b]]).T / params[2]
    data_info = np.array([x, y, np.ones(np.shape(x))]).T
    zscores = np.dot(data_info, z_coefs)
    zscores = np.clip(zscores, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscores)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscores)
        prA = 1 - prB

    resp = np.random.uniform(size=prB.shape) < prB
    resp = resp.astype(int)

    return cat, x, y, resp
