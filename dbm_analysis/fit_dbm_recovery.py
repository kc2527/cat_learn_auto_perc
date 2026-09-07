import argparse
import os

import numpy as np
import pandas as pd

from dbm_models import (
    fit_dbm,
    nll_bias_guess,
    nll_gcc_eq,
    nll_glc,
    nll_rand_guess,
    nll_unix,
    nll_uniy,
    val_bias_guess,
    val_gcc_eq,
    val_glc,
    val_rand_guess,
    val_unix,
    val_uniy,
    stable_seed,
)
from trial_data import load_home_trial_data, load_trial_data


GROUP_COLS = ["context", "subject", "session", "phase", "block", "probe_condition"]

MODELS = [
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
FIT_SIDE = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3]
K = [0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
MODEL_NAMES = [
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
MODEL_FAMILY = {
    "nll_rand_guess": "guessing",
    "nll_bias_guess": "guessing",
    "nll_unix": "rule-based",
    "nll_uniy": "rule-based",
    "nll_glc": "procedural",
    "nll_gcc_eq": "rule-based",
}


def model_key(model_name):
    """Remove a trailing response-direction suffix, if present."""
    parts = model_name.split("_")
    return "_".join(parts[:-1]) if parts[-1].isdigit() else model_name


def generate_responses(model_name, params, cat, x, y, z_limit=3):
    """Simulate responses from a fitted DBM using the block's stimuli."""
    resp0 = np.zeros_like(cat, dtype=int)
    if model_name == "nll_rand_guess":
        response = val_rand_guess(tuple(params), z_limit, cat, x, y, resp0, 0)[3]
    elif model_name == "nll_bias_guess":
        response = val_bias_guess(tuple(params), z_limit, cat, x, y, resp0, 0)[3]
    else:
        side = int(model_name.split("_")[-1])
        if model_name.startswith("nll_unix"):
            response = val_unix(tuple(params), z_limit, cat, x, y, resp0, side)[3]
        elif model_name.startswith("nll_uniy"):
            response = val_uniy(tuple(params), z_limit, cat, x, y, resp0, side)[3]
        elif model_name.startswith("nll_glc"):
            response = val_glc(tuple(params), z_limit, cat, x, y, resp0, side)[3]
        elif model_name.startswith("nll_gcc_eq"):
            response = val_gcc_eq(tuple(params), z_limit, cat, x, y, resp0, side)[3]
        else:
            raise ValueError(f"Unknown true model: {model_name}")
    return np.asarray(response).reshape(-1).astype(int)


def load_best_fits(fit_path, block=None, context="lab"):
    dbm = pd.read_csv(fit_path)
    if "context" not in dbm.columns:
        dbm["context"] = context
    required = set(GROUP_COLS + ["p", "nll", "bic", "model", "n_trials"])
    missing = required - set(dbm.columns)
    if missing:
        raise ValueError(f"{fit_path} is missing required columns: {sorted(missing)}")
    dbm["subject"] = dbm["subject"].astype(int)
    dbm["session"] = dbm["session"].astype(int)
    dbm["probe_condition"] = dbm["probe_condition"].astype(str)
    if block is not None:
        dbm = dbm.loc[dbm["block"] == block].copy()
    if dbm.empty:
        raise ValueError("No empirical fits remain after filtering.")

    bic_by_model = dbm.groupby(GROUP_COLS + ["model"], as_index=False)["bic"].min()
    best_idx = bic_by_model.groupby(GROUP_COLS)["bic"].idxmin()
    best = bic_by_model.loc[best_idx, GROUP_COLS + ["model"]].rename(
        columns={"model": "true_model"}
    )
    params = (
        dbm.merge(best, left_on=GROUP_COLS + ["model"],
                  right_on=GROUP_COLS + ["true_model"], how="inner")
        .groupby(GROUP_COLS + ["true_model"], as_index=False)
        .agg(true_params=("p", lambda values: tuple(values.to_numpy())))
    )
    return best.merge(params, on=GROUP_COLS + ["true_model"], how="inner")


def main():
    parser = argparse.ArgumentParser(
        description="Run DBM recovery using empirical 50-trial lab fits."
    )
    parser.add_argument("--n-reps", type=int, default=2)
    parser.add_argument(
        "--block", type=str, default=None,
        help="Optional block label, e.g. train_01 or probe_01."
    )
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=462)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--context", choices=["lab", "home"], default="lab")
    parser.add_argument(
        "--timeout-policy", choices=["incorrect", "exclude"], default="incorrect",
        help="Must match the timeout policy used for the empirical fit file.",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--fit-path", type=str, default=None
    )
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.n_reps < 1:
        raise ValueError("n_reps must be >= 1")
    if args.num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if not 0 <= args.chunk_index < args.num_chunks:
        raise ValueError("chunk_index must be in [0, num_chunks)")

    data_dir = args.data_dir or ("behavioural_data" if args.context == "lab" else "at_home_data")
    sensitivity_suffix = "_timeout_excluded" if args.timeout_policy == "exclude" else ""
    fit_path = args.fit_path or f"dbm_fits/dbm_results_{args.context}_blocks{sensitivity_suffix}.csv"
    out_dir = args.out_dir or f"dbm_fits/recovery_chunks{sensitivity_suffix}"
    d = (
        load_trial_data(data_dir=data_dir, timeout_policy=args.timeout_policy)
        if args.context == "lab"
        else load_home_trial_data(data_dir=data_dir, timeout_policy=args.timeout_policy)
    )
    if args.block is not None:
        d = d.loc[d["block"] == args.block].copy()
        if d.empty:
            raise ValueError(f"Unknown or empty block: {args.block}")

    best = load_best_fits(fit_path, block=args.block, context=args.context)
    sim_in = d.merge(best, on=GROUP_COLS, how="inner")
    if sim_in.empty:
        raise ValueError("No trial data matched the empirical best fits.")

    groups = list(sim_in.groupby(GROUP_COLS + ["true_model", "true_params"], sort=False))
    if args.max_groups is not None:
        if args.max_groups < 1:
            raise ValueError("max_groups must be >= 1 when provided")
        groups = groups[:args.max_groups]
    chunk_groups = groups[args.chunk_index::args.num_chunks]
    print(
        f"Total groups={len(groups)}, chunk_index={args.chunk_index}, "
        f"num_chunks={args.num_chunks}, groups_in_chunk={len(chunk_groups)}, "
        f"n_reps={args.n_reps}, block={args.block or 'all'}, "
        f"timeout_policy={args.timeout_policy}"
    )

    records = []
    for rep in range(args.n_reps):
        for group_key, group_df in chunk_groups:
            *meta, true_model, true_params = group_key
            metadata = dict(zip(GROUP_COLS, meta))
            true_params = np.asarray(true_params, dtype=float)

            x = group_df["x"].to_numpy(dtype=float)
            y = group_df["y"].to_numpy(dtype=float)
            cat = group_df["cat"].to_numpy(dtype=int)
            x = (x - x.min()) / (x.max() - x.min()) * 100
            y = (y - y.min()) / (y.max() - y.min()) * 100

            # val_* functions use NumPy's legacy global random generator.
            np.random.seed(stable_seed(args.seed, *meta, true_model, rep, "simulate"))
            resp = generate_responses(true_model, true_params, cat, x, y)

            simulated = pd.DataFrame({"cat": cat, "x": x, "y": y, "resp": resp})
            for column, value in metadata.items():
                simulated[column] = value
            fit_seed = stable_seed(args.seed, *meta, true_model, rep, "fit")
            fit = fit_dbm(simulated, MODELS, FIT_SIDE, K, MODEL_NAMES, fit_seed)
            recovered_model = (
                fit.groupby("model", as_index=False)["bic"].min()
                .sort_values("bic", kind="stable")
                .iloc[0]["model"]
            )
            true_key = model_key(true_model)
            recovered_key = model_key(recovered_model)
            records.append({
                **metadata,
                "n_trials": len(group_df),
                "rep": rep,
                "chunk_index": args.chunk_index,
                "num_chunks": args.num_chunks,
                "true_model": true_model,
                "recovered_model": recovered_model,
                "true_family": MODEL_FAMILY[true_key],
                "recovered_family": MODEL_FAMILY[recovered_key],
                "success_strict": int(recovered_model == true_model),
                "success_family": int(MODEL_FAMILY[true_key] == MODEL_FAMILY[recovered_key]),
            })

    result = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    block_tag = args.block or "all"
    out_path = os.path.join(
        out_dir,
        f"fit_dbm_recovery_block_{block_tag}_chunk_{args.chunk_index:04d}_of_{args.num_chunks:04d}.csv",
    )
    result.to_csv(out_path, index=False)
    print(f"Wrote chunk results to: {out_path}")


if __name__ == "__main__":
    main()
