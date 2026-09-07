import argparse
import os
import time

import pandas as pd

from dbm_models import fit_dbm, get_dbm_fit_inputs


GROUP_COLS = ["context", "subject", "session", "phase", "block", "probe_condition"]
OUT_COLS = GROUP_COLS + ["n_trials", "p", "nll", "bic", "model"]


def fit_groups(groups, models, side, k, model_names, seed, optimizer_workers):
    rec = []

    for _, group_df in groups:
        fit = fit_dbm(
            group_df,
            models,
            side,
            k,
            model_names,
            seed,
            optimizer_workers,
        ).reset_index(drop=True)

        meta = group_df.iloc[0][GROUP_COLS].to_dict()
        for key, value in meta.items():
            fit[key] = value
        fit["n_trials"] = len(group_df)
        rec.append(fit[OUT_COLS])

    if len(rec) == 0:
        raise ValueError("No groups to fit.")
    return rec[0] if len(rec) == 1 else pd.concat(rec, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=462)
    parser.add_argument("--optimizer-workers", type=int, default=1)
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--context", choices=["lab", "home"], default="lab")
    parser.add_argument(
        "--timeout-policy", choices=["incorrect", "exclude"], default="incorrect",
        help="Encode timeouts as incorrect (primary analysis) or exclude them (sensitivity analysis).",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if args.chunk_index is not None and (
        args.chunk_index < 0 or args.chunk_index >= args.num_chunks
    ):
        raise ValueError("chunk_index must be in [0, num_chunks)")

    data_dir = args.data_dir or ("behavioural_data" if args.context == "lab" else "at_home_data")
    sensitivity_suffix = "_timeout_excluded" if args.timeout_policy == "exclude" else ""
    out_path = args.out_path or f"dbm_fits/dbm_results_{args.context}_blocks{sensitivity_suffix}.csv"
    out_dir_chunks = args.out_dir or f"dbm_fits/dbm_results_chunks_{args.context}{sensitivity_suffix}"
    d, models, side, k, model_names = get_dbm_fit_inputs(
        data_dir=data_dir, context=args.context, timeout_policy=args.timeout_policy
    )
    groups = list(d.groupby(GROUP_COLS, sort=False))

    if args.chunk_index is None:
        selected_groups = groups
        mode = "full"
        print(f"Fitting all groups={len(groups)}")
    else:
        selected_groups = groups[args.chunk_index::args.num_chunks]
        mode = "chunk"
        print(
            f"Total groups={len(groups)}, chunk_index={args.chunk_index}, "
            f"num_chunks={args.num_chunks}, groups_in_chunk={len(selected_groups)}"
        )

    t0 = time.time()
    out = fit_groups(
        selected_groups,
        models,
        side,
        k,
        model_names,
        args.seed,
        args.optimizer_workers,
    )

    if mode == "full":
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"Wrote DBM fits to: {out_path}")
    else:
        os.makedirs(out_dir_chunks, exist_ok=True)
        out_path = os.path.join(
            out_dir_chunks,
            f"dbm_results_chunk_{args.chunk_index:04d}_of_{args.num_chunks:04d}.csv",
        )
        out.to_csv(out_path, index=False)
        print(f"Wrote DBM chunk to: {out_path}")

    elapsed = time.time() - t0
    print(f"Elapsed seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
