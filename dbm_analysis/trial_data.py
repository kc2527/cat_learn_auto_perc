"""Validated perc trial-data loaders for the DBM pipeline."""

from pathlib import Path

import pandas as pd


# Participant 303 withdrew from the study. Their raw data are retained but
# intentionally excluded from all DBM analyses.
EXCLUDED_SUBJECTS = {303}

# Authoritative downloaded-filename -> completion-order session mapping from
# code/behavioural_results.py.
HOME_SESSION_CORRECTIONS = {
    "sub_134_perc_sess_001_part_001_date_2026_08_05_data.csv": 1,
    "sub_134_perc_sess_001_part_001_date_2026_08_06_data.csv": 2,
    "sub_134_perc_sess_002_part_001_date_2026_08_07_data.csv": 3,
    "sub_134_perc_sess_003_part_001_date_2026_08_13_data.csv": 4,
    "sub_134_perc_sess_004_part_001_date_2026_08_14_data.csv": 5,
    "sub_134_perc_sess_005_part_001_date_2026_08_16_data.csv": 6,
    "sub_134_perc_sess_006_part_001_date_2026_08_19_data.csv": 7,
    "sub_134_perc_sess_007_part_001_date_2026_08_20_data.csv": 8,
    "sub_134_perc_sess_007_part_002_date_2026_08_20_data.csv": 8,
    "sub_134_perc_sess_008_part_001_date_2026_08_23_data.csv": 9,
    "sub_134_perc_sess_009_part_001_date_2026_08_25_data.csv": 10,
    "sub_134_perc_sess_010_part_001_date_2026_08_29_data.csv": 11,
    "sub_134_perc_sess_011_part_001_date_2026_08_30_data.csv": 12,
    "sub_134_perc_sess_012_part_001_date_2026_09_01_data.csv": 13,
    "sub_134_perc_sess_013_part_001_date_2026_09_02_data.csv": 14,
    "sub_134_perc_sess_014_part_001_date_2026_09_03_data.csv": 15,
    "sub_482_perc_sess_001_part_001_date_2026_08_07_data.csv": 1,
    "sub_482_perc_sess_001_part_001_date_2026_08_09_data.csv": 2,
    "sub_482_perc_sess_002_part_001_date_2026_08_10_data.csv": 3,
    "sub_482_perc_sess_002_part_001_date_2026_08_15_data.csv": 4,
    "sub_482_perc_sess_003_part_001_date_2026_08_16_data.csv": 5,
    "sub_482_perc_sess_003_part_001_date_2026_08_17_data.csv": 6,
    "sub_482_perc_sess_003_part_001_date_2026_08_19_data.csv": 7,
    "sub_482_perc_sess_003_part_001_date_2026_08_23_data.csv": 8,
    "sub_482_perc_sess_003_part_001_date_2026_08_24_data.csv": 9,
    "sub_482_perc_sess_004_part_001_date_2026_08_29_data.csv": 10,
    "sub_482_perc_sess_005_part_001_date_2026_08_30_data.csv": 11,
    "sub_482_perc_sess_006_part_001_date_2026_08_31_data.csv": 12,
    "sub_482_perc_sess_007_part_001_date_2026_09_05_data.csv": 13,
    "sub_482_perc_sess_008_part_001_date_2026_09_06_data.csv": 14,
    "sub_998_perc_sess_008_part_001_date_2026_08_31_data.csv": 8,
    "sub_998_perc_sess_009_part_001_date_2026_09_01_data.csv": 9,
    "sub_998_perc_sess_003_part_001_date_2026_09_06_data.csv": 10,
}
LAB_SESSION_CORRECTIONS = {
    "sub_875_pace_2026_s2_sess_001_part_001_date_2026_08_28_data.csv": 4,
    "sub_444_perc_sess_001_part_001_date_2026_09_03_data.csv": 4,
}
LAB_SUBJECT_CORRECTIONS = {
    "sub_444_perc_sess_001_part_001_date_2026_09_03_data.csv": 998,
}
INVALID_LAB_PROBE_SESSIONS = {(875, 4)}
REQUIRED_COLUMNS = {"subject_id", "session_num", "session_part", "trial", "phase", "cat", "resp", "x", "y", "probe_condition"}
VALUE_MAP = {"A": 0, "B": 1, "0": 0, "1": 1, 0: 0, 1: 1}


def _resolve_data_dir(data_dir):
    path = Path(data_dir)
    if path.exists():
        return path
    path = Path(__file__).resolve().parent.parent / data_dir
    if path.exists():
        return path
    raise FileNotFoundError(f"No CSV files found under {data_dir}")


def _read_file(path, session_corrections=None, subject_corrections=None):
    d = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(d.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if d["subject_id"].nunique() != 1 or d["session_num"].nunique() != 1:
        raise ValueError(f"{path} contains multiple subjects or sessions")
    d["source_file"] = path.name
    d["source_row"] = range(len(d))
    if session_corrections and path.name in session_corrections:
        d["session_num"] = session_corrections[path.name]
    if subject_corrections and path.name in subject_corrections:
        d["subject_id"] = subject_corrections[path.name]
    d["cat"] = d["cat"].replace(VALUE_MAP)
    d["resp"] = d["resp"].replace(VALUE_MAP)
    d["is_timeout"] = d["resp"].eq("Timeout")
    valid = d["cat"].isin([0, 1]) & (d["resp"].isin([0, 1]) | d["is_timeout"])
    if not valid.all():
        raise ValueError(f"{path} has {(~valid).sum()} rows with unmapped cat/resp values")
    return d


def _finish_common(d, context):
    d = d.rename(columns={"subject_id": "subject", "session_num": "session"})
    d[["subject", "session", "cat"]] = d[["subject", "session", "cat"]].astype(int)
    d["acc"] = d["cat"] == d["resp"]
    d["probe_condition"] = d["probe_condition"].astype(str)
    d["context"] = context
    return d


def _apply_timeout_policy(d, timeout_policy, group_cols):
    if timeout_policy not in {"incorrect", "exclude"}:
        raise ValueError("timeout_policy must be 'incorrect' or 'exclude'")
    n_timeout = int(d["is_timeout"].sum())
    if timeout_policy == "incorrect":
        d.loc[d["is_timeout"], "resp"] = 1 - d.loc[d["is_timeout"], "cat"]
        print(f"Encoding {n_timeout} timeout trials as incorrect responses.")
    else:
        affected = d.loc[d["is_timeout"]].groupby(group_cols).size()
        print(
            f"Sensitivity mode: excluding {n_timeout} timeout trials from "
            f"{len(affected)} DBM blocks; planned block boundaries are unchanged."
        )
        d = d.loc[~d["is_timeout"]].copy()
    d["resp"] = d["resp"].astype(int)
    d["acc"] = d["cat"] == d["resp"]
    return d


def load_trial_data(data_dir="behavioural_data", timeout_policy="incorrect"):
    """Load complete perc lab sessions as 12 train and one probe block.

    ``timeout_policy='exclude'`` is the documented future sensitivity check.
    It removes timeouts after planned block assignment, so retained trials keep
    their original 50-trial block membership.
    """
    paths = sorted(_resolve_data_dir(data_dir).rglob("*.csv"))
    moved_lab = Path(__file__).resolve().parent.parent / "at_home_data/998/sub_444_perc_sess_001_part_001_date_2026_09_03_data.csv"
    if moved_lab.exists() and moved_lab not in paths:
        paths.append(moved_lab)
    records = []
    for path in paths:
        d = _read_file(path, LAB_SESSION_CORRECTIONS, LAB_SUBJECT_CORRECTIONS)
        if int(d["subject_id"].iloc[0]) in EXCLUDED_SUBJECTS:
            print(f"Excluding participant 303 withdrawal: {path.name}")
            continue
        records.append(d)
    d = pd.concat(records, ignore_index=True)
    d["phase"] = d["phase"].replace({"test": "probe"})
    unexpected = sorted(set(d["phase"]) - {"train", "probe"})
    if unexpected:
        raise ValueError(f"Unexpected lab phase values: {unexpected}")
    d = d.sort_values(["subject_id", "session_num", "session_part", "trial", "source_file", "source_row"]).copy()
    duplicate = d.duplicated(["subject_id", "session_num", "session_part", "trial"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate lab trial identifiers detected")
    d = _finish_common(d, "lab")
    invalid_probe = (
        d[["subject", "session"]].apply(tuple, axis=1).isin(INVALID_LAB_PROBE_SESSIONS)
        & d["phase"].eq("probe")
    )
    if invalid_probe.any():
        print("Excluding 50 invalid probe trials for 875/session 4 (wrong probe category structure).")
        d = d.loc[~invalid_probe].copy()
    counts = d.groupby(["subject", "session", "phase"]).size().unstack(fill_value=0)
    for key, value in counts.iterrows():
        expected_probe = 0 if key in INVALID_LAB_PROBE_SESSIONS else 50
        if value.get("train", 0) != 600 or value.get("probe", 0) != expected_probe:
            raise ValueError(f"Incomplete lab session {key}: {value.to_dict()}")
    d["phase_trial"] = d.groupby(["subject", "session", "phase"]).cumcount()
    d["block"] = pd.NA
    train = d["phase"].eq("train")
    d.loc[train, "block"] = "train_" + (d.loc[train, "phase_trial"] // 50 + 1).astype(str).str.zfill(2)
    d.loc[d["phase"].eq("probe"), "block"] = "probe_01"
    d = _apply_timeout_policy(d, timeout_policy, ["subject", "session", "phase", "block"])
    block_sizes = d.groupby(["subject", "session", "phase", "block"]).size()
    if timeout_policy == "incorrect" and not block_sizes.eq(50).all():
        raise ValueError(f"Lab DBM blocks must contain 50 trials: {block_sizes[~block_sizes.eq(50)].to_dict()}")
    print(f"Loaded {counts.shape[0]} complete perc lab sessions ({len(d)} trials); 875/session 4 probe excluded.")
    return d.reset_index(drop=True)


def load_home_trial_data(data_dir="at_home_data", timeout_policy="incorrect"):
    """Load complete perc home sessions as eight 50-trial train blocks.

    ``timeout_policy='exclude'`` is the documented future sensitivity check.
    """
    records = []
    for path in sorted(_resolve_data_dir(data_dir).rglob("*.csv")):
        # 998's valid fourth lab session is stored in this directory.
        if path.name in LAB_SESSION_CORRECTIONS:
            continue
        d = _read_file(path, HOME_SESSION_CORRECTIONS)
        if int(d["subject_id"].iloc[0]) in EXCLUDED_SUBJECTS:
            print(f"Excluding participant 303 withdrawal: {path.name}")
            continue
        if set(d["phase"]) != {"train"}:
            raise ValueError(f"Non-training file in at-home data: {path}")
        records.append(d)
    d = pd.concat(records, ignore_index=True)
    d = d.sort_values(["subject_id", "session_num", "session_part", "trial", "source_file", "source_row"]).copy()
    duplicate = d.duplicated(["subject_id", "session_num", "session_part", "trial"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate at-home trial identifiers detected")
    d = _finish_common(d, "home")
    sizes = d.groupby(["subject", "session"]).size()
    complete, incomplete = sizes[sizes.eq(400)].index, sizes[~sizes.eq(400)]
    if not incomplete.empty:
        print(f"Excluding incomplete at-home sessions: {incomplete.to_dict()}")
    d = d.set_index(["subject", "session"]).loc[complete].reset_index()
    if not d.groupby(["subject", "session"])["probe_condition"].nunique().eq(1).all():
        raise ValueError("At-home sessions must have exactly one probe-condition assignment")
    d["phase_trial"] = d.groupby(["subject", "session"]).cumcount()
    d["block"] = "train_" + (d["phase_trial"] // 50 + 1).astype(str).str.zfill(2)
    d = _apply_timeout_policy(d, timeout_policy, ["subject", "session", "block"])
    block_sizes = d.groupby(["subject", "session", "block"]).size()
    if timeout_policy == "incorrect" and not block_sizes.eq(50).all():
        raise ValueError(f"At-home DBM blocks must contain 50 trials: {block_sizes[~block_sizes.eq(50)].to_dict()}")
    print(f"Loaded {len(complete)} complete perc at-home sessions ({len(d)} trials).")
    return d.reset_index(drop=True)
