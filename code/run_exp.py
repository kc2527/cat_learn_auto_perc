# -*- coding: utf-8 -*-
"""
Run category learning experiment using PsychoPy.
"""

from datetime import datetime, timedelta
import hashlib
import os
import sys
import numpy as np
import pandas as pd
from psychopy import visual, core
from psychopy.hardware import keyboard
from util_func_eeg import EEGPort
from util_func_pid import prompt_for_pid
from util_func_session_man import resolve_session
from util_func_stimcat import make_stim_cats
from util_func_stimcat import plot_stim_space_examples
from util_func_stimcat import stim_xy_to_sf_ori_deg
from util_func_stimcat import transform_stim

EEG_ENABLED = False
EEG_PORT_ADDRESS = '0x3FD8'
EEG_DEFAULT_PULSE_MS = 10
STUDY_TAG = "perc"

TRIG = {

    # -------------------- Experiment structure --------------------
    "EXP_START": 10,
    "ITI_ONSET": 11,
    "EXP_END": 15,

    # -------------------- Stimulus onset --------------------
    # Training trials
    "STIM_ONSET_A_TRAIN": 20,
    "STIM_ONSET_B_TRAIN": 21,

    # Probe trials
    "STIM_ONSET_A_PROBE": 22,
    "STIM_ONSET_B_PROBE": 23,

    # -------------------- Responses --------------------
    # Training trials
    "RESP_A_TRAIN": 30,
    "RESP_B_TRAIN": 31,

    # Probe trials
    # RESP_A_PROBE is now 34 because 32 does not work with other trigger cables
    "RESP_A_PROBE": 34,
    "RESP_B_PROBE": 33,

    # -------------------- Feedback --------------------
    # Training trials
    "FB_COR_TRAIN": 40,
    "FB_INC_TRAIN": 41,

    # Probe trials
    "FB_COR_PROBE": 42,
    "FB_INC_PROBE": 43,

    # Timeouts
    "FB_TIMEOUT_TRAIN": 44,
    "FB_TIMEOUT_PROBE": 45,
}

PID_DIGITS = 3
CONDITION_BY_SUBJECT = {
    "002": 90,
    "134": 180,
    "268": 90,
    "303": 180,
    "482": 90,
    "527": 180,
    "639": 90,
    "707": 180,
    "875": 90,
    "998": 180,
}

PIXELS_PER_INCH = 227 / 2
PX_PER_CM = PIXELS_PER_INCH / 2.54
SIZE_CM = 5
SIZE_PX = int(SIZE_CM * PX_PER_CM)
RESUME_WINDOW = timedelta(hours=12)
NEW_SESSION_COOLDOWN = timedelta(hours=8)
RESPONSE_LIMIT_MS = 1500
FIRST_EEG_RESPONSE_LIMIT_MS = 5000
FIRST_EEG_SLOW_TRIALS = 500
# 1. EEG session 1 is fixed at 600 for response-limit familiarisation.
# 2. Sessions 2-8 use seven balanced locations spanning the IQR:
#    [150, 200, 250, 300, 350, 400, 450]
# 3. These seven locations were shuffled once using:
# np.random.default_rng(2026).permutation([150, 200, 250, 300, 350, 400, 450])
PROBE_LOCATIONS = [600, 150, 300, 400, 200, 450, 350, 250]

# ----------------------------------------------------------------------------------


def stable_int_seed(label):
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


if __name__ == "__main__":

    # --------------------------- Experiment parameters ---------------------------
    if EEG_ENABLED:
        n_train = 600
        n_test = 50
    else:
        n_train = 400
        n_test = 0

    n_total = n_train + n_test

    # --------------------------- Display / geometry -------------------------------

    win = visual.Window(
        size=(1920, 1080),
        fullscr=True,
        units='pix',
        color=(0.494, 0.494, 0.494),
        colorSpace='rgb',
        winType='pyglet',
        useRetina=True,
        waitBlanking=True,
    )
    win.mouseVisible = False

    # --------------------------- Stim objects ------------------------------------
    fix_h = visual.Line(win,
                        start=(0, -10),
                        end=(0, 10),
                        lineColor='white',
                        lineWidth=8)
    fix_v = visual.Line(win,
                        start=(-10, 0),
                        end=(10, 0),
                        lineColor='white',
                        lineWidth=8)

    init_text = visual.TextStim(win,
                                text="Please press the space bar to begin",
                                color='white',
                                height=32)

    speed_notice_text = visual.TextStim(
        win,
        text=(
            "The response time limit will now become shorter.\n\n"
            "Please respond as accurately as you can.\n\n"
            "Press the space bar to continue"
        ),
        color='white',
        height=32)

    break_text = visual.TextStim(
        win,
        text=(
            "Another block is about to start.\n"
            "Feel free to wriggle, blink, and get comfortable before you begin.\n\n"
            "Press SPACE when you are ready."
        ),
        color='white',
        height=32)

    finished_text = visual.TextStim(
        win,
        text="You finished! Thank you for participating!",
        color='white',
        height=32)

    grating = visual.GratingStim(win,
                                 tex='sin',
                                 mask='circle',
                                 texRes=256,
                                 interpolate=True,
                                 size=(SIZE_PX, SIZE_PX),
                                 units='pix',
                                 sf=0.0,
                                 ori=0.0)

    fb_ring = visual.Circle(win,
                            radius=(SIZE_PX // 2 + 10),
                            edges=128,
                            fillColor=None,
                            lineColor='white',
                            lineWidth=10,
                            units='pix',
                            pos=(0, 0))

    too_slow_text = visual.TextStim(
        win,
        text="Too slow",
        color='black',
        height=32,
        pos=(0, SIZE_PX // 2 + 60),
    )

    # --------------------------- response and clocks -----------------------------
    kb = keyboard.Keyboard()
    default_kb = keyboard.Keyboard()

    global_clock = core.Clock()
    state_clock = core.Clock()
    stim_clock = core.Clock()

    # --------------------------- Subject handling --------------------------------
    dir_data = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(dir_data, exist_ok=True)

    subject, condition = prompt_for_pid(win, PID_DIGITS, CONDITION_BY_SUBJECT)

    # ---------------------------  Session handling -------------------------------
    session_info = resolve_session(
        dir_data,
        subject,
        n_total,
        resume_window=RESUME_WINDOW,
        new_session_cooldown=NEW_SESSION_COOLDOWN,
        study_tag=STUDY_TAG,
    )
    session_num = session_info["session_num"]
    part_num = session_info["part_num"]
    today_key = session_info["today_key"]
    f_name = session_info["f_name"]
    full_path = session_info["full_path"]
    n_done = session_info["n_done"]

    if session_num == 8:
        eeg_break_trials = (299, 499)
    else:
        eeg_break_trials = (249, 499)

    # --------------------------- Stimuli and Categories  ---------------------------
    session_seed = stable_int_seed(f"{subject}_{session_num:03d}_exp")
    schedule_rng = np.random.default_rng(session_seed)
    n_stimuli_per_category = n_total // 2
    ds, ds_90, ds_180 = make_stim_cats(
        n_stimuli_per_category,
        random_seed=session_seed,
    )

    ds_train = ds.copy()
    ds_train = ds_train.sample(
        frac=1,
        random_state=int(schedule_rng.integers(0, 2**32 - 1)),
    ).reset_index(drop=True)
    ds_train = ds_train.iloc[:n_train, :]
    ds_train["phase"] = "train"

    if condition == 90:
        ds_test = ds_90.copy()
    elif condition == 180:
        ds_test = ds_180.copy()

    ds_test = ds_test.sample(
        frac=1,
        random_state=int(schedule_rng.integers(0, 2**32 - 1)),
    ).reset_index(drop=True)
    ds_test = ds_test.iloc[:n_test, :]
    ds_test["phase"] = "test"

    if EEG_ENABLED:
        probe_location = PROBE_LOCATIONS[session_num - 1]
        ds = pd.concat([
            ds_train.iloc[:probe_location],
            ds_test,
            ds_train.iloc[probe_location:]
        ]).reset_index(drop=True)
    else:
        ds = ds_train.reset_index(drop=True)

    # NOTE: Uncomment to visualize stimulus space scatter
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(6, 6))
    # sns.scatterplot(data=ds, x='x', y='y', hue='cat', ax=ax[0, 0])
    # sns.scatterplot(data=ds, x='xt', y='yt', hue='cat', ax=ax[0, 1])
    # plt.show()

    # # NOTE: Uncomment to visualise gratings in stim space
    # x = np.array([25, 50, 75])
    # x_A = x - 10
    # x_B = x + 10
    # y_A = x + 10
    # y_B = x - 10
    # x = np.concat([x_A, x_B])
    # y = np.concat([y_A, y_B])
    # dss = pd.DataFrame({'x':x, 'y':y})
    # plot_stim_space_examples(dss, win, grating, PX_PER_CM)

    trial = n_done - 1

    # --------------------------- EEG init ----------------------------------------
    eeg = EEGPort(
        win,
        address=EEG_PORT_ADDRESS,
        enabled=EEG_ENABLED,
        default_ms=EEG_DEFAULT_PULSE_MS,
    )

    # --------------------------- State machine setup ------------------------------
    time_state = 0.0
    state_current = "state_init"
    state_entry = True

    resp_key = ""
    resp = ""
    fb = ""
    rt = -1
    trial = n_done - 1
    phase = ""
    cat = ""
    gap_ms = 0
    sf_cycles_per_pix = np.nan
    ori_deg = np.nan
    trig_stim = np.nan
    trig_resp = np.nan
    trig_fb = np.nan
    t_resp = np.nan
    response_limit_ms = RESPONSE_LIMIT_MS

    # Record keeping
    trial_data = {
        "subject_id": [],
        "session_num": [],
        "session_part": [],
        "trial": [],
        "phase": [],
        "cat": [],
        "resp_key": [],
        "resp": [],
        "fb": [],
        "rt": [],
        "ts_iso": [],
        "eeg_enabled": [],
        "trigger_stim": [],
        "trigger_resp": [],
        "trigger_fb": [],
        "t_stim": [],
        "t_resp": [],
        "t_fb": [],
        "port_address": [],
        "probe_condition": [],
        "x": [],
        "y": [],
        "xt": [],
        "yt": []
    }

    flip_times = {
        "t_stim": np.nan,
        "t_fb": np.nan,
    }

    # --------------------------- Main loop ---------------------------------------
    running = True
    while running:

        if default_kb.getKeys(keyList=['escape'], waitRelease=False):
            running = False
            break

        eeg.update(global_clock)

        # --------------------- STATE: INIT ---------------------
        if state_current == "state_init":
            if state_entry:
                state_clock.reset()
                win.color = (0.494, 0.494, 0.494)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0
            init_text.draw()

            keys = kb.getKeys(keyList=['space'], waitRelease=False, clear=True)

            if keys:
                eeg.flip_pulse(TRIG["EXP_START"], global_clock=global_clock)
                state_current = "state_iti"
                state_entry = True

            win.flip()

        # --------------------- STATE: BREAK ---------------------
        elif state_current == "state_break":
            if state_entry:
                kb.clearEvents()
                state_entry = False

            break_text.draw()

            keys = kb.getKeys(keyList=['space'], waitRelease=False, clear=True,)

            if keys:
                state_current = "state_iti"
                state_entry = True

            win.flip()

        # --------------------- STATE: SPEED NOTICE ---------------------
        elif state_current == "state_speed_notice":
            if state_entry:
                kb.clearEvents()
                state_entry = False

            speed_notice_text.draw()

            keys = kb.getKeys(keyList=['space'], waitRelease=False, clear=True,)

            if keys:
                state_current = "state_iti"
                state_entry = True

            win.flip()

        # --------------------- STATE: FINISHED ---------------------
        elif state_current == "state_finished":
            if state_entry:
                eeg.flip_pulse(TRIG["EXP_END"], global_clock=global_clock)
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0
            finished_text.draw()
            win.flip()

        # --------------------- STATE: ITI ---------------------
        elif state_current == "state_iti":
            if state_entry:
                state_clock.reset()
                eeg.flip_pulse(TRIG["ITI_ONSET"], global_clock=global_clock)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            fix_h.draw()
            fix_v.draw()

            if time_state > 1000:
                resp_key = ""
                resp = ""
                fb = ""
                rt = -1
                t_resp = np.nan
                state_clock.reset()
                trial += 1

                # when experiment finishes, calculate accuracy for that session
                # and display on screen
                if trial >= n_total:
                    if EEG_ENABLED:
                        df = pd.read_csv(full_path)
                    else:
                        session_name = (
                            f"sub_{subject}_{STUDY_TAG}_sess_"
                            f"{session_num:03d}_part_"
                        )
                        join_files = sorted(
                            os.path.join(dir_data, fn)
                            for fn in os.listdir(dir_data)
                            if fn.startswith(session_name) and fn.endswith(".csv")
                        )
                        df = pd.concat([pd.read_csv(f) for f in join_files], ignore_index=True)

                    train_df = df[df["phase"] == "train"]
                    accuracy = 100 * (train_df["fb"] == "Correct").mean()

                    saved_df = pd.read_csv(full_path)
                    saved_df["session_accuracy_pct"] = round(accuracy, 1)
                    saved_df.to_csv(full_path, index=False)

                    performance_message = ""
                    if ((session_num <= 2 and accuracy < 70)
                         or (3 <= session_num < 5 and accuracy < 80)
                         or (session_num >= 5 and accuracy < 85)):
                        performance_message = ("100% is possible on this task! Please reach out "
                                               "to the research team.")

                    elif accuracy < 90 and session_num >= 3:
                        performance_message = ("100% is possible on this task! Keep trying!")

                    elif accuracy < 95 and session_num >= 5:
                        performance_message = ("100% is possible on this task! You're getting "
                                               "closer, keep going!")

                    elif 95 <= accuracy < 99:
                        performance_message = ("You're almost there!")

                    elif 99 <= accuracy < 100:
                        performance_message = ("Wanna guess how many you got wrong...?")

                    elif accuracy == 100:
                        performance_message = ("Congratulations on being awesome!")

                    finished_text.text = ("You finished! Thank you for participating!\n\n"
                                          f"Training accuracy: {accuracy:.1f}%")

                    if performance_message:
                        finished_text.text += f"\n\n{performance_message}"

                    state_current = "state_finished"
                    state_entry = True

                else:
                    sf_cycles_per_pix, ori_deg = stim_xy_to_sf_ori_deg(
                        ds['x'].iloc[trial],
                        ds['y'].iloc[trial],
                        PX_PER_CM,
                    )
                    cat = str(ds['cat'].iloc[trial]).upper()
                    if cat not in {"A", "B"}:
                        raise ValueError(
                            f"Category labels must be 'A' or 'B'. Got: {cat}")
                    phase = ds['phase'].iloc[trial]
                    if (EEG_ENABLED and session_num == 1
                            and trial < FIRST_EEG_SLOW_TRIALS):
                        response_limit_ms = FIRST_EEG_RESPONSE_LIMIT_MS
                    else:
                        response_limit_ms = RESPONSE_LIMIT_MS
                    trig_stim = np.nan
                    trig_resp = np.nan
                    trig_fb = np.nan
                    flip_times["t_stim"] = np.nan
                    flip_times["t_fb"] = np.nan

                    grating.sf = sf_cycles_per_pix
                    grating.ori = ori_deg
                    grating.pos = (0, 0)

                    kb.clearEvents()
                    gap_ms = np.random.randint(200, 401)
                    state_current = "state_pre_stim_gap"
                    state_entry = True

            win.flip()

        # --------------------- STATE: PRE-STIM GAP ---------------------
        elif state_current == "state_pre_stim_gap":
            if state_entry:
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            fix_h.draw()
            fix_v.draw()

            if time_state >= gap_ms:
                state_current = "state_stim"
                state_entry = True

            win.flip()

        # --------------------- STATE: STIM ---------------------
        elif state_current == "state_stim":
            if state_entry:
                if phase == 'train':
                    if cat == "A":
                        trig = TRIG["STIM_ONSET_A_TRAIN"]
                    else:
                        trig = TRIG["STIM_ONSET_B_TRAIN"]
                elif phase == 'test':
                    if cat == "A":
                        trig = TRIG["STIM_ONSET_A_PROBE"]
                    else:
                        trig = TRIG["STIM_ONSET_B_PROBE"]
                else:
                    trig = np.nan

                if not np.isnan(trig):
                    eeg.flip_pulse(trig, global_clock=global_clock)
                    trig_stim = int(trig)

                state_clock.reset()
                stim_clock.reset()

                win.callOnFlip(lambda: flip_times.__setitem__("t_stim", global_clock.getTime()))
                win.callOnFlip(kb.clock.reset)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            grating.draw()

            keys = kb.getKeys(keyList=['d', 'k'], waitRelease=False)
            if keys and keys[-1].rt * 1000.0 <= response_limit_ms:
                k = keys[-1]
                resp_key = k.name
                rt = k.rt * 1000.0
                if phase == 'train':
                    if k.name == 'd':
                        resp_label = "A"
                        trig = TRIG["RESP_A_TRAIN"]
                    else:
                        resp_label = "B"
                        trig = TRIG["RESP_B_TRAIN"]
                elif phase == 'test':
                    if k.name == 'd':
                        resp_label = "A"
                        trig = TRIG["RESP_A_PROBE"]
                    else:
                        resp_label = "B"
                        trig = TRIG["RESP_B_PROBE"]
                else:
                    resp_label = "none"
                    trig = np.nan

                if not np.isnan(trig):
                    eeg.pulse_now(trig, global_clock=global_clock)
                    trig_resp = int(trig)
                t_resp = global_clock.getTime()

                if cat == resp_label:
                    fb = "Correct"
                else:
                    fb = "Incorrect"
                resp = resp_label

                state_clock.reset()
                gap_ms = np.random.randint(200, 401)
                state_current = "state_pre_feedback_gap"
                state_entry = True

            elif time_state >= response_limit_ms:
                resp_key = "Timeout"
                resp = "Timeout"
                fb = "Too slow"
                rt = -1
                t_resp = np.nan

                state_clock.reset()
                gap_ms = np.random.randint(200, 401)
                state_current = "state_pre_feedback_gap"
                state_entry = True

            win.flip()

        # --------------------- STATE: PRE-FEEDBACK GAP ---------------------
        elif state_current == "state_pre_feedback_gap":
            if state_entry:
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            grating.draw()

            if time_state >= gap_ms:
                state_current = "state_feedback"
                state_entry = True

            win.flip()

        # --------------------- STATE: FEEDBACK ---------------------
        elif state_current == "state_feedback":
            if state_entry:
                if phase == 'train':
                    if fb == "Correct":
                        fb_ring.lineColor = 'green'
                        trig = TRIG["FB_COR_TRAIN"]
                    elif fb == "Incorrect":
                        fb_ring.lineColor = 'red'
                        trig = TRIG["FB_INC_TRAIN"]
                    elif fb == "Too slow":
                        fb_ring.lineColor = 'black'
                        trig = TRIG["FB_TIMEOUT_TRAIN"]
                    else:
                        trig = np.nan
                elif phase == 'test':
                    if fb == "Correct":
                        fb_ring.lineColor = 'green'
                        trig = TRIG["FB_COR_PROBE"]
                    elif fb == "Incorrect":
                        fb_ring.lineColor = 'red'
                        trig = TRIG["FB_INC_PROBE"]
                    elif fb == "Too slow":
                        fb_ring.lineColor = 'black'
                        trig = TRIG["FB_TIMEOUT_PROBE"]
                    else:
                        trig = np.nan
                else:
                    trig = np.nan

                if not np.isnan(trig):
                    eeg.flip_pulse(trig, global_clock=global_clock)
                    trig_fb = int(trig)

                win.callOnFlip(lambda: flip_times.__setitem__("t_fb", global_clock.getTime()))
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            grating.draw()
            fb_ring.draw()
            if fb == "Too slow":
                too_slow_text.draw()

            if time_state > 1000:

                probe_condition = condition
                xt, yt = transform_stim(ds["x"].iloc[trial], ds["y"].iloc[trial])

                trial_data["subject_id"].append(subject)
                trial_data["session_num"].append(session_num)
                trial_data["session_part"].append(part_num)
                trial_data["trial"].append(trial)
                trial_data["phase"].append(phase)
                trial_data["cat"].append(cat)
                trial_data["resp_key"].append(resp_key)
                trial_data["resp"].append(resp)
                trial_data["fb"].append(fb)
                trial_data["rt"].append(rt)
                trial_data["ts_iso"].append(datetime.now().isoformat())
                trial_data["eeg_enabled"].append(int(bool(EEG_ENABLED)))
                trial_data["trigger_stim"].append(trig_stim)
                trial_data["trigger_resp"].append(trig_resp)
                trial_data["trigger_fb"].append(trig_fb)
                trial_data["t_stim"].append(flip_times["t_stim"])
                trial_data["t_resp"].append(t_resp)
                trial_data["t_fb"].append(flip_times["t_fb"])
                trial_data["port_address"].append(
                    EEG_PORT_ADDRESS if EEG_ENABLED else "")
                trial_data["probe_condition"].append(probe_condition)
                trial_data["x"].append(ds["x"].iloc[trial])
                trial_data["y"].append(ds["y"].iloc[trial])
                trial_data["xt"].append(xt)
                trial_data["yt"].append(yt)

                pd.DataFrame(trial_data).to_csv(full_path, index=False)

                if (EEG_ENABLED
                        and session_num == 1
                        and trial == FIRST_EEG_SLOW_TRIALS - 1):
                    state_current = "state_speed_notice"
                elif EEG_ENABLED and trial in eeg_break_trials:
                    state_current = "state_break"
                else:
                    state_current = "state_iti"

                state_entry = True
                rt = -1

            win.flip()

    # --------------------------- Cleanup ------------------------------------------
    eeg.close()
    win.close()
    core.quit()
    sys.exit()
