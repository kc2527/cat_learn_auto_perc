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
# importing funcs from util_func files
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
    "RESP_A_PROBE": 32,
    "RESP_B_PROBE": 33,

    # -------------------- Feedback --------------------
    # Training trials
    "FB_COR_TRAIN": 40,
    "FB_INC_TRAIN": 41,

    # Probe trials
    "FB_COR_PROBE": 42,
    "FB_INC_PROBE": 43,
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

# ----------------------------------------------------------------------------------

# creates a seed for each participants session so that the stimuli and
# presentation order are preseved for reproducibility
# called later for session_seed = stable_int_seed(sub + session_num)
def stable_int_seed(label):
    
    # hashlib.sha256 = hash function (256-bit fingerprint = 32 bytes)
    # .encode = converts string into raw bytes
    # .digest = gets hash result as raw bytes
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()

    # digest[:8] = first 8 bytes from 32-byte hash
    # int.from_bytes = inteprets 8 bytes as one integer
    return int.from_bytes(digest[:8], "big") % (2**32)


if __name__ == "__main__":

    # --------------------------- Experiment parameters ---------------------------
    if EEG_ENABLED:
        n_train = 550
        n_test = 100
    else:
        n_train = 300
        n_test = 0

    n_total = n_train + n_test

    # --------------------------- Display / geometry -------------------------------

    # size = window res in pixels
    # fullscr = True = run in fullscreen mode
    # units = pix = default coordinate units are pixels
    # color = background colour values
    # colorSpace = 'rgb' = intergrets color as RGB space
    # winType = 'pyglet' = backend used to create/manage OpenGL window
    # useRetina = True = use retina pixel density on retinahidpi displays
        # maybe this is why stimuli show up differently on mac/windows
    # waitBlanking = improves timing stabiility
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
    
    # PsychoPy assumes centre of window is (0,0)
    # left/down is negative, right/up is positive
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

    # --------------------------- response and clocks -----------------------------
    # initliases keyboard objects
    kb = keyboard.Keyboard()
    default_kb = keyboard.Keyboard()

    # can use getTime() to return the elapsed seconds since that clock started
    global_clock = core.Clock()
    state_clock = core.Clock()
    stim_clock = core.Clock()

    # --------------------------- Subject handling --------------------------------

    # __file__ = path of current script file
    # os.path.dirname(__file__) = folder containing this script
    # os.path.join = go up one folder then into data folder
    # os.path.abspath() = convert into full absolute path
    # == dir_data becomes data directory path
    dir_data = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data"))
        
        # create data directory if it doesn't exist
        # exist_ok = True = no error if it already exists
    os.makedirs(dir_data, exist_ok=True)

    # calls prompt_for_pid function from util_func_pid 
    # onscreen prompt appears, enforces PID digits and validates ID exist by
    # condition_by_subject
    subject, condition = prompt_for_pid(win, PID_DIGITS, CONDITION_BY_SUBJECT)

    # ---------------------------  session handling -------------------------------

    # calls resolve_session function from util_func_session_man
    # scans dir_data for subjects existing .csv file
    # groups files by session and checks how many trials are done
    # if there is a recent incomplete session within resume_window, it resumes
    # that session and increments part_num
    # builds next output filename/path and returns metadata
    session_info = resolve_session(
        dir_data,
        subject,
        n_total,
        resume_window=RESUME_WINDOW,
        new_session_cooldown=NEW_SESSION_COOLDOWN,
    )

    # information is being taken from the previous function (session_info)
    # this extracts info from each field

    # added into saved trial rows
    session_num = session_info["session_num"]
    
    # added into saved trial rows 
    part_num = session_info["part_num"]

    # returned during previous function 
    # created from current date and used in filename creation (w/in resolve_session)
    today_key = session_info["today_key"]

    # built in resolve_session and then used to create full_path 
    # (os.path.join(dir_data, f_name)
    f_name = session_info["f_name"]
    
    # used when writing .csv on every trial
    full_path = session_info["full_path"]

    # used to resume trial index instead of starting from 0
    n_done = session_info["n_done"]

    # --------------------------- Stimuli and Categories  ---------------------------

    # {sesion_num:03d} = ptython f-string number formatting
    # d = format as decimal integer, 3 = minimum width 3 characters
    # 0 = pad with leading 0's if needed
    session_seed = stable_int_seed(f"{subject}_{session_num:03d}_exp")

    # random number machine to shuffle data presentation order
    # uses session seed so that this is preproducible for the same participant number + seed
    schedule_rng = np.random.default_rng(session_seed)

    n_stimuli_per_category = n_total // 2
    ds, ds_90, ds_180 = make_stim_cats(
        n_stimuli_per_category,

        # session seed used here too
        random_seed=session_seed,
    )

    ds_train = ds.copy()
    
    # sample(frac=1) = shuffle all rows
    # random_state = gives pandas seed for that shuffle (reproducible)
    # schedule_rng.integers = generates one integer seed
    # int() = converts numpy integer type to a plain python int
    # reset_index(drop=True) = renumbers rows to 0,1,2 after shuffling and discards old vals
    ds_train = ds_train.sample(
        frac=1,
        random_state=int(schedule_rng.integers(0, 2**32 - 1)),
    ).reset_index(drop=True)

    # keep only the first n_train rows after shuffling with [, :] meaning all columns
    ds_train = ds_train.iloc[:n_train, :]

    # add/set a column named phase with value ["train"] for every remaining row
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

    ds = pd.concat([ds_train, ds_test]).reset_index(drop=True)

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

    # initialise variables to prevent 'referenced before assignment' errors + some
    resp_key = ""
    resp = ""
    fb = ""

    # placeholder before update
    rt = -1

    # if n_done = 0 then it will be -1, when trial is updated (trial+=1) it brings it 
    # to 0 for the first trial. makes life simpler 
    trial = n_done - 1
    phase = ""
    cat = ""
    
    # initialises jitter
    gap_ms = 0

    # initialised and then also reset at each trial (except for cycles and ori)
    # this is so that if an event does not occur, it is marked as nan instead of 0
    sf_cycles_per_pix = np.nan
    ori_deg = np.nan
    trig_stim = np.nan
    trig_resp = np.nan
    trig_fb = np.nan
    t_resp = np.nan

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

    # psychopy draw() cues visuals, display update comes at win.flip()
    # so to get accurate timing of stimulus presentation, log times at flip
    # this gets filled using the win.callOnFlip function later
    flip_times = {
        "t_stim": np.nan,
        "t_fb": np.nan,
    }

    # --------------------------- Main loop ---------------------------------------
    running = True
    while running:

        # keyList = only watch these keys
        # waitRelease = False = register on key-down
        # clear = True/False = whether to clear buffer for returned keys
        # i.e. if clear = False, key press is stored, if True, key press is
        # immediately cleared
        if default_kb.getKeys(keyList=['escape'], waitRelease=False):
            running = False
            break

        # this is used to clear the port after each trigger (sent by eeg.flip_pulse or
        # eeg.pulse_now when a stimulus appears or response is recorded)
        # update() is defined in util_func_eeg.py and runs every frame
        # update(global_clock) compares global_clock.getTime() to self._clear_at
        # and when global_clock.getTime() >= self._clear_at, it sends 0 to the port
        # to clear it (and prevent a trigger from staying on too long)

        eeg.update(global_clock)

        # --------------------- STATE: INIT ---------------------
        if state_current == "state_init":
            if state_entry:
                state_clock.reset()
                win.color = (0.494, 0.494, 0.494)
                state_entry = False

            time_state = state_clock.getTime() * 1000.0
            init_text.draw()

            # when participant presses a key (e.g., space bar) psychopy adds this event
            # to the queue. getKeys() then checks this queue and returns matching events
            # with clear = True, returned events are removed from queue
            # with clear = False, events may remain and be seen again later
            keys = kb.getKeys(keyList=['space'], waitRelease=False, clear=True)

            # if keys = True if at least one space bar press was detected
            # eeg.flip_pulse() schedules exp_start trigger to be sent on next flip and 
            # sets an internal future clear time based on global_clock (all calculated
            # within the flip_pulse function from util_func_eeg)
            # state_entry = True = tells next state to run its "entry" code block 
            # (i.e., the first little if statement which usually resets the state clock 
            # counter, sends the trigger, then sets entry to False)
            if keys:
                eeg.flip_pulse(TRIG["EXP_START"], global_clock=global_clock)
                state_current = "state_iti"
                state_entry = True

            # called once per loop iteration (one frame), so this can be called
            # again many times within one state while that state is active
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

            # clear everything from previous trial for clean slate for next trial
            if time_state > 1000:
                # reset to empy at trial start
                resp_key = ""
                resp = ""
                fb = ""
                rt = -1
                t_resp = np.nan

                # starts timing for the next state transition  
                # state transition occurs after this if statement so start timer
                # as this if statement runs through
                state_clock.reset()
                
                # moves to next trial index
                trial += 1
                if trial >= n_total:
                    state_current = "state_finished"
                    state_entry = True
                else:
                    # ds = trial table, trial = current row index
                    # grabs this trials rows stimulus coordinates and plugs them
                    # into function from util_func_stimcat and converts (x,y) into 
                    # actual grating settings
                    sf_cycles_per_pix, ori_deg = stim_xy_to_sf_ori_deg(
                        ds['x'].iloc[trial],
                        ds['y'].iloc[trial],
                        PX_PER_CM,
                    )

                    # cat = trial rows class label
                    # upper() prevents case mismatch (a -> A), if statement just makes
                    # sure that error will spring if not A or B category
                    # phase is per-row metadata used to choose trigger codes + feedback
                    cat = str(ds['cat'].iloc[trial]).upper()
                    if cat not in {"A", "B"}:
                        raise ValueError(
                            f"Category labels must be 'A' or 'B'. Got: {cat}")
                    phase = ds['phase'].iloc[trial]

                    # start next trial with empty (.nan) for current trial record 
                    trig_stim = np.nan
                    trig_resp = np.nan
                    trig_fb = np.nan
                    flip_times["t_stim"] = np.nan
                    flip_times["t_fb"] = np.nan

                    # grating is psychopy stimulus object and this updates properties
                    # in state_stim, grating.draw() shows configured stimulus
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
            
            # selects/assigns the trigger code to trif based on phase and cat
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

                # if trig is not nan, send the assigned trig trigger 
                if not np.isnan(trig):
                    eeg.flip_pulse(trig, global_clock=global_clock)

                    # stores trigger code as integer in trig_stim which is written to .csv
                    trig_stim = int(trig)

                state_clock.reset()
                
                # reset here so that it runs once (not every frame) so that the clock
                # measures the elapsed time since stimulus_state onset 
                # resetting here gives clean zero-time reference for this stim period
                stim_clock.reset()

                win.callOnFlip(lambda: flip_times.__setitem__("t_stim", global_clock.getTime()))
                win.callOnFlip(kb.clock.reset)
                state_entry = False

            # getTime() is in secs, this converts to ms 
            time_state = state_clock.getTime() * 1000.0

            # draws grating decided in state_iti 
            grating.draw()

            # polls keyboard buffer for either d or k key
            keys = kb.getKeys(keyList=['d', 'k'], waitRelease=False)

            # k = keys[-1] is a list of KeyPress objects returned this frame, [-1] picks
            # the most recent one if multiple occured
            # k.rt is converted to ms by * 1000
            # getKeys() creates KeyPress objects with fields like .name and .rt
            # this code reads those fields and copies them into trial variables
            if keys:
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

                # absolute experiment timestamp of response which is different from
                # from rt (calculated by k.rt) 
                t_resp = global_clock.getTime()

                if cat == resp_label:
                    fb = "Correct"
                else:
                    fb = "Incorrect"
            
                # stores participants response ("A"/"B") seperately from correctness
                # later saved as in trial_data as "resp"
                resp = resp_label

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

            # same grating for this trial from iti
            # draws same stimulus again
            grating.draw()

            # jitter
            if time_state >= gap_ms:
                state_current = "state_feedback"
                state_entry = True

            win.flip()

        # --------------------- STATE: FEEDBACK ---------------------

        # uses phase and feedback to determine line colour and trigger code
        elif state_current == "state_feedback":
            if state_entry:
                if phase == 'train':
                    if fb == "Correct":
                        fb_ring.lineColor = 'green'
                        trig = TRIG["FB_COR_TRAIN"]
                    else:
                        fb_ring.lineColor = 'red'
                        trig = TRIG["FB_INC_TRAIN"]
                elif phase == 'test':
                    if fb == "Correct":
                        fb_ring.lineColor = 'green'
                        trig = TRIG["FB_COR_PROBE"]
                    else:
                        fb_ring.lineColor = 'red'
                        trig = TRIG["FB_INC_PROBE"]
                else:
                    trig = np.nan

                if not np.isnan(trig):
                    eeg.flip_pulse(trig, global_clock=global_clock)
                    trig_fb = int(trig)

                # schedules flip_times to be written at next screen flip
                # win.callOnFlip() is a function that is queued to run on next screen flip
                # when this runs, callOnFlip stores callback, on next win.flip(), screen
                # is updated, at that moment, psychopy executes queued callbacks and the
                # lambda runs and writes the timestamp to flip_times["t_fb"]
                win.callOnFlip(lambda: flip_times.__setitem__("t_fb", global_clock.getTime()))

                # lambda creates a small anonymous function (lambda args: expression)
                # here it means "a function with no arguments that, when called, sets 
                # flip_times to current time. it wraps the action so that it happens later
                # not now. without it, the callOnFlip would execute immediately and pass
                # ther result (None) to callOnFlip.

                # __setitem__ = "dunder" method (double __)
                # obj.__setitem__(key, value)
                # example: these two are the same thing 
                # flip_times["t_fb"] = 12.345
                # flip_times.__setitem__("t_fb", 12.345)

                # this always returns None but if it is called later (instead of immediately)
                # then this None is ignored (getTime() produces number, __setitem__ uses
                # that number to set flip_times to some number, then __setitem__ returns
                # none and this is ignored because callOnFlip just calls the function, it
                # does not use what the function returns

                # also, if called immediately, there would be a valid time but too early
                # (before flip), then it would execute __setitem__ which returns none, and
                # that none would be passed to callOnFlip

                # if called immediately: dict would get updated with the valid time from
                # getTime(), but, callOnFlip takes a function, not a timestamp, so the 
                # None returned from __setitem__ would get passed into callOnFlip instead
                # so nothing useful is scheduled (for what we want in this task -- 
                # technically, it produces a valid scheduling timestamp but we don't want
                # the scheduling time, we want the display onset) -- i.e., if callOnFlip
                # gets None, it would not have a valid callback to run at flip, hence why
                # we use lambda (to delay this) 

                # starts feedback state timer at 0
                state_clock.reset()
                state_entry = False

            time_state = state_clock.getTime() * 1000.0

            # draw same grating used on this trial + feedback ring initialised in  
            # state entry
            grating.draw()
            fb_ring.draw()

            if time_state > 1000:

                # copies expt doncition to trial variable, used below for .csv save
                probe_condition = condition

                # takes this trials (x,y) and transforms them to be saved into .csv
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

                state_current = "state_iti"
                state_entry = True

                # ensures rt cannot carry forward into next states
                rt = -1

            win.flip()

    # --------------------------- Cleanup ------------------------------------------
    eeg.close()
    win.close()
    core.quit()
    sys.exit()
