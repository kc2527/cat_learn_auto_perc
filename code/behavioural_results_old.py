import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

dir_data = '../at_home_data'
dir_data_lab = '../behavioural_data'

df_train_rec = []
df_lab_rec = []

# not reading in cp task
for fd in os.listdir(dir_data):
    dir_data_fd = os.path.join(dir_data, fd)
    if os.path.isdir(dir_data_fd):
        for fs in os.listdir(dir_data_fd):
            f_full_path = os.path.join(dir_data_fd, fs)
            if os.path.isfile(f_full_path) and 'task_cp_' not in fs:

                df = pd.read_csv(f_full_path)

                # Participant 134 completed two files labelled session 1.
                # Re-number their sessions in completion-date order.
                if fs == 'sub_134_perc_sess_001_part_001_date_2026_08_05_data.csv':
                    df['session_num'] = 1

                elif fs == 'sub_134_perc_sess_001_part_001_date_2026_08_06_data.csv':
                    df['session_num'] = 2

                elif fs == 'sub_134_perc_sess_002_part_001_date_2026_08_07_data.csv':
                    df['session_num'] = 3

                elif fs == 'sub_134_perc_sess_003_part_001_date_2026_08_13_data.csv':
                    df['session_num'] = 4

                elif fs == 'sub_134_perc_sess_004_part_001_date_2026_08_14_data.csv':
                    df['session_num'] = 5

                elif fs == 'sub_134_perc_sess_005_part_001_date_2026_08_16_data.csv':
                    df['session_num'] = 6

                elif fs == 'sub_134_perc_sess_006_part_001_date_2026_08_19_data.csv':
                    df['session_num'] = 7

                elif fs == 'sub_134_perc_sess_007_part_001_date_2026_08_20_data.csv':
                    df['session_num'] = 8

                elif fs == 'sub_134_perc_sess_007_part_002_date_2026_08_20_data.csv':
                    df['session_num'] = 8

                elif fs == 'sub_134_perc_sess_008_part_001_date_2026_08_23_data.csv':
                    df['session_num'] = 9

                elif fs == 'sub_134_perc_sess_009_part_001_date_2026_08_25_data.csv':
                    df['session_num'] = 10

                # Participant 482 also has repeated session-number labels.
                # Re-number their sessions in completion-date order.
                elif fs == 'sub_482_perc_sess_001_part_001_date_2026_08_07_data.csv':
                    df['session_num'] = 1

                elif fs == 'sub_482_perc_sess_001_part_001_date_2026_08_09_data.csv':
                    df['session_num'] = 2

                elif fs == 'sub_482_perc_sess_002_part_001_date_2026_08_10_data.csv':
                    df['session_num'] = 3

                elif fs == 'sub_482_perc_sess_002_part_001_date_2026_08_15_data.csv':
                    df['session_num'] = 4

                elif fs == 'sub_482_perc_sess_003_part_001_date_2026_08_16_data.csv':
                    df['session_num'] = 5

                elif fs == 'sub_482_perc_sess_003_part_001_date_2026_08_17_data.csv':
                    df['session_num'] = 6

                elif fs == 'sub_482_perc_sess_003_part_001_date_2026_08_19_data.csv':
                    df['session_num'] = 7

                elif fs == 'sub_482_perc_sess_003_part_001_date_2026_08_23_data.csv':
                    df['session_num'] = 8

                elif fs == 'sub_482_perc_sess_003_part_001_date_2026_08_24_data.csv':
                    df['session_num'] = 9


                df['f_name'] = fs
                df_train_rec.append(df)

for fd in os.listdir(dir_data_lab):
    dir_data_lab_fd = os.path.join(dir_data_lab, fd)
    if os.path.isdir(dir_data_lab_fd):
        for fs in os.listdir(dir_data_lab_fd):
            f_full_path = os.path.join(dir_data_lab_fd, fs)
            if os.path.isfile(f_full_path) and fs.endswith('.csv'):

                    df = pd.read_csv(f_full_path)
                    df['f_name'] = fs
                    df_lab_rec.append(df)

d_lab = pd.concat(df_lab_rec, ignore_index=True)
d_home = pd.concat(df_train_rec, ignore_index=True)

block_size = 25

# NOTE: Setting dfs up 
# lab data 
# 600 trials -- train, 50 trials -- test
dd_lab = d_lab.sort_values(['subject_id', 'session_num', 'session_part',
                             'trial']).reset_index(drop=True)

dd_lab['acc'] = (dd_lab['cat'] == dd_lab['resp']).astype(int)
dd_lab['trial'] = dd_lab.groupby(['subject_id', 'session_num']).cumcount()
dd_lab['n_trials'] = dd_lab.groupby(['subject_id', 'session_num'])['trial'].transform('count')
dd_lab['block'] = dd_lab.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)

# ds for all, train, and test trials 
dd_lab_all = (dd_lab.groupby(['subject_id', 'session_num', 'block',
                              'probe_condition', 'phase'],
                             as_index=False)[['acc', 'rt']].mean().sort_values(['session_num',
                                                                        'subject_id',
                                                                        'block']))

dd_lab_train = dd_lab[dd_lab['phase'] == 'train'].groupby(['subject_id',
                                                           'session_num',
                                                           'probe_condition',
                                                           'block'])['acc'].mean().reset_index()

dd_lab_test = dd_lab[dd_lab['phase'] == 'test'].groupby(['subject_id',
                                                         'session_num',
                                                         'probe_condition',
                                                         'block'])['acc'].mean().reset_index()

# at-home data
# 400 trials -- train
dd_home = d_home.sort_values(['subject_id', 'session_num', 'session_part',
                               'trial']).reset_index(drop=True)

dd_home['acc'] = (dd_home['cat'] == dd_home['resp']).astype(int)
dd_home['trial'] = dd_home.groupby(['subject_id', 'session_num']).cumcount()
dd_home['n_trials'] = dd_home.groupby(['subject_id', 'session_num'])['trial'].transform('count')
dd_home['block'] = dd_home.groupby(['subject_id', 'session_num'])['trial'].transform(lambda x: x // block_size)

# NOTE: Inspect performance
# -- LAB -- 
# average accuracy per lab day (train trials)
remap = {1: 0.5, 2: 3.5, 3: 6.5, 4 : 9.5}

dd_lab_pd_avg = dd_lab_train.groupby(['subject_id', 'session_num'])['acc'].mean().reset_index()
dd_lab_pd_avg['session_type'] = 'Lab'
dd_lab_pd_avg['session_num'] = dd_lab_pd_avg['session_num'].map(remap)

# -- HOME -- 
dd_home_pd_avg = dd_home.groupby(['subject_id', 'session_num'])['acc'].mean().reset_index()
dd_home_pd_avg['session_type'] = 'Home'

# -- Combine lab and home --
dd_pd_avg = pd.concat([dd_lab_pd_avg, dd_home_pd_avg])
dd_pd_avg = dd_pd_avg.sort_values(['subject_id', 'session_num', 'session_type'])

# NOTE: Plots 
# -- LAB --
days_lab = sorted(d_lab['session_num'].unique()[:5])

# accuracy across task across days
fig, ax = plt.subplots(1, len(days_lab), squeeze = False, figsize=(24, 3.5), sharey=True)
for a, day in zip(ax.flat, days_lab):
      sns.lineplot(
          data=dd_lab[dd_lab['session_num'] == day],
          x='block',
          y='acc',
          hue='subject_id',
          legend=False,
          errorbar=None,
          ax=a
      )
      a.set_title(f'Day {day}')
      a.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# rts across task across days
fig, ax = plt.subplots(1, len(days_lab), squeeze = False, figsize=(24, 3.5),
                       sharey=False)
for a, day in zip(ax.flat, days_lab):
      sns.lineplot(
          data=dd_lab[dd_lab['session_num'] == day],
          x='block',
          y='rt',
          hue='subject_id',
          legend=False,
          errorbar=None,
          ax=a
      )
      a.set_title(f'Day {day}')
plt.tight_layout()
plt.show()

# average accuracy across participants across days
dd_lab_pd_avg['subject_id'] = dd_lab_pd_avg['subject_id'].astype('category')

fig, ax = plt.subplots(1, 1, squeeze = False)
sns.pointplot(data=dd_lab_pd_avg,
              x='session_num',
              y='acc',
              hue='subject_id',
              )
plt.tight_layout()
plt.show()

# -- HOME -- 
days_home = dd_home['session_num'].unique()[:16]

# accuracy across whole task across days
fig, ax = plt.subplots(1, len(days_home), squeeze = False, figsize=(24, 3.5), sharey=True)
for a, day in zip(ax.flat, days_home):
      sns.lineplot(
          data=dd_home[dd_home['session_num'] == day],
          x='block',
          y='acc',
          hue='subject_id',
          legend=False,
          errorbar=None,
          ax=a
      )
      a.set_title(f'Day {day}')
      a.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# average accuracy in task per day across days across all participants
dd_home_pd_avg['subject_id'] = dd_home_pd_avg['subject_id'].astype('category')

fig, ax = plt.subplots(1, 1, squeeze = False)
sns.pointplot(data=dd_home_pd_avg,
              x='session_num',
              y='acc',
              hue='subject_id',
              )
plt.tight_layout()
plt.show()

# -- 90 vs 180 COST -- 
# completed 650 trials of train, no test trials were completed
d_cost = dd_lab_all.copy() 

# dropping removed participants
drop_subs_exc = [303]
d_cost = d_cost[~((d_cost['subject_id'].isin(drop_subs_exc)))]

probe_location_block = {
        1: 24, 
        2: 6, 
        3: 12, 
        4: 16, 
        5: 8, 
        6: 18, 
        7: 14, 
        8: 10
        }

d_cost['probe_location_block'] = d_cost['session_num'].map(probe_location_block)

# the two training blocks immediately before the probe begins
pre_probe = (
    (d_cost['phase'] == 'train') &
    (d_cost['block'] >= d_cost['probe_location_block'] - 2) &
    (d_cost['block'] < d_cost['probe_location_block'])
)

probe = d_cost['phase'] == 'test'

d = d_cost[pre_probe | probe].copy()

dd = d.groupby(['subject_id', 'session_num', 'phase',
                           'probe_condition'])[['acc', 'rt']].mean().reset_index()

dd_wide = (dd.pivot_table(
    index=['subject_id', 'session_num', 'probe_condition'],
    columns='phase',
    values='acc',
    aggfunc='mean')
  .reset_index()
)

dd_wide['diff_score'] = dd_wide['train'] - dd_wide['test']

dd_wide['probe_condition'] = dd_wide['probe_condition'].astype('category')
dd_wide['subject_id'] = dd_wide['subject_id'].astype('category')

sns.set_palette('rocket', 2)

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
sns.pointplot(data=dd_wide,
              x = 'session_num',
              y = 'diff_score',
              hue = 'probe_condition',
              errorbar='se',
              linestyle='none',
              dodge=True
)
plt.show()

fig, ax = plt.subplots(1, 2, sharey=True, squeeze=False, figsize=(10, 5))
sns.lineplot(data=dd_wide[dd_wide['probe_condition'] == 90],
             x = 'session_num',
             y = 'diff_score',
             hue = 'subject_id',
             marker='o',
             ax=ax[0, 0]
)
ax[0,0].set_title('90')

sns.lineplot(data=dd_wide[dd_wide['probe_condition'] == 180],
             x = 'session_num',
             y = 'diff_score',
             hue = 'subject_id',
             marker='o',
             ax=ax[0, 1]
)
ax[0,1].set_title('180')

sns.move_legend(ax[0, 0], 'upper left', bbox_to_anchor=(1, 1))
sns.move_legend(ax[0, 1], 'upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

plt.savefig('90vs180.png')


dd_wide_rt = (dd.pivot_table(
    index=['subject_id', 'session_num', 'probe_condition'],
    columns='phase',
    values='rt',
    aggfunc='mean')
  .reset_index()
)

dd_wide_rt['diff_score'] = dd_wide_rt['train'] - dd_wide_rt['test']

dd_wide_rt['probe_condition'] = dd_wide_rt['probe_condition'].astype('category')
dd_wide_rt['subject_id'] = dd_wide_rt['subject_id'].astype('category')

sns.set_palette('rocket', 2)

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
sns.pointplot(data=dd_wide_rt,
              x = 'session_num',
              y = 'diff_score',
              hue = 'probe_condition',
              errorbar='se',
              linestyle='none',
              dodge=True
)
plt.show()
