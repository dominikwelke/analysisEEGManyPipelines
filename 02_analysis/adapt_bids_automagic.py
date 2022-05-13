import pandas as pd
from glob import glob
from os import rename, remove
import os.path as op

sub_ids = [i for i in range(1,34)]

base_folder_bids = '../../dataset/eeg_automagic'
base_folder_events = '../../dataset/events'  # where to find the detailed events

# get participants.tsv
fname_participants = op.join(base_folder_bids,'participants.tsv')
d_participants = pd.read_csv(fname_participants, sep='\t')
if 'automagic_code' not in d_participants.keys():
    d_participants['automagic_code'] = [pd.NA]*len(d_participants)

# loop over participants
for sub_id in sub_ids:

    try:
        folder_eeg = op.join(base_folder_bids, 'sub-{0:03d}/eeg'.format(sub_id))

        ## adapt events.tsv
        # load event
        fname_events = op.join(base_folder_events, 'EMP{0:02d}_events.csv'.format(sub_id))
        fname_bids_events = op.join(
            folder_eeg,
            'sub-{0:03d}_task-xxxx_events.tsv'.format(sub_id))
        # fname_bids_events_rename = '../../dataset/eeg_automagic/sub-{0:03d}/eeg/sub-{0:03d}_task-xxxx_events.tsv_'.format(sub_id)

        d_events = pd.read_csv(fname_events)
        d_bids_events = pd.read_csv(fname_bids_events, sep='\t')

        # create new event names
        d_events['cat'] = 'trial-' + d_events['trial'].astype(str) + '/' + \
                          d_events['scene_category'] + '/' + \
                          d_events['old'] + '/' + \
                          d_events['behavior'] + '/' + \
                          d_events['subsequent_memory']

        # feed into new bids_events
        d_bids_events_new = d_bids_events[2:].copy().reset_index(drop=True)
        d_bids_events_new.trial_type = d_events.cat.copy()
        d_bids_events_new['value_tmp'] = d_events.trigger.copy().astype(str)

        assert (d_bids_events_new.value.equals(d_bids_events_new.value_tmp))
        d_bids_events_new = d_bids_events_new[
            ['onset', 'duration', 'sample', 'trial_type', 'response_time',
             'stim_file', 'value']]

        # save new file
        # rename(fname_bids_events, fname_bids_events_rename)
        d_bids_events_new.to_csv(fname_bids_events, sep='\t', index=False)

        del d_events, d_bids_events, d_bids_events_new


        ## adapt brainvision data files (unfortunately mne bids function doesnt work due to .dat naming! create PR)
        fname_vhdr = glob(op.join(folder_eeg, '*eeg.vhdr').format(sub_id))[0]
        fname_vhdr_renamed = op.join(folder_eeg, 'sub-{0:03d}_task-xxxx_eeg.vhdr').format(sub_id)
        assert(fname_vhdr != fname_vhdr_renamed)
        fname_vmrk = glob(op.join(folder_eeg, '*eeg.vmrk').format(sub_id))[0]
        fname_vmrk_renamed = op.join(folder_eeg, 'sub-{0:03d}_task-xxxx_eeg.vmrk').format(sub_id)
        fname_eeg_file = glob(op.join(folder_eeg, '*eeg.dat').format(sub_id))[0]
        fname_eeg_file_renamed = op.join(folder_eeg, 'sub-{0:03d}_task-xxxx_eeg.dat').format(sub_id)

        with open(fname_vhdr, 'r', encoding='UTF') as f:
            d_vhdr = f.readlines()
            #print(d_vhdr)
        d_vhdr[4] = 'DataFile=sub-001_task-xxxx_eeg.dat\n'
        d_vhdr[5] = 'MarkerFile=sub-001_task-xxxx_eeg.vmrk\n'

        with open(fname_vhdr_renamed, 'w', encoding='UTF') as f:
            [f.write(l) for l in d_vhdr]

        remove(fname_vhdr)
        rename(fname_vmrk, fname_vmrk_renamed)
        rename(fname_eeg_file, fname_eeg_file_renamed)


        ## adapt channels.tsv
        fname_channels = op.join(folder_eeg, 'sub-{:03d}_task-xxxx_channels.tsv'.format(sub_id))
        d_channels = pd.read_csv(fname_channels, sep='\t')[:-2]
        assert(len(d_channels) == 70)
        d_channels.to_csv(fname_channels, sep='\t', index=False, na_rep='n/a')


        ## adapt sidecar json
        fname_sidecar = glob(op.join(folder_eeg, '*eeg.json').format(sub_id))[0]
        fname_sidecar_new = op.join(folder_eeg, 'sub-{:03d}_task-xxxx_eeg.json'.format(sub_id))
        assert(fname_sidecar != fname_sidecar_new)

        with open(fname_sidecar, 'r', encoding='UTF') as f:
            d_sidecar = f.readlines()
            #print(d_sidecar)
        d_sidecar[1] = '  "InstitutionAddress": "Fliednerstraße 21, 48149 Münster, Germany",\n'
        d_sidecar[2] = '  "InstitutionName": "University of Münster",\n'

        with open(fname_sidecar_new, 'w', encoding='UTF') as f:
            [f.write(l) for l in d_sidecar]

        remove(fname_sidecar)


        ## add automagic quality assessment to participants.tsv
        assessment_id = fname_sidecar.split('_')[-2][5:]
        d_participants.automagic_code[sub_id-1] = assessment_id

    except AssertionError:
        print('sub-{:03d} skipped - already processed'.format(sub_id))

    except FileNotFoundError:
        print('sub-{:03d} skipped - no data'.format(sub_id))

# save final participants.tsv
d_participants.to_csv(fname_participants, sep='\t', na_rep='n/a', index= False)
