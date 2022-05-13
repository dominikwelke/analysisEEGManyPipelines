"""
There is an effect of scene category (i.e., a difference between images showing
man-made vs. natural environments) on the amplitude of the N1 component, i.e.
the first major negative EEG voltage deflection.
"""
import mne
from mne_bids import BIDSPath, read_raw_bids

import matplotlib.pyplot as plt


sub_id = 1

# set bids paths
bids_root = '../../dataset/eeg_automagic'

subject = '{:03d}'.format(sub_id)
datatype = 'eeg'
task = 'xxxx'
suffix = 'eeg'

bids_path = BIDSPath(subject=subject, task=task,
                     suffix=suffix, datatype=datatype, root=bids_root)
print(bids_path)

# load data
raw = read_raw_bids(bids_path=bids_path, verbose=False)
#print(raw.info['subject_info'])
events, event_id = mne.events_from_annotations(raw)

assert(raw.info['line_freq'] == 50.)
assert(raw.info['sfreq'] == 512.)


# set correct ref - QUESTION?!
raw.load_data()
raw, _ = mne.set_eeg_reference(raw, ['POz'])
#raw, _ = mne.set_eeg_reference(raw, 'average')
print(raw.info)

# plot montage (AF and IO postitions to be fixed)

picks = mne.pick_channels(raw.info.ch_names,include=[],exclude=['Afp9', 'Afp10', 'IO1', 'IO2'])
raw.pick(picks)

raw = raw.set_montage('standard_1020', on_missing='warn')


# epochs
epochs = mne.Epochs(
    raw,
    events=events,
    event_id=event_id,
    tmin=-0.3, tmax=0.7,
    preload=True)

# evoked
evoked_nat = epochs['natural'].average()
evoked_art = epochs['manmade'].average()
evoked_diff = evoked_nat.copy()
evoked_diff.data = evoked_art.get_data() - evoked_nat.get_data()

# plot
#f1 = evoked_nat.plot(spatial_colors=True)
#f3 = evoked_art.plot(spatial_colors=True)
plt.figure()
evoked_diff.plot(spatial_colors=True)
plt.show()

plt.figure()
evoked_diff.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)
plt.show()
#f4 = evoked_art.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)