"""
There is an effect of scene category (i.e., a difference between images showing
man-made vs. natural environments) on the amplitude of the N1 component, i.e.
the first major negative EEG voltage deflection.
"""
import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
import numpy as np

sub_ids = [i for i in range(1,34)]

bids_root = '../../dataset/eeg_BIDS/derivatives/automagic'

for sub_id in sub_ids:

    # set bids paths

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

    # filter data
    raw = raw.filter(.1,None)

    # epochs
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=-0.3, tmax=0.7,
        baseline=None,
        preload=True)

    # subset
    epochs_nat = epochs['natural']
    epochs_art = epochs['manmade']

    # evoked
    evoked_nat = epochs_nat.average()
    evoked_art = epochs_art.average()

    evoked_diff = mne.combine_evoked([evoked_nat,evoked_art],weights=[1,-1])

    # plot
    fig = evoked_diff.plot(spatial_colors=True,gfp=True)
    fig.savefig('sub-{:03d}_diff_butterflygfp.svg'.format(sub_id))

    fig = evoked_diff.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)
    fig.savefig('sub-{:03d}_diff_topo.svg'.format(sub_id))

    fig = evoked_diff.plot_joint(times=[0.1, 0.3, 0.4])
    fig.savefig('sub-{:03d}_diff_joint.svg'.format(sub_id))

    evokeds = dict(natural=evoked_nat, manmade=evoked_art)
    fig = mne.viz.plot_compare_evokeds(evokeds, combine='mean')
    fig[0].savefig('sub-{:03d}_compare.svg'.format(sub_id))

    # extract data for subsequent stats test
    tmp1 = np.expand_dims(evoked_nat.get_data(), axis=0).transpose(0,2,1)
    tmp2 = np.expand_dims(evoked_art.get_data(), axis=0).transpose(0,2,1)
    if sub_id == 1:
        X = [tmp1,tmp2]
    else:
        X[0] = np.concatenate([X[0],tmp1], axis=0)
        X[1] = np.concatenate([X[1],tmp2], axis=0)

# create grand average evoked object
evoked_diff_group = evoked_diff.copy()
evoked_diff_group.nave = len(sub_ids)
evoked_diff_group.data = X[0].mean(axis=0).transpose(1,0) - X[1].mean(axis=0).transpose(1,0)

# stats
# see https://mne.tools/stable/auto_tutorials/stats-sensor-space/20_erp_stats.html#sphx-glr-auto-tutorials-stats-sensor-space-20-erp-stats-py

# Calculate adjacency matrix between sensors from their locations
adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, "eeg")

# Extract data: transpose because the cluster test requires channels to be last
# In this case, inference is done over items. In the same manner, we could
# also conduct the test over, e.g., subjects.
#X = [epochs_nat.get_data().transpose(0, 2, 1),
#     epochs_art.get_data().transpose(0, 2, 1)]
tfce = dict(start=.1, step=.1)  # ideally start and step would be smaller

# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(
    X, tfce, adjacency=adjacency,
    n_permutations=10000)  # a more standard number would be 1000+
significant_points = cluster_pv.reshape(t_obs.shape).T < .05
print(str(significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
time_unit = dict(time_unit="s")
fig = evoked_diff_group.plot_joint(
    title="natural vs. manmade objects", ts_args=time_unit,
    topomap_args=time_unit,
    times=[0.09, .18, .3, .6])  # show difference wave
fig.savefig('ave_joint.svg')

# Create ROIs by checking channel labels
selections = mne.channels.make_1020_channel_selections(evoked_diff.info, midline="12z")

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked_diff_group.plot_image(
    axes=axes, group_by=selections, colorbar=False, show=False,
    mask=significant_points, show_names="all", titles=None,
    **time_unit)
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=.3,
             label="µV")

plt.show()
fig.savefig('ave_cluster.svg')
