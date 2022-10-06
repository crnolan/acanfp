# %% [markdown]
# Phil, if you make markdown cells like this in VS code, we can export
# to a Jupyter notebook at a later date.
#
# Explore our fibre photometry data

# %%
import numpy as np
import pandas as pd
import scipy.signal as sig
import holoviews as hv
from holoviews import opts
import datashader as ds
from holoviews.operation.datashader import datashade, dynspread
hv.extension('bokeh')
import panel
panel.extension(comms='vscode')


# %%
# ## Import data

# Start by importing the downsampled photometry data into a Pandas
# dataframe, and load the events into a long format dataframe.

# %% 
column_map = {
    'Time (sec)': 'time',
    'filt465': 'f465',
    'filt405': 'f405'
}
df = pd.read_csv('../etc/2_R1_LateAcq_signals.csv').rename(column_map, axis=1)
fs = 1 / (df.iloc[1].time - df.iloc[0].time)
events_df = pd.read_csv('../etc/2_R1_LateAcq_EventTimes.csv').stack()
events_df = events_df[~events_df.isna()]

# %% [markdown]
# Now visualise the 405 and 465 signals

# %%
curves = {'465': hv.Curve((df.time, df.f465)),
          '405': hv.Curve((df.time, df.f405))}
overlay = hv.NdOverlay(curves, kdims='Channel')
spread = dynspread(datashade(overlay,
                             aggregator=ds.by('Channel', ds.count())))
spread.redim(x='time', y='raw').opts(width=800)

# %% [markdown] Let's create a mask and visualise the mask over the
# signal.
#
# Note: We should probably create a function library that makes the
# visualisation a little simpler for the students.

# %%
mask = pd.Series(index=df.time, dtype=bool)
mask.name = 'mask'
mask[:] = False
mask.loc[:33] = True
mask.loc[1243:] = True

def get_masked_regions(ts, mask):
    mask_diff = np.concatenate([mask[:1].astype(int),
                                np.diff(mask.astype(int))])
    mask_onset = ts[mask_diff == 1]
    mask_offset = ts[mask_diff == -1]
    if mask_offset.shape[0] < mask_onset.shape[0]:
        mask_offset = np.concatenate([mask_offset, ts[-1:]])
    return mask_onset, mask_offset


def plot_regions(onsets, offsets):
    return hv.Overlay([hv.VSpan(t0, t1) for t0, t1 in zip(onsets, offsets)])


mask_regions = get_masked_regions(mask.index, mask.to_numpy())
mask_spans = plot_regions(*mask_regions)

(spread * mask_spans).opts(
    opts.VSpan(fill_color='red', line_width=0)).redim(x='time', y='raw').opts(
        width=800)

# %% [markdown]
# Now fit the 465 signal and get a dF/F

# %%
def normalise(signal, control, mask, fs, method='fit', detrend=True):
    fit = np.polyfit(control[~mask], signal[~mask], deg=1)
    signal_fit = signal.copy()
    signal_fit[~mask] = fit[0] * control[~mask] + fit[1]
    df = signal - signal_fit
        
    if method == 'fit':
        return df / signal_fit
    elif method == 'const':
        return df / np.mean(signal)
    elif method == 'df':
        return df
    elif method == 'yfit':
        return signal_fit
    else:
        raise ValueError("Unrecognised normalisation method {}".format(method))


dff = normalise(df.f465.to_numpy(), df.f405.to_numpy(), mask, fs)

# %% [markdown]
# ... and plot the fitted signal

# %%
curves = {'dff': hv.Curve((df.time, dff))}
overlay = hv.NdOverlay(curves, kdims='Channel')
spread = dynspread(datashade(overlay,
                             aggregator=ds.by('Channel', ds.count())))
(spread * mask_spans).opts(
    opts.VSpan(fill_color='red', line_width=0)).redim(x='time', y='raw').opts(
        width=800)


# %%
