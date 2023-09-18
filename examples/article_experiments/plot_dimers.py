# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import h5py
import re
import seaborn as sns

from pyscf.data.elements import ELEMENTS, CONFIGURATION

folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
folder = os.path.join(folder, "checkpoint_dimers")

tms = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]


def heatmap(data, row_labels, col_labels, molecules):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    from matplotlib import pyplot as plt

    plt.rcParams["savefig.pad_inches"] = 0

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot()

    ax = sns.heatmap(
        data,
        cmap="jet",
        annot=False,
        linewidths=0.5,
        vmax=1,
        xticklabels=col_labels,
        yticklabels=row_labels,
    )

    for molecule in molecules:
        a1, a2 = re.findall("[A-Z][^A-Z]*", molecule.split("_")[1])
        ax.add_patch(
            Rectangle((atoms.index(a1), atoms.index(a2)), 1, 1, fill=False, edgecolor="gold", lw=2)
        )
        ax.add_patch(
            Rectangle((atoms.index(a2), atoms.index(a1)), 1, 1, fill=False, edgecolor="gold", lw=2)
        )

    plt.autoscale(tight=True)

    plt.yticks(rotation=0, fontsize=14)
    plt.xticks(rotation=0, fontsize=14)

    return fig, ax


def annotate_heatmap(
    im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = plt.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# From a dict of dicts, create a dataframe
atoms = ELEMENTS[1:37]
atoms.remove("He")
atoms.remove("Ne")
atoms.remove("Ar")
atoms.remove("Kr")


def dict_to_df(dictionary):
    result_dict = {}
    for a1 in atoms:
        a1_list = []
        for a2 in atoms:
            found = False
            for k, v in dictionary.items():
                ka1, ka2 = re.findall("[A-Z][^A-Z]*", k)
                if (ka1 == a1 and ka2 == a2) or (ka1 == a2 and ka2 == a1):
                    a1_list.append(v)
                    found = True
            if not found:
                a1_list.append(np.nan)
        result_dict[a1] = a1_list
    dataframe = pd.DataFrame(result_dict)
    # print(dataframe.head())
    return dataframe


def average_error(prediction_dict, target_dict):
    """Computes the average error between the predictions and the targets in dimers containing/not containing transition metals"""
    mae_tms = []
    mae_no_tms = []
    for k in target_dict.keys():
        a1, a2 = re.findall("[A-Z][^A-Z]*", k)
        if a1 in tms or a2 in tms:
            mae_tms.append(abs(target_dict[k] - prediction_dict[k]))
        else:
            mae_no_tms.append(abs(target_dict[k] - prediction_dict[k]))
    return np.mean(mae_tms), np.mean(mae_no_tms)


# read json
data_file = os.path.join(
    folder, "dimers_wB97X_V.hdf5"
)  # todo: change the name of the data with the ground truth
targets_dict = {}
with h5py.File(data_file, "r") as f:
    for fkey in f.keys():
        key = fkey.split("_")[1]
        targets_dict[key] = f[fkey]["energy"][()]
    molecules = list(f.keys())
molecules = []


########################## Predictions #################################
predictions_json = os.path.join(
    folder, "dimers_wB97X_V.json"
)  # todo: change the name of the json with the predictions
with open(predictions_json, "r") as json_file:
    predictions_dict = json.load(json_file)

diff_dict = {}
for k, v in targets_dict.items():
    diff_dict[k] = abs(v - predictions_dict[k])
results_df = dict_to_df(diff_dict)
results_array = results_df.to_numpy()

training_hdf5_file = (
    "dimers_wB97X_V.hdf5"  # todo: change the name of the data the model was trained on
)
data_file = os.path.join(folder, training_hdf5_file)
with h5py.File(data_file, "r") as f:
    molecules = list(f.keys())

# make heatmap
fig, ax = heatmap(data=results_array, row_labels=atoms, col_labels=atoms, molecules=molecules)

file = os.path.join(folder, training_hdf5_file.split(".")[0] + ".pdf")
fig.savefig(file, dpi=300)
plt.close()

print(
    "When trained on non-transition metals, the MAE (Ha) for dimers containing / not-containing transition metals is: ",
    np.round(average_error(predictions_dict, targets_dict)[0], 4),
    np.round(average_error(predictions_dict, targets_dict)[1], 4),
)
