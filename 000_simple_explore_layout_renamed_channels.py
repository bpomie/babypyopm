""" 000_simple_explore_layout_renamed_channels.py
    Script for handling sensor layout operations in OPM data.


    Create and apply sensor montages for OPM recordings from a CSV sensor-location file.

    This script reads a raw FIF file and a sensor-location CSV, updates sensor
    positions and orientation vectors in the MNE Raw object, optionally applies
    coordinate-system rotations, renames channels to slot-based labels, and builds
    an MNE montage for visualization and downstream analysis.

    The script is designed to work with any helmet layout provided as a CSV
    containing sensor positions (X, Y, Z) and orientation vectors
    (x_i...z_k), including infant, adult, smart-helmet, and custom OPM layouts.

    Main features
    -------------
    - Load sensor geometry from CSV
    - Update sensor positions and orientation vectors in raw.info
    - Handle missing or unmatched sensors
    - Convert coordinates from mm to m
    - Optional X/Y/Z-axis rotation transforms
    - Rename channels using slot-based naming conventions
    - Create and apply an MNE montage
    - Visualize 2D and 3D sensor layouts
    - Save a new FIF containing the updated montage information

        -----
        Authors:
        Anna Kowalczyk <a.u.kowalczyk@bham.ac.uk>
        Ana Pesquita <a.pesquita@aston.ac.uk>
        Barbara Pomiechowska <b.pomiechowska@bham.ac.uk>

"""
import os
import re
import glob
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

subj = 'sub-107'
task = 'oddballTones'

# Path to your project folder (contains 'data/' and 'montages/')
root_data_path = '/Users/a.pesquita@bham.ac.uk/Documents/GitHub/babypyopm/Untitled'

# Rotation options:
# None
# "z180"
# "x180"
# "y180"

#ROTATION = None
ROTATION = "z180"
# =============================================================================
# ANONYMISATION
# =============================================================================

ANONYMISE = True

# =============================================================================
# PATHS
# =============================================================================

path_data = os.path.join(root_data_path, 'data')
path_montages = os.path.join(root_data_path, 'montages')

os.makedirs(path_montages, exist_ok=True)
os.makedirs(
    os.path.join(path_data, subj, 'raw_rotated_sensorlocations'),
    exist_ok=True
)

# Path task
match_task = os.path.join(
    path_data, subj, 'raw_recording',
    f"*_{subj}_file-{task}_raw.fif"
)

files_task = glob.glob(match_task)
print(files_task)

if not files_task:
    raise FileNotFoundError(f"No task file matched: {match_task}")

path_task_data_raw = files_task[0]

save_task = os.path.join(
    path_data, subj, 'raw_rotated_sensorlocations',
    f"{subj}_file-{task}_upright_wsensorlocations_raw.fif"
)

# Path emptyroom
match_emptyroom = os.path.join(
    path_data, subj, 'raw_recording',
    f"*_{subj}_file-emptyroom_raw.fif"
)

files_emptyroom = glob.glob(match_emptyroom)
print(files_emptyroom)

if not files_emptyroom:
    raise FileNotFoundError(f"No emptyroom file matched: {match_emptyroom}")

path_emptyroom_data_raw = files_emptyroom[0]

save_emptyroom = os.path.join(
    path_data, subj, 'raw_rotated_sensorlocations',
    f"{subj}_file-emptyroom_upright_wsensorlocations_raw.fif"
)

# Path sensor locations
layout_file = os.path.join(
    path_data, subj, f"{subj}_sensor_locations.csv"
)

# PRINT PATHS
print("Task data raw FIF  :", path_task_data_raw)
print("Empty room data FIF:", path_emptyroom_data_raw)
print("Sensor locations CSV:", layout_file)

# Recordings to process: (input fif, output fif, tag used in plot filenames)
recordings = [
    (path_task_data_raw, save_task, task),
    (path_emptyroom_data_raw, save_emptyroom, 'emptyroom'),
]

# =============================================================================
# LOAD CSV (shared by both recordings)
# =============================================================================

print("Loading CSV...")
layout = pd.read_csv(layout_file)

layout = layout.dropna(subset=["channel_name"])

layout["channel_name"] = (
    layout["channel_name"]
    .astype(str)
    .str.strip()
)


# =============================================================================
# PROCESS EACH RECORDING (task, then emptyroom)
# =============================================================================

for fif_file, output_fif, tag in recordings:

    print("\n" + "#" * 60)
    print(f"RECORDING: {tag}")
    print("#" * 60)

    # =============================================================================
    # LOAD
    # =============================================================================

    print("Loading FIF...")
    raw = mne.io.read_raw_fif(
        fif_file,
        preload=True
    )

    # =============================================================================
    # SENSOR MATCHING
    # =============================================================================

    csv_channels = set(layout["channel_name"])

    fif_channels = set(raw.ch_names)

    missing_from_csv = []
    matched = 0

    # =============================================================================
    # UPDATE LOCATIONS
    # =============================================================================

    for ch in raw.info["chs"]:

        ch_name = ch["ch_name"]

        if ch_name not in csv_channels:

            ch["loc"][:12] = 0
            missing_from_csv.append(ch_name)

            continue

        matched += 1

        row = layout.loc[
            layout["channel_name"] == ch_name
        ].iloc[0]

        vals = row[
            [
                "X", "Y", "Z",
                "x_i", "x_j", "x_k",
                "y_i", "y_j", "y_k",
                "z_i", "z_j", "z_k"
            ]
        ].to_numpy(dtype=float)

        ch["loc"][:12] = vals

    missing_from_fif = sorted(
        csv_channels - fif_channels
    )

    # =============================================================================
    # REPORT MATCHING
    # =============================================================================

    print("\n" + "=" * 60)
    print("SENSOR MATCHING SUMMARY")
    print("=" * 60)

    print(f"Channels in FIF : {len(fif_channels)}")
    print(f"Channels in CSV : {len(csv_channels)}")
    print(f"Matched         : {matched}")

    print(
        f"\nMissing from CSV ({len(missing_from_csv)}):"
    )

    for ch in missing_from_csv:
        print("  ", ch)

    print(
        f"\nMissing from FIF ({len(missing_from_fif)}):"
    )

    for ch in missing_from_fif:
        print("  ", ch)
        
    # =============================================================================
    # DROP SENSORS AT (0,0,0)
    # =============================================================================

    zero_loc_chs = [
        ch["ch_name"]
        for ch in raw.info["chs"]
        if ch["kind"] == mne.io.constants.FIFF.FIFFV_MEG_CH
        and np.allclose(ch["loc"][:3], 0)
    ]

    print("\n" + "=" * 60)
    print("DROPPING SENSORS AT (0,0,0)")
    print("=" * 60)

    print(f"Channels to drop ({len(zero_loc_chs)}):")

    for ch_name in zero_loc_chs:
        print("  ", ch_name)

    if zero_loc_chs:
        raw.drop_channels(zero_loc_chs)

    print(f"\nChannels remaining: {len(raw.ch_names)}")

    # =============================================================================
    # MM -> M
    # =============================================================================

    for ch in raw.info["chs"]:

        if (
            ch["kind"]
            == mne.io.constants.FIFF.FIFFV_MEG_CH
        ):

            ch["loc"][:3] /= 1000

    # =============================================================================
    # ROTATIONS
    # =============================================================================

    ROTATIONS = {

        None: np.eye(3),

        "z180": np.array([
            [-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0,  1]
        ]),

        "x180": np.array([
            [ 1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1]
        ]),

        "y180": np.array([
            [-1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0, -1]
        ])
    }

    R = ROTATIONS[ROTATION]

    print(f"\nApplying rotation: {ROTATION}")

    for ch in raw.info["chs"]:

        if (
            ch["kind"]
            != mne.io.constants.FIFF.FIFFV_MEG_CH
        ):
            continue

        loc = ch["loc"]

        loc[:3]   = R @ loc[:3]
        loc[3:6]  = R @ loc[3:6]
        loc[6:9]  = R @ loc[6:9]
        loc[9:12] = R @ loc[9:12]

    # =============================================================================
    # RENAME CHANNELS
    # side + slot + sensor direction
    # =============================================================================

    side_col = None

    for candidate in ["side", "channel_side", "helmet_side"]:

        if candidate in layout.columns:
            side_col = candidate
            break

    print(f"Side column: {side_col}")

    if side_col is None:
        raise ValueError(
            "Could not find side column. Expected one of: "
            "side, channel_side, helmet_side"
        )

    rename_dict = {}

    for _, row in layout.iterrows():

        old_name = str(
            row["channel_name"]
        ).strip()

        if old_name not in raw.ch_names:
            continue

        side = str(
            row[side_col]
        ).strip()

        side_letter = side[0].upper()

        slot = str(
            row["slot"]
        ).strip()

        # remove leading L/R from slot if already present
        # examples:
        # L11 -> 11
        # R_01 -> 01
        # Left side + L11 => L11 not LL11

        slot = re.sub(
            r"^[LRlr][_-]?",
            "",
            slot
        )

        match = re.search(
            r'_(bx|by|bz)$',
            old_name,
            re.IGNORECASE
        )

        if match:
            direction = match.group(1).lower()
        else:
            direction = "bz"

        # new_name = f"{side_letter}{slot}_{direction}" Changed to not include direction so it later matches the naming convention in the bad channel files
        new_name = f"{side_letter}{slot}"

        rename_dict[old_name] = new_name


    print("\nRename preview:")

    for old_name, new_name in list(rename_dict.items())[:20]:
        print(f"{old_name} -> {new_name}")


    # Check for duplicates before renaming
    new_names = list(rename_dict.values())

    duplicates = pd.Series(new_names)
    duplicates = duplicates[duplicates.duplicated()]

    if len(duplicates):

        print("\nDuplicate names found:")

        for d in duplicates.unique():
            print(d)

        raise ValueError(
            "Renaming would create duplicate channel names."
        )

    raw.rename_channels(rename_dict)

    print("\nExample renamed channels:")

    for ch in raw.ch_names[:10]:
        print(ch)

    # =============================================================================
    # BUILD MONTAGE
    # =============================================================================

    ch_pos = {

        ch["ch_name"]: ch["loc"][:3]

        for ch in raw.info["chs"]

        if not np.allclose(
            ch["loc"][:3],
            0
        )
    }

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        coord_frame="head"
    )

    raw.set_montage(
        montage,
        on_missing="ignore"
    )

    print(
        f"\nMontage created with "
        f"{len(ch_pos)} channels."
    )

    # =============================================================================
    # ANONYMISE
    # =============================================================================

    if ANONYMISE:

        print("\nAnonymising recording...")


        raw.set_meas_date(None)

        print("Done.")

    # =============================================================================
    # SAVE NEW FIF
    # =============================================================================

    raw.save(
        output_fif,
        overwrite=True
    )

    print(
        f"\nSaved montage FIF:\n"
        f"{output_fif}"
    )

    # =============================================================================
    # CHECK SENSOR LAYOUT
    # =============================================================================

    raw.plot_sensors(
        kind="topomap",
        show_names=True
    )

    plt.title(f"{subj} {tag}")
    plt.savefig(
        os.path.join(path_montages, f"plot_montage_{tag}_{subj}")
    )

    raw.plot_sensors(
        kind="3d",
        show_names=True
    )

    plt.title(f"{subj} {tag}")
    plt.savefig(
        os.path.join(path_montages, f"plot_3D_montage_{tag}_{subj}")
    )

    # =============================================================================
    # 3D MONTAGE WITH SENSOR ORIENTATIONS
    # =============================================================================

    fig = raw.plot_sensors(
        kind="3d",
        show_names=False
    )

    ax = fig.gca()

    positions = []
    x_orient = []
    y_orient = []
    z_orient = []

    for ch in raw.info["chs"]:

        if (
            ch["kind"]
            == mne.io.constants.FIFF.FIFFV_MEG_CH
        ):

            loc = ch["loc"]

            positions.append(loc[:3])
            x_orient.append(loc[3:6])
            y_orient.append(loc[6:9])
            z_orient.append(loc[9:12])

    positions = np.array(positions)
    x_orient = np.array(x_orient)
    y_orient = np.array(y_orient)
    z_orient = np.array(z_orient)

    scale = 0.02

    ax.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
              x_orient[:, 0], x_orient[:, 1], x_orient[:, 2],
              length=scale, color='red', normalize=True, label='X-axis')

    ax.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
              y_orient[:, 0], y_orient[:, 1], y_orient[:, 2],
              length=scale, color='green', normalize=True, label='Y-axis')

    ax.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
              z_orient[:, 0], z_orient[:, 1], z_orient[:, 2],
              length=scale, color='blue', normalize=True, label='Z-axis')

    ax.legend()

    plt.title(f"{subj} {tag}")
    plt.savefig(
        os.path.join(
            path_montages,
            f"plot_3D_w_orientations_montage_{tag}_{subj}"
        )
    )

    # =============================================================================
    # PSD CHECK
    # =============================================================================

    raw.compute_psd(
        fmax=120
    ).plot()

    plt.suptitle(f"PSD {tag}: {subj}")

    plt.show()

print("\nAll recordings processed.")