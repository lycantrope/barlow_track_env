# %%
import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import dateutil
import hdf5plugin
import numpy as np
import pynwb
import tifffile
from hdmf.backends.hdf5.h5_utils import H5DataIO
from ndx_multichannel_volume import (
    CElegansSubject,
    ImagingVolume,
    MultiChannelVolumeSeries,
    OpticalChannelPlus,
    OpticalChannelReferences,
)
from tqdm import tqdm

os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH


def rigid(sx, sy, rot, tx, ty) -> np.ndarray:
    scale = np.array(
        [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cos, sin = np.cos(rot), np.sin(rot)
    rotate = np.array(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translate = np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ]
    )

    return translate @ rotate @ scale


def load_mif_channel(
    height: int,
    width: int,
    frames: int,
    slices: int,
    path,
    flipX: bool,
    flipY: bool,
    scaleX: float,
    scaleY: float,
    rotXY: float,
    transX: float,
    transY: float,
    numC: int,
    useC: int,
    padZ: int = 0,
    batchsize: int = 64,
    **kwargs,
):
    with tifffile.TiffFile(path) as tif:
        axes = tif.series[0].axes.upper()

    data = tifffile.memmap(path)
    shape = data.shape
    # Instead of: assert len(axes) == len(shape)
    # Try forcing them to match:
    if len(axes) != len(shape):
        # Common issue: tifffile adds axes for dimensions of size 1
        # that memmap might have stripped.
        raise ValueError(
            f"Metadata axes {axes} ({len(axes)}) don't match data shape {shape} ({len(shape)})"
        )
    dims = {c: shape[i] for i, c in enumerate(axes)}

    target_c_idx = useC - 1
    if 0 > target_c_idx or target_c_idx > dims.get("C", 1):
        raise IndexError(
            f"Channel {useC} requested, but only {dims.get('C')} channels exist."
        )

    actual_z = dims.get("Z", 1)
    expected_z = padZ + actual_z
    if slices != expected_z:
        raise ValueError(
            f"Z-Slice Mismatch: MIF expects {slices}, calculation gives {expected_z}"
        )
    M = rigid(scaleX, scaleY, rotXY, transX, transY)
    M_inv = np.linalg.inv(M)

    # Create a grid of (x, y) coordinates
    grid = np.indices((height, width), dtype=np.float32)

    # To get your (3, N) matrix for the affine transformation:
    coords = np.vstack(
        [
            grid[1].reshape(1, -1),  # X
            grid[0].reshape(1, -1),  # Y
            np.ones((1, height * width), dtype=np.float32),
        ]
    )
    # 2. Apply your affine/perspective matrix M to the grid
    # M is your 3x3 matrix
    transformed_coords = M_inv @ coords
    # Perspective division
    map_x = (
        (transformed_coords[0] / transformed_coords[2])
        .reshape(height, width)
        .astype(np.float32)
    )
    map_y = (
        (transformed_coords[1] / transformed_coords[2])
        .reshape(height, width)
        .astype(np.float32)
    )

    idx = np.arange(frames, dtype=int)
    batched_idx = np.array_split(idx, max(1, (frames // batchsize)))

    for batch in batched_idx:
        # we read a batch of data to reduce the io access
        if "T" in dims:
            batch_data = data[batch]
        else:
            batch_data = data[None, ...]

        if flipX:
            batch_data = np.flip(batch_data, -1)
        if flipY:
            batch_data = np.flip(batch_data, -2)

        for src in batch_data:
            if "C" in dims:
                src = src[useC - 1]

            # wrapping single channel in list to make it iterable.
            if "Z" not in dims:
                src = src[None, ...]

            # 3. Apply the map to all slices in the Z-stack
            # This moves the loop into C++ internal logic
            final_z_stack = np.zeros((expected_z, height, width), dtype=src[0].dtype)
            for i, slice_z in enumerate(src):
                cv2.remap(
                    src=slice_z,
                    map1=map_x,
                    map2=map_y,
                    interpolation=cv2.INTER_CUBIC,
                    dst=final_z_stack[i + padZ],
                    borderMode=cv2.BORDER_REPLICATE,
                )

            yield final_z_stack


def parse_raw_str(x: str):
    try:
        ret = float(x)
        return int(ret) if ret.is_integer() else ret
    except ValueError:
        pass

    true_flags = ("true", "t", "1", "y", "yes")
    false_flags = ("false", "f", "0", "n", "no")
    if x.lower() in true_flags or x.lower() in false_flags:
        return x.lower() in true_flags

    return x


class MIFLoader:
    def __init__(self, mif_path):
        mif_path = Path(mif_path)
        mif_obj = mif_path.open("r")
        metadata = {}
        channel_props = defaultdict(dict)
        regex = re.compile(r"^([a-zA-Z]+)(\d+)?\s*=\s*(\S+);$")
        for line in mif_obj.readlines():
            res = regex.match(line)
            if res is None:
                continue
            key, gp, val = res.group(1), res.group(2), res.group(3)

            val = parse_raw_str(val)
            # Only the attributes end with number will be assigned to channels group.
            if gp is None:
                metadata[key] = val
            else:
                channel_props[gp][key] = val

        for k in channel_props:
            p = Path(channel_props[k]["path"])

            if not p.is_absolute():
                # If the file is not point to the correct
                p = mif_path.parent.joinpath(p)
            channel_props[k]["path"] = p
            channel_props[k]["useC"] = channel_props[k].get("useC", 1)

        width = metadata["width"]
        height = metadata["height"]
        slices = metadata.get("slices", 1)
        frames = metadata.get("frames", 1)

        # T, C, Z, Y, X
        self.shape = (frames, len(channel_props), slices, height, width)
        self.sorted_channels = [
            channel_props[ch] for ch in sorted(channel_props.keys(), key=int)
        ]

    def iter(self):
        T, C, Z, Y, X = self.shape
        loaders = [
            load_mif_channel(width=X, height=Y, frames=T, slices=Z, **chan_props)
            for chan_props in self.sorted_channels
        ]
        for vol in tqdm(zip(*loaders), total=T):
            vol = np.stack(vol, axis=0)
            yield vol


def get_datetime():
    return datetime.now(dateutil.tz.gettz("Japan"))


def init_nwbfile(
    strain,
    strain_info,
    session_date=None,
    description: str = "",
    growth_stage: str = "YA",
    cultivation_temp=20.0,
    lab="TOY",
    institution="UTokyo",
):
    if session_date is None:
        session_date = get_datetime()

    identifier = str(uuid4())

    subject = CElegansSubject(
        # This is the same as the NWBFile identifier for us, but does not have to be. It should just identify the subject for this trial uniquely.
        subject_id=identifier,
        # Age is optional but should be specified in ISO 8601 duration format similarly to what is shown here for growth_stage_time
        # age = pd.Timedelta(hours=2, minutes=30).isoformat(),
        # Date of birth is a required field but if you do not know or if it's not relevant, you can just use the current date or the date of the experiment
        date_of_birth=session_date,
        # Specify growth stage of worm - should be one of two-fold, three-fold, L1-L4, YA, OA, dauer, post-dauer L4, post-dauer YA, post-dauer OA
        growth_stage=growth_stage,
        # Specify temperature at which animal was cultivated
        cultivation_temp=cultivation_temp,
        description=strain_info,
        # Currently using the ontobee species link until NWB adds support for C. elegans
        species="http://purl.obolibrary.org/obo/NCBITaxon_6239",
        # Currently just using O for other until support added for other gender specifications
        sex="O",
        strain=strain,
    )
    return pynwb.NWBFile(
        session_description=description,
        # Can use any identity marker that is specific to an individual trial. We use date-time to specify trials
        identifier=identifier,
        session_start_time=session_date,
        lab=lab,
        institution=institution,
        related_publications="",
        subject=subject,
    )


def generate_calcium_imaging_data(
    nwb_obj: pynwb.NWBFile,
    loader: MIFLoader,
    rate,
    grid_spacing,
    ref_chan,
    indicator_chan,
    device_name="Olympus Spinning Disk Microscopy",
    compression=True,
):
    # 1. Create the Device (NWB requirement)
    device = nwb_obj.create_device(name=device_name)  # type: ignore

    # 2. Define Optical Channels (one for each in your MIF)
    # This matches your description: ch1=ref, ch2=indicator
    channels = [ref_chan, indicator_chan]
    optical_channel_plus = []
    opt_channels_ref = []
    for name, wave in channels:
        w1, w2, w3 = wave.split("-")
        excite = float(w1)
        emiss_mid = float(w2)
        emiss_range = float(w3[:-1])
        chan = OpticalChannelPlus(
            name=name,
            description=wave,
            excitation_lambda=excite,
            excitation_range=[excite, excite],
            emission_range=[emiss_mid - emiss_range / 2, emiss_mid + emiss_range / 2],
            emission_lambda=emiss_mid,
        )

        optical_channel_plus.append(chan)
        opt_channels_ref.append(wave)

    order_optical_channels = OpticalChannelReferences(
        name="order_optical_channels",
        channels=opt_channels_ref,
    )

    # 3. Create the Imaging Volume
    imaging_vol = ImagingVolume(
        name="WholeBrainVolume",
        optical_channel_plus=optical_channel_plus,
        order_optical_channels=order_optical_channels,
        device=device,
        grid_spacing=grid_spacing,
        grid_spacing_unit="micrometer",
        description="Whole Brain Calcium imaging setup",
        location="Whole Brain",
    )
    # Assign optical_channel to suppress warning.
    imaging_vol.optical_channel.extend(optical_channel_plus)
    nwb_obj.add_imaging_plane(imaging_vol)  # type: ignore
    # 4. Handle Compression
    # Note: Ensure data is an Iterable or a Chunked array if it's huge
    # Transpose TCZYX => TXYZC
    T, C, Z, Y, X = loader.shape
    chunks = (1, X, Y, Z, C)
    loader_transpose = pynwb.DataChunkIterator(
        data=(vol.T for vol in loader.iter()),
        maxshape=(T, X, Y, Z, C),
    )
    if compression:
        # My personal prefer zstd.
        data = H5DataIO(
            data=loader_transpose,
            chunks=chunks,
            allow_plugin_filters=True,
            **hdf5plugin.Zstd(clevel=3),  # type: ignore
        )
    else:
        data = H5DataIO(
            data=loader_transpose,
            chunks=chunks,
        )

    # 5. The Series
    calcium_image_series = MultiChannelVolumeSeries(
        name="CalciumImageSeries",
        imaging_volume=imaging_vol,
        data=data,
        description="Ch1: reference, Ch2: calcium indicator",
        dimension=(T, X, Y, Z, C),
        comments="Data layout: TXYZC",
        unit="raw uint16",  # unit is usually physical, e.g., 'bits' or 'n.a.'
        rate=float(rate),
        device=device,
    )

    nwb_obj.add_acquisition(nwbdata=calcium_image_series)


# %%
# HYPERPARAMS for creating files
CHANNELS = {
    "GCaMP": ("GCaMP", "488-515-30m"),
    "tdTomato": ("tdTomato", "561-641-75m"),
    "TagBFP": ("TagBFP", "405-440-40m"),
    "TagRFP675": ("TagRFP675", "640-750-100m"),
}


# Lab information
labcode = "TOY"
institution = "the University of Tokyo"

# %%

"""
In the mif please make sure that your reference channels was assign to the first channel.
path1 = "260214_cam2_tdTomato.tif"
"""

# Strain information
strain_name = "TOY1"
strain_info = "Is[H20::nls3::GCaMP6f+H20::nls2::tdTomato]; lite-1(ce314) gur-3(ok2245)"

# %%
# Acquisition information
reference_channel = CHANNELS["tdTomato"]
indicator_channel = CHANNELS["GCaMP"]

volumns_per_second = 1.5
grid_spacing_xyz = (0.35, 0.35, 1.5)  # unit: micrometer per pixel

# %%

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="MIF file path (*.mif)", required=True)
    args = parser.parse_args()

    mif_path = Path(args.input)

    if not mif_path.is_file() or not mif_path.suffix == ".mif":
        parser.error(f"Input file is is not a valid mif file. {args.input}")

    nwb_obj = init_nwbfile(
        strain=strain_name,
        strain_info=strain_info,
        lab=labcode,
        institution=institution,
    )

    # Create lazy loader to parse data from MIF-Tiff.
    mif_loader = MIFLoader(mif_path)

    # Add mif_loader into nwb_file
    generate_calcium_imaging_data(
        nwb_obj,
        loader=mif_loader,
        ref_chan=reference_channel,  # This will be channel1
        indicator_chan=indicator_channel,  # indicator will be channel2
        rate=volumns_per_second,  # volumns per second
        grid_spacing=grid_spacing_xyz,
    )

    # Writing data into disk
    with pynwb.NWBHDF5IO(path=mif_path.with_suffix(".nwb"), mode="w") as io:
        io.write(container=nwb_obj)
    logger.info(f"Successfully conversion of mif to nwb at: {str(mif_path)}")


if __name__ == "__main__":
    main()

# %%
