import torch
from torch.nn import (
    Module,
    Conv2d,
    BatchNorm2d,
    Identity,
    UpsamplingBilinear2d,
    Mish,
    ReLU,
    Sequential,
)
from torch.nn.functional import interpolate, grid_sample, pad
import numpy as np
from copy import deepcopy
import os, argparse, math
import tifffile as tif
from typing import Tuple, List, Mapping

from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    ensure_tuple,
    fall_back_tuple,
    look_up_option,
    convert_to_dst_type,
)
from monai.utils.misc import ensure_tuple_size, ensure_tuple_rep, issequenceiterable
from monai.networks.layers.convutils import gaussian_1d
from monai.networks.layers.simplelayers import separable_filtering
import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from segmentation_models_pytorch import MAnet

from skimage.io import imread as io_imread
from skimage.util.dtype import dtype_range
from skimage._shared.utils import _supported_float_type
from scipy.ndimage import find_objects, binary_fill_holes

import random

########################### Data Loading Modules #########################################################
DTYPE_RANGE = dtype_range.copy()
DTYPE_RANGE.update((d.__name__, limits) for d, limits in dtype_range.items())
DTYPE_RANGE.update(
    {
        "uint10": (0, 2 ** 10 - 1),
        "uint12": (0, 2 ** 12 - 1),
        "uint14": (0, 2 ** 14 - 1),
        "bool": dtype_range[bool],
        "float": dtype_range[np.float64],
    }
)


def _output_dtype(dtype_or_range, image_dtype):
    if type(dtype_or_range) in [list, tuple, np.ndarray]:
        # pair of values: always return float.
        return _supported_float_type(image_dtype)
    if type(dtype_or_range) == type:
        # already a type: return it
        return dtype_or_range
    if dtype_or_range in DTYPE_RANGE:
        # string key in DTYPE_RANGE dictionary
        try:
            # if it's a canonical numpy dtype, convert
            return np.dtype(dtype_or_range).type
        except TypeError:  # uint10, uint12, uint14
            # otherwise, return uint16
            return np.uint16
    else:
        raise ValueError(
            "Incorrect value for out_range, should be a valid image data "
            f"type or a pair of values, got {dtype_or_range}."
        )


def intensity_range(image, range_values="image", clip_negative=False):
    if range_values == "dtype":
        range_values = image.dtype.type

    if range_values == "image":
        i_min = np.min(image)
        i_max = np.max(image)
    elif range_values in DTYPE_RANGE:
        i_min, i_max = DTYPE_RANGE[range_values]
        if clip_negative:
            i_min = 0
    else:
        i_min, i_max = range_values
    return i_min, i_max


def rescale_intensity(image, in_range="image", out_range="dtype"):
    out_dtype = _output_dtype(out_range, image.dtype)

    imin, imax = map(float, intensity_range(image, in_range))
    omin, omax = map(
        float, intensity_range(image, out_range, clip_negative=(imin >= 0))
    )
    image = np.clip(image, imin, imax)

    if imin != imax:
        image = (image - imin) / (imax - imin)
        return np.asarray(image * (omax - omin) + omin, dtype=out_dtype)
    else:
        return np.clip(image, omin, omax).astype(out_dtype)


def _normalize(img):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [0, 99.5])
    img_norm = rescale_intensity(
        img, in_range=(percentiles[0], percentiles[1]), out_range="uint8"
    )

    return img_norm.astype(np.uint8)


def pred_transforms(img):
    # LoadImage
    # img = (
    #     tif.imread(filename)
    #     if filename.endswith(".tif") or filename.endswith(".tiff")
    #     else io_imread(filename)
    # )

    # if len(img.shape) == 2:
    #     img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
    # elif len(img.shape) == 3 and img.shape[-1] > 3:
    #     img = img[:, :, :3]

    img = img.astype(np.float32)
    img = _normalize(img)
    img = np.moveaxis(img, -1, 0)
    img = (img - img.min()) / (img.max() - img.min())

    return torch.FloatTensor(img).unsqueeze(0)


################################################################################

########################### MODEL Architecture #################################
class SegformerGH(MAnet):
    def __init__(
        self,
        encoder_name: str = "mit_b5",
        encoder_weights="imagenet",
        decoder_channels=(256, 128, 64, 32, 32),
        decoder_pab_channels=256,
        in_channels: int = 3,
        classes: int = 3,
    ):
        super(SegformerGH, self).__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_pab_channels=decoder_pab_channels,
            in_channels=in_channels,
            classes=classes,
        )

        convert_relu_to_mish(self.encoder)
        convert_relu_to_mish(self.decoder)

        self.cellprob_head = DeepSegmantationHead(
            in_channels=decoder_channels[-1], out_channels=1, kernel_size=3,
        )
        self.gradflow_head = DeepSegmantationHead(
            in_channels=decoder_channels[-1], out_channels=2, kernel_size=3,
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        gradflow_mask = self.gradflow_head(decoder_output)
        cellprob_mask = self.cellprob_head(decoder_output)

        masks = torch.cat([gradflow_mask, cellprob_mask], dim=1)

        return masks


class DeepSegmantationHead(Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d_1 = Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        bn = BatchNorm2d(in_channels // 2)
        conv2d_2 = Conv2d(
            in_channels // 2,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        mish = Mish(inplace=True)

        upsampling = (
            UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else Identity()
        )
        activation = Identity()
        super().__init__(conv2d_1, mish, bn, conv2d_2, upsampling, activation)


def convert_relu_to_mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, ReLU):
            setattr(model, child_name, Mish(inplace=True))
        else:
            convert_relu_to_mish(child)


#####################################################################################

########################### Sliding Window Inference #################################
class GaussianFilter(Module):
    def __init__(
        self, spatial_dims, sigma, truncated=4.0, approx="erf", requires_grad=False,
    ) -> None:
        if issequenceiterable(sigma):
            if len(sigma) != spatial_dims:  # type: ignore
                raise ValueError
        else:
            sigma = [deepcopy(sigma) for _ in range(spatial_dims)]  # type: ignore
        super().__init__()
        self.sigma = [
            torch.nn.Parameter(
                torch.as_tensor(
                    s,
                    dtype=torch.float,
                    device=s.device if isinstance(s, torch.Tensor) else None,
                ),
                requires_grad=requires_grad,
            )
            for s in sigma  # type: ignore
        ]
        self.truncated = truncated
        self.approx = approx
        for idx, param in enumerate(self.sigma):
            self.register_parameter(f"kernel_sigma_{idx}", param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _kernel = [
            gaussian_1d(s, truncated=self.truncated, approx=self.approx)
            for s in self.sigma
        ]
        return separable_filtering(x=x, kernels=_kernel)


def compute_importance_map(
    patch_size, mode=BlendMode.CONSTANT, sigma_scale=0.125, device="cpu"
):
    mode = look_up_option(mode, BlendMode)
    device = torch.device(device)

    center_coords = [i // 2 for i in patch_size]
    sigma_scale = ensure_tuple_rep(sigma_scale, len(patch_size))
    sigmas = [i * sigma_s for i, sigma_s in zip(patch_size, sigma_scale)]

    importance_map = torch.zeros(patch_size, device=device)
    importance_map[tuple(center_coords)] = 1
    pt_gaussian = GaussianFilter(len(patch_size), sigmas).to(
        device=device, dtype=torch.float
    )
    importance_map = pt_gaussian(importance_map.unsqueeze(0).unsqueeze(0))
    importance_map = importance_map.squeeze(0).squeeze(0)
    importance_map = importance_map / torch.max(importance_map)
    importance_map = importance_map.float()

    return importance_map


def first(iterable, default=None):
    for i in iterable:
        return i

    return default


def dense_patch_slices(image_size, patch_size, scan_interval):
    num_spatial_dims = len(image_size)
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(
                d
                for d in range(num)
                if d * scan_interval[i] + patch_size[i] >= image_size[i]
            )
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    return [tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]


def get_valid_patch_size(image_size, patch_size):
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))


class Resize:
    def __init__(self, spatial_size):
        self.size_mode = "all"
        self.spatial_size = spatial_size

    def __call__(self, img):
        input_ndim = img.ndim - 1  # spatial ndim
        output_ndim = len(ensure_tuple(self.spatial_size))

        if output_ndim > input_ndim:
            input_shape = ensure_tuple_size(img.shape, output_ndim + 1, 1)
            img = img.reshape(input_shape)

        spatial_size_ = fall_back_tuple(self.spatial_size, img.shape[1:])

        if (
            tuple(img.shape[1:]) == spatial_size_
        ):  # spatial shape is already the desired
            return img

        img_, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)

        resized = interpolate(
            input=img_.unsqueeze(0), size=spatial_size_, mode="nearest",
        )
        out, *_ = convert_to_dst_type(resized.squeeze(0), img)
        return out


def sliding_window_inference(
    inputs,
    roi_size,
    sw_batch_size,
    predictor,
    overlap,
    mode=BlendMode.CONSTANT,
    sigma_scale=0.125,
    padding_mode=PytorchPadMode.CONSTANT,
    cval=0.0,
    sw_device=None,
    device=None,
    roi_weight_map=None,
):
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    batch_size, _, *image_size_ = inputs.shape

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
    )
    pad_size = []

    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])

    inputs = pad(
        inputs,
        pad=pad_size,
        mode=look_up_option(padding_mode, PytorchPadMode).value,
        value=cval,
    )

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        importance_map = compute_importance_map(
            valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device
        )

    importance_map = convert_data_type(importance_map, torch.Tensor, device, compute_dtype)[0]  # type: ignore
    # handle non-positive weights
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(
        compute_dtype
    )

    # Perform predictions
    dict_key, output_image_list, count_map_list = None, [], []
    _initialized_ss = -1
    is_tensor_output = (
        True  # whether the predictor's output is a tensor (instead of dict/tuple)
    )

    # for each patch
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)]
            + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(
            sw_device
        )
        seg_prob_out = predictor(window_data)  # batched patch segmentation

        # convert seg_prob_out to tuple seg_prob_tuple, this does not allocate new memory.
        seg_prob_tuple: Tuple[torch.Tensor, ...]
        if isinstance(seg_prob_out, torch.Tensor):
            seg_prob_tuple = (seg_prob_out,)
        elif isinstance(seg_prob_out, Mapping):
            if dict_key is None:
                dict_key = sorted(seg_prob_out.keys())  # track predictor's output keys
            seg_prob_tuple = tuple(seg_prob_out[k] for k in dict_key)
            is_tensor_output = False
        else:
            seg_prob_tuple = ensure_tuple(seg_prob_out)
            is_tensor_output = False

        # for each output in multi-output list
        for ss, seg_prob in enumerate(seg_prob_tuple):
            seg_prob = seg_prob.to(device)  # BxCxMxNxP or BxCxMxN

            # compute zoom scale: out_roi_size/in_roi_size
            zoom_scale = []
            for axis, (img_s_i, out_w_i, in_w_i) in enumerate(
                zip(image_size, seg_prob.shape[2:], window_data.shape[2:])
            ):
                _scale = out_w_i / float(in_w_i)

                zoom_scale.append(_scale)

            if _initialized_ss < ss:  # init. the ss-th buffer at the first iteration
                # construct multi-resolution outputs
                output_classes = seg_prob.shape[1]
                output_shape = [batch_size, output_classes] + [
                    int(image_size_d * zoom_scale_d)
                    for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)
                ]
                # allocate memory to store the full output and the count for overlapping parts
                output_image_list.append(
                    torch.zeros(output_shape, dtype=compute_dtype, device=device)
                )
                count_map_list.append(
                    torch.zeros(
                        [1, 1] + output_shape[2:], dtype=compute_dtype, device=device
                    )
                )
                _initialized_ss += 1

            # resizing the importance_map
            resizer = Resize(spatial_size=seg_prob.shape[2:])

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                # zoom roi
                original_idx_zoom = list(
                    original_idx
                )  # 4D for 2D image, 5D for 3D image
                for axis in range(2, len(original_idx_zoom)):
                    zoomed_start = original_idx[axis].start * zoom_scale[axis - 2]
                    zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]

                    original_idx_zoom[axis] = slice(
                        int(zoomed_start), int(zoomed_end), None
                    )
                importance_map_zoom = resizer(importance_map.unsqueeze(0))[0].to(
                    compute_dtype
                )
                # store results and weights
                output_image_list[ss][original_idx_zoom] += (
                    importance_map_zoom * seg_prob[idx - slice_g]
                )
                count_map_list[ss][original_idx_zoom] += (
                    importance_map_zoom.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(count_map_list[ss][original_idx_zoom].shape)
                )

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] = (output_image_list[ss] / count_map_list.pop(0)).to(
            compute_dtype
        )

    # remove padding if image_size smaller than roi_size
    for ss, output_i in enumerate(output_image_list):
        zoom_scale = [
            seg_prob_map_shape_d / roi_size_d
            for seg_prob_map_shape_d, roi_size_d in zip(output_i.shape[2:], roi_size)
        ]

        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(
                pad_size[sp * 2],
                image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2],
            )
            slice_dim = slice(
                int(round(slice_dim.start * zoom_scale[num_spatial_dims - sp - 1])),
                int(round(slice_dim.stop * zoom_scale[num_spatial_dims - sp - 1])),
            )
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_i.shape):
            final_slicing.insert(0, slice(None))
        output_image_list[ss] = output_i[final_slicing]

    if dict_key is not None:  # if output of predictor is a dict
        final_output = dict(zip(dict_key, output_image_list))
    else:
        final_output = tuple(output_image_list)  # type: ignore

    return final_output[0] if is_tensor_output else final_output  # type: ignore


def _get_scan_interval(
    image_size, roi_size, num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    scan_interval = []

    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)

    return tuple(scan_interval)


#####################################################################################

########################### Main Inference Functions #################################
def post_process(pred_mask, device):
    dP, cellprob = pred_mask[:2], 1 / (1 + np.exp(-pred_mask[-1]))
    H, W = pred_mask.shape[-2], pred_mask.shape[-1]

    if np.prod(H * W) < (5000 * 5000):
        pred_mask = compute_masks(
            dP,
            cellprob,
            use_gpu=True,
            flow_threshold=0.4,
            device=device,
            cellprob_threshold=0.4,
        )[0]

    else:
        print("\n[Whole Slide] Grid Prediction starting...")
        roi_size = 2000

        # Get patch grid by roi_size
        if H % roi_size != 0:
            n_H = H // roi_size + 1
            new_H = roi_size * n_H
        else:
            n_H = H // roi_size
            new_H = H

        if W % roi_size != 0:
            n_W = W // roi_size + 1
            new_W = roi_size * n_W
        else:
            n_W = W // roi_size
            new_W = W

        # Allocate values on the grid
        pred_pad = np.zeros((new_H, new_W), dtype=np.uint32)
        dP_pad = np.zeros((2, new_H, new_W), dtype=np.float32)
        cellprob_pad = np.zeros((new_H, new_W), dtype=np.float32)

        dP_pad[:, :H, :W], cellprob_pad[:H, :W] = dP, cellprob

        for i in range(n_H):
            for j in range(n_W):
                print("Pred on Grid (%d, %d) processing..." % (i, j))
                dP_roi = dP_pad[
                    :,
                    roi_size * i : roi_size * (i + 1),
                    roi_size * j : roi_size * (j + 1),
                ]
                cellprob_roi = cellprob_pad[
                    roi_size * i : roi_size * (i + 1),
                    roi_size * j : roi_size * (j + 1),
                ]

                pred_mask = compute_masks(
                    dP_roi,
                    cellprob_roi,
                    use_gpu=True,
                    flow_threshold=0.4,
                    device=device,
                    cellprob_threshold=0.4,
                )[0]

                pred_pad[
                    roi_size * i : roi_size * (i + 1),
                    roi_size * j : roi_size * (j + 1),
                ] = pred_mask

        pred_mask = pred_pad[:H, :W]

    cell_idx, cell_sizes = np.unique(pred_mask, return_counts=True)
    cell_idx, cell_sizes = cell_idx[1:], cell_sizes[1:]
    cell_drop = np.where(cell_sizes < np.mean(cell_sizes) - 2.7 * np.std(cell_sizes))

    for drop_cell in cell_idx[cell_drop]:
        pred_mask[pred_mask == drop_cell] = 0

    return pred_mask


def hflip(x):
    """flip batch of images horizontally"""
    return x.flip(3)


def vflip(x):
    """flip batch of images vertically"""
    return x.flip(2)


class DualTransform:
    identity_param = None

    def __init__(
        self, name: str, params,
    ):
        self.params = params
        self.pname = name

    def apply_aug_image(self, image, *args, **params):
        raise NotImplementedError

    def apply_deaug_mask(self, mask, *args, **params):
        raise NotImplementedError


class HorizontalFlip(DualTransform):
    """Flip images horizontally (left->right)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = hflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = hflip(mask)
        return mask


class VerticalFlip(DualTransform):
    """Flip images vertically (up->down)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = vflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = vflip(mask)
        return mask


#################### GradFlow Modules ##################################################
from scipy.ndimage.filters import maximum_filter1d
import scipy.ndimage
import fastremap
from skimage import morphology

from scipy.ndimage import mean

torch_GPU = torch.device("cuda")
torch_CPU = torch.device("cpu")


def _extend_centers_gpu(
    neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device("cuda")
):
    if device is not None:
        device = device
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)

    T = torch.zeros((nimg, Ly, Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device).long()
    isneigh = torch.from_numpy(isneighbor).to(device)
    for i in range(n_iter):
        T[:, meds[:, 0], meds[:, 1]] += 1
        Tneigh = T[:, pt[:, :, 0], pt[:, :, 1]]
        Tneigh *= isneigh
        T[:, pt[0, :, 0], pt[0, :, 1]] = Tneigh.mean(axis=1)
    del meds, isneigh, Tneigh
    T = torch.log(1.0 + T)
    # gradient positions
    grads = T[:, pt[[2, 1, 4, 3], :, 0], pt[[2, 1, 4, 3], :, 1]]
    del pt
    dy = grads[:, 0] - grads[:, 1]
    dx = grads[:, 2] - grads[:, 3]
    del grads
    mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)
    return mu_torch


def diameters(masks):
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts ** 0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi ** 0.5) / 2
    return md, counts ** 0.5


def masks_to_flows_gpu(masks, device=None):
    if device is None:
        device = torch.device("cuda")

    Ly0, Lx0 = masks.shape
    Ly, Lx = Ly0 + 2, Lx0 + 2

    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y - 1, y + 1, y, y, y - 1, y - 1, y + 1, y + 1), axis=0)
    neighborsX = np.stack((x, x, x, x - 1, x + 1, x - 1, x + 1, x - 1, x + 1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = scipy.ndimage.find_objects(masks)

    centers = np.zeros((masks.max(), 2), "int")
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si

            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            yi, xi = np.nonzero(masks[sr, sc] == (i + 1))
            yi = yi.astype(np.int32) + 1  # add padding
            xi = xi.astype(np.int32) + 1  # add padding
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi - xmed) ** 2 + (yi - ymed) ** 2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i, 0] = ymed + sr.start
            centers[i, 1] = xmed + sc.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:, :, 0], neighbors[:, :, 1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array(
        [[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices]
    )
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = _extend_centers_gpu(
        neighbors, centers, isneighbor, Ly, Lx, n_iter=n_iter, device=device
    )

    # normalize
    mu /= 1e-20 + (mu ** 2).sum(axis=0) ** 0.5

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y - 1, x - 1] = mu
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c


def masks_to_flows(masks, use_gpu=False, device=None):
    if masks.max() == 0 or (masks != 0).sum() == 1:
        # dynamics_logger.warning('empty masks!')
        return np.zeros((2, *masks.shape), "float32")

    if use_gpu:
        if use_gpu and device is None:
            device = torch_GPU
        elif device is None:
            device = torch_CPU
        masks_to_flows_device = masks_to_flows_gpu

    if masks.ndim == 3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], device=device)[0]
            mu[[1, 2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:, y], device=device)[0]
            mu[[0, 2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:, :, x], device=device)[0]
            mu[[0, 1], :, :, x] += mu0
        return mu
    elif masks.ndim == 2:
        mu, mu_c = masks_to_flows_device(masks, device=device)
        return mu

    else:
        raise ValueError("masks_to_flows only takes 2D or 3D arrays")


def steps2D_interp(p, dP, niter, use_gpu=False, device=None):
    shape = dP.shape[1:]
    if use_gpu:
        if device is None:
            device = torch_GPU
        shape = (
            np.array(shape)[[1, 0]].astype("float") - 1
        )  # Y and X dimensions (dP is 2.Ly.Lx), flipped X-1, Y-1
        pt = (
            torch.from_numpy(p[[1, 0]].T).float().to(device).unsqueeze(0).unsqueeze(0)
        )  # p is n_points by 2, so pt is [1 1 2 n_points]
        im = (
            torch.from_numpy(dP[[1, 0]]).float().to(device).unsqueeze(0)
        )  # covert flow numpy array to tensor on GPU, add dimension
        # normalize pt between  0 and  1, normalize the flow
        for k in range(2):
            im[:, k, :, :] *= 2.0 / shape[k]
            pt[:, :, :, k] /= shape[k]

        # normalize to between -1 and 1
        pt = pt * 2 - 1

        # here is where the stepping happens
        for t in range(niter):
            # align_corners default is False, just added to suppress warning
            dPt = grid_sample(im, pt, align_corners=False)

            for k in range(2):  # clamp the final pixel locations
                pt[:, :, :, k] = torch.clamp(
                    pt[:, :, :, k] + dPt[:, k, :, :], -1.0, 1.0
                )

        # undo the normalization from before, reverse order of operations
        pt = (pt + 1) * 0.5
        for k in range(2):
            pt[:, :, :, k] *= shape[k]

        p = pt[:, :, :, [1, 0]].cpu().numpy().squeeze().T
        return p

    else:
        assert print("ho")


def follow_flows(dP, mask=None, niter=200, interp=True, use_gpu=True, device=None):
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)

    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    p = np.array(p).astype(np.float32)

    inds = np.array(np.nonzero(np.abs(dP[0]) > 1e-3)).astype(np.int32).T

    if inds.ndim < 2 or inds.shape[0] < 5:
        return p, None

    if not interp:
        assert print("woo")

    else:
        p_interp = steps2D_interp(
            p[:, inds[:, 0], inds[:, 1]], dP, niter, use_gpu=use_gpu, device=device
        )
        p[:, inds[:, 0], inds[:, 1]] = p_interp

    return p, inds


def flow_error(maski, dP_net, use_gpu=False, device=None):
    if dP_net.shape[1:] != maski.shape:
        print("ERROR: net flow is not same size as predicted masks")
        return

    # flows predicted from estimated masks
    dP_masks = masks_to_flows(maski, use_gpu=use_gpu, device=device)
    # difference between predicted flows vs mask flows
    flow_errors = np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += mean(
            (dP_masks[i] - dP_net[i] / 5.0) ** 2,
            maski,
            index=np.arange(1, maski.max() + 1),
        )

    return flow_errors, dP_masks


def remove_bad_flow_masks(masks, flows, threshold=0.4, use_gpu=False, device=None):
    merrors, _ = flow_error(masks, flows, use_gpu, device)
    badi = 1 + (merrors > threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks


def get_masks(p, iscell=None, rpad=20):
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)

    for i in range(dims):
        pflows.append(p[i].flatten().astype("int32"))
        edges.append(np.arange(-0.5 - rpad, shape0[i] + 0.5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims == 3:
        expand = np.nonzero(np.ones((3, 3, 3)))
    else:
        expand = np.nonzero(np.ones((3, 3)))
    for e in expand:
        e = np.expand_dims(e, 1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter == 0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i, e in enumerate(expand):
                epix = e[:, np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix >= 0, epix < shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix] > 2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter == 4:
                pix[k] = tuple(pix[k])

    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k

    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.9
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True)  # convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0

def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes
    (might have issues at borders between cells, todo: check and fix)
    
    Parameters
    ----------------
    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1
    Returns
    ---------------
    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            elif npix > 0:   
                if msk.ndim==3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:          
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1
    return masks

def compute_masks(
    dP,
    cellprob,
    p=None,
    niter=200,
    cellprob_threshold=0.4,
    flow_threshold=0.4,
    interp=True,
    resize=None,
    use_gpu=False,
    device=None,
):
    """compute masks using dynamics from dP, cellprob, and boundary"""

    cp_mask = cellprob > cellprob_threshold
    cp_mask = morphology.remove_small_holes(cp_mask, area_threshold=16)
    cp_mask = morphology.remove_small_objects(cp_mask, min_size=16)

    if np.any(cp_mask):  # mask at this point is a cell cluster binary map, not labels
        # follow flows
        if p is None:
            p, inds = follow_flows(
                dP * cp_mask / 5.0,
                niter=niter,
                interp=interp,
                use_gpu=use_gpu,
                device=device,
            )
            if inds is None:
                shape = resize if resize is not None else cellprob.shape
                mask = np.zeros(shape, np.uint16)
                p = np.zeros((len(shape), *shape), np.uint16)
                return mask, p

        # calculate masks
        mask = get_masks(p, iscell=cp_mask)

        # flow thresholding factored out of get_masks
        shape0 = p.shape[1:]
        if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
            # make sure labels are unique at output of get_masks
            mask = remove_bad_flow_masks(
                mask, dP, threshold=flow_threshold, use_gpu=use_gpu, device=device
            )
        
        mask = fill_holes_and_remove_small_masks(mask, min_size=15)
        
    else:  # nothing to compute, just make it compatible
        shape = resize if resize is not None else cellprob.shape
        mask = np.zeros(shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)
        return mask, p
    
    return mask, p

def visualize_instance_seg_mask(mask):
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    labels = np.unique(mask)
    label2color = {label: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for label in labels if label > 0}
    label2color[0] = (0, 0, 0)
    for label in labels:
        image[mask==label, :] = label2color[label]
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         if np.max(label2color[mask[i, j]]) > 0:
    #             print('####', np.max(label2color[mask[i, j]]), np.min(label2color[mask[i, j]]))
    #         image[i, j, :] = label2color[mask[i, j]]
    # image = image / 255
    image = image.astype(np.uint8)
    return image

def predict(img):
    # Dataset parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./weights/main_model.pth"
    
    model_path2 = "./weights/sub_model.pth"
    ###
    # model = torch.load(model_path, map_location=device)
    model = SegformerGH(
        encoder_name="mit_b5",
        encoder_weights=None,
        decoder_channels=(1024, 512, 256, 128, 64),
        decoder_pab_channels=256,
        in_channels=3,
        classes=3,
    )

    model.load_state_dict(torch.load(model_path,map_location="cpu"))
    model = model.to(device)
    model.eval()
    hflip_tta = HorizontalFlip()
    vflip_tta = VerticalFlip()

    img_data = pred_transforms(img)
    img_data = img_data.to(device)
    img_size = img_data.shape[-1] * img_data.shape[-2]
    
    if img_size < 1150000 and 900000 < img_size:
        overlap = 0.5
    else:
        overlap = 0.6
    print("start")
    with torch.no_grad():
        img0 = img_data
        outputs0 = sliding_window_inference(
            img0,
            512,
            4,
            model,
            padding_mode="reflect",
            mode="gaussian",
            overlap=overlap,
            device="cpu",
        )
        outputs0 = outputs0.cpu().squeeze()

        if img_size < 2000 * 2000:
            
            model.load_state_dict(torch.load(model_path2, map_location=device))
            model.eval()
            
            img2 = hflip_tta.apply_aug_image(img_data, apply=True)
            outputs2 = sliding_window_inference(
                img2,
                512,
                4,
                model,
                padding_mode="reflect",
                mode="gaussian",
                overlap=overlap,
                device="cpu",
            )
            outputs2 = hflip_tta.apply_deaug_mask(outputs2, apply=True)
            outputs2 = outputs2.cpu().squeeze()

            outputs = torch.zeros_like(outputs0)
            outputs[0] = (outputs0[0] + outputs2[0]) / 2
            outputs[1] = (outputs0[1] - outputs2[1]) / 2
            outputs[2] = (outputs0[2] + outputs2[2]) / 2
            
        elif img_size < 5000*5000:
            # Hflip TTA
            img2 = hflip_tta.apply_aug_image(img_data, apply=True)
            outputs2 = sliding_window_inference(
                img2,
                512,
                4,
                model,
                padding_mode="reflect",
                mode="gaussian",
                overlap=overlap,
                device="cpu",
            )
            outputs2 = hflip_tta.apply_deaug_mask(outputs2, apply=True)
            outputs2 = outputs2.cpu().squeeze()
            img2 = img2.cpu()
            
            ##################
            #                #
            #    ensemble    #
            #                #
            ##################
            
            model.load_state_dict(torch.load(model_path2, map_location=device))
            model.eval()
            
            img1 = img_data
            outputs1 = sliding_window_inference(
                img1,
                512,
                4,
                model,
                padding_mode="reflect",
                mode="gaussian",
                overlap=overlap,
                device="cpu",
            )
            outputs1 = outputs1.cpu().squeeze()
            
            # Vflip TTA
            img3 = vflip_tta.apply_aug_image(img_data, apply=True)
            outputs3 = sliding_window_inference(
                img3,
                512,
                4,
                model,
                padding_mode="reflect",
                mode="gaussian",
                overlap=overlap,
                device="cpu",
            )
            outputs3 = vflip_tta.apply_deaug_mask(outputs3, apply=True)
            outputs3 = outputs3.cpu().squeeze()
            img3 = img3.cpu()

            # Merge Results
            outputs = torch.zeros_like(outputs0)
            outputs[0] = (outputs0[0] + outputs1[0] + outputs2[0] - outputs3[0]) / 4
            outputs[1] = (outputs0[1] + outputs1[1] - outputs2[1] + outputs3[1]) / 4
            outputs[2] = (outputs0[2] + outputs1[2] + outputs2[2] + outputs3[2]) / 4
        else:
            outputs = outputs0

        pred_mask = post_process(outputs.squeeze(0).cpu().numpy(), device)
    # return img_data, seg_rgb, join(os.getcwd(), 'segmentation.tiff')
    # return visualize_instance_seg_mask(pred_mask)
    return pred_mask