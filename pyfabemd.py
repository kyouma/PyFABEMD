import os
import typing
import warnings

import numpy as np
import psutil
import scipy.ndimage
import scipy.spatial


RAM_LIMIT = 1  # Split image patches into batches with the batch size limited by this amount of GB


def limit_ram_usage(ratio: float = 0.9, use_available: bool = True):
    if os.name != 'posix':
        warnings.warn(
            message='The operating system is not POSIX. The memory will not be limited.'
        )
        return

    assert 0 <= ratio <= 1.0, 'Argument `ratio` must be between 0 and 1.'
    import resource

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    if soft_limit == -1:
        total_memory = (
            psutil.virtual_memory().available
            if use_available
            else psutil.virtual_memory().total
        )
        memory_limit = int(total_memory * ratio)
        soft_limit = min(soft_limit, memory_limit)
        hard_limit = min(hard_limit, memory_limit)
        resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))


def _find_local_extrema(
    image: np.typing.NDArray,
    extrema_type: typing.Literal['max', 'min'],
    extrema_scan_radius: int,
) -> np.typing.NDArray:
    if extrema_type == 'min':
        image = -image

    # Initialize service variables
    ndims = len(image.shape)
    window_sizes = (2 * extrema_scan_radius + 1,) * ndims
    center_ix = (extrema_scan_radius,) * ndims
    window_axes = tuple(range(-1, -1 - ndims, -1))
    padding_cval = image.min()

    # Search for non-strict extrema with the separable extrema filter, and then check and filter detected points
    extrema_filtered_image = scipy.ndimage.maximum_filter(
        image, size=window_sizes, mode='constant', cval=padding_cval
    )
    extrema_map = image >= extrema_filtered_image
    padded_image = np.pad(
        image,
        pad_width=extrema_scan_radius,
        mode='constant',
        constant_values=padding_cval,
    )
    patches = np.lib.stride_tricks.sliding_window_view(
        padded_image, window_shape=window_sizes
    )

    # Process non-strict local extrema in batches, check for strictness and remove not strict ones
    ram_limit = int(RAM_LIMIT * 1024**3)
    extrema_ix = np.where(extrema_map)
    extrema_ix = list(zip(*extrema_ix))
    batch_size = max(
        1,
        int(
            np.floor(ram_limit / np.prod(window_sizes, dtype=float) / patches.itemsize)
        ),
    )
    extrema_ix = [
        extrema_ix[i : i + batch_size] for i in range(0, len(extrema_ix), batch_size)
    ]
    extrema_ix = [tuple(map(np.array, zip(*ix))) for ix in extrema_ix]
    for ix in extrema_ix:
        patches_ = patches[ix]  # Here the program may crash by RAM

        windowed_comparison_map = image[ix][(...,) + (np.newaxis,) * ndims] > patches_
        windowed_comparison_map[(...,) + center_ix] = True
        true_map = windowed_comparison_map.all(axis=window_axes)

        extrema_map[tuple(x[~true_map] for x in ix)] = False

    extrema_coords = np.array(np.where(extrema_map)).T

    return extrema_coords


class IntrinsicModeFunctionData(typing.NamedTuple):
    imf: typing.Optional[np.typing.NDArray]
    residue: np.typing.NDArray
    imf_smoothing_distance: typing.Optional[int]


def _calculate_imf(
    input_image: np.typing.NDArray,
    extrema_scan_radius: int,
    smooth_by_which_distance: typing.Literal['max', 'min'],
    verbose: bool,
) -> IntrinsicModeFunctionData:
    # Find local extrema
    max_coords = _find_local_extrema(input_image, 'max', extrema_scan_radius)
    if verbose:
        print(f'    {len(max_coords)} local maxima have been found.')
    min_coords = _find_local_extrema(input_image, 'min', extrema_scan_radius)
    if verbose:
        print(f'    {len(min_coords)} local minima have been found.')

    # Check for the stopping criterion
    if (len(max_coords) + len(min_coords)) < 3:  # This criterion seems to be better than "any len(...) < 2"
        if verbose:
            print('\nThe extrema count stopping criteria has been achieved.')
            return IntrinsicModeFunctionData(
                imf=None, residue=input_image.copy(), imf_smoothing_distance=None
            )

    # Calculate the distances between the local extrema
    if verbose:
        print('    Calculating the distances between the local extrema... ', end='')
    smoothing_distance = []
    for coords in [max_coords, min_coords]:
        if coords.shape[0] > 1:
            neighbor_distances, _ = scipy.spatial.KDTree(coords).query(  # It may crash by RAM without KDTree
                coords, k=2, workers=-1
            )
            closest_neighbor_distances = neighbor_distances[:, 1]
            smoothing_distance.append(
                closest_neighbor_distances.max()
                if smooth_by_which_distance == 'max'
                else closest_neighbor_distances.min()
            )
        if verbose:
            print('Done', end=' and... ' if coords is max_coords else '\n')

    smoothing_distance = (
        max(smoothing_distance)
        if smooth_by_which_distance == 'max'
        else min(smoothing_distance)
    )
    smoothing_distance = 2 * int(np.ceil(smoothing_distance / 2)) + 1
    if verbose:
        print(f'    The new smoothing window size is {smoothing_distance}.')

    # Calculate coarse envelopes
    if verbose:
        print('    Calculating the coarse upper envelope... ', end='')
    upper_envelope = scipy.ndimage.maximum_filter(
        input_image, smoothing_distance, mode='nearest'
    )
    if verbose:
        print('Done')
    if verbose:
        print('    Calculating the coarse lower envelope... ', end='')
    lower_envelope = scipy.ndimage.minimum_filter(
        input_image, smoothing_distance, mode='nearest'
    )
    if verbose:
        print('Done')

    # Calculate residue with envelope smoothing (may have some issues on the border of the image)
    if verbose:
        print('    Smoothing the envelopes... ', end='')
    residue = np.mean([upper_envelope, lower_envelope], axis=0)
    residue = scipy.ndimage.uniform_filter(residue, smoothing_distance, mode='nearest')
    if verbose:
        print('Done')

    imf = input_image - residue
    return IntrinsicModeFunctionData(
        imf=imf, residue=residue, imf_smoothing_distance=smoothing_distance
    )


def fabemd(
    input_image: np.typing.NDArray,
    max_modes: typing.Optional[int] = None,
    initial_extrema_scan_radius: int = 1,
    smooth_by_which_distance: typing.Literal['max', 'min'] = 'min',
    increase_extrema_scan_radius_monotonically: bool = True,
    verbose: bool = False,
) -> tuple[list[np.typing.NDArray], np.typing.NDArray, list[int]]:
    '''
    Apply Fast and Adaptive Bidimensional Empirical Mode Decomposition [1, 2] algorithm to the image.

    The implementation is based on the description from [3], but with the extrema search radius monotonically increasing.

    References
    ----------
    [1] S. M. A. Bhuiyan, R. R. Adhami and J. F. Khan, "A novel approach of fast and adaptive bidimensional empirical mode decomposition," 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, Las Vegas, NV, USA, 2008, pp. 1313-1316. https://doi.org/10.1109/ICASSP.2008.4517859.

    [2] Bhuiyan, S.M.A., Adhami, R.R. & Khan, J.F. Fast and Adaptive Bidimensional Empirical Mode Decomposition Using Order-Statistics Filter Based Envelope Estimation. EURASIP J. Adv. Signal Process. 2008, 728356 (2008). https://doi.org/10.1155/2008/728356

    [3] M. U. Ahmed and D. P. Mandic, "Image fusion based on Fast and Adaptive Bidimensional Empirical Mode Decomposition," 2010 13th International Conference on Information Fusion, Edinburgh, UK, 2010, pp. 1-6. https://doi.org/10.1109/ICIF.2010.5711841.

    Parameters
    ----------
    image : np.ndarray
        Input N-dimensional 1-channel image.
    max_modes : int or None, optional
        Maximum number of intrinsic mode functions (IMFs) to compute, besides the residue. Unlimited if `None`. The default is `None`.
    initial_extrema_scan_radius : int, optional
        Initial radius to scan for local extrema. The default is 1.
    smooth_by_which_distance : ['max', 'min'], optional
        Which distance between the nearest extrema to use for smoothing. The default is 'min'.
    increase_extrema_scan_radius_monotonically : bool, optional
        Update the local extrema scan radius with the calculated smoothing window size, or always use the initial extrema scan radius. The default is True.
    verbose : bool, optional
        Print progress report during steps execution. The default is False.

    Returns
    -------
    imfs_list : list[np.ndarray]
        The list of length <number of IMFs> of the IMFs arrays, each of shape <input image shape>.
    residue : np.ndarray
        The residue array, of shape <input image shape>.
    imf_smoothing_window_sizes_list : list[int]
        The list of length <number of IMFs> of the envelope smoothing window sizes for each iteration.
    '''
    assert initial_extrema_scan_radius > 0
    assert smooth_by_which_distance in ['max', 'min']
    assert (max_modes is None) or (max_modes > 0)

    if not np.issubdtype(input_image.dtype, np.floating):
        warnings.warn(
            'Argument `image` is not of dtype `float`. Converting to `float32`.'
        )
        input_image = input_image.astype('float32')

    if not input_image.dtype == 'float32':
        warnings.warn('Converting `image` to `float32` for better convergence.')
        input_image = input_image.astype('float32')

    imfs_list = []
    imf_smoothing_window_sizes_list = []

    image_to_decompose = input_image
    extrema_scan_radius = initial_extrema_scan_radius
    while (max_modes is None) or (len(imfs_list) < max_modes):
        if verbose:
            print(f'Calculating the {len(imfs_list)}-th IMF.')
            print(f'    The local extrema search radius is {extrema_scan_radius}.')

        imf_calculation_result = _calculate_imf(
            input_image=image_to_decompose,
            extrema_scan_radius=extrema_scan_radius,
            smooth_by_which_distance=smooth_by_which_distance,
            verbose=verbose,
        )
        image_to_decompose = imf_calculation_result.residue

        if imf_calculation_result.imf is not None:
            imfs_list.append(imf_calculation_result.imf)
            imf_smoothing_window_sizes_list.append(
                imf_calculation_result.imf_smoothing_distance
            )

            # We can choose to update the extrema search radius with the calculated smoothing radius
            if increase_extrema_scan_radius_monotonically:
                extrema_scan_radius = max(
                    extrema_scan_radius,
                    imf_calculation_result.imf_smoothing_distance // 2,
                )
        else:
            break

    if verbose:
        print('\nFinished.')

    residue = image_to_decompose

    return imfs_list, residue, imf_smoothing_window_sizes_list
