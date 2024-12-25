import typing

import numpy as np
import scipy


RAM_LIMIT = 1  # split image patches into batches with the batch size limited by this amount of GB


def _find_local_extrema(image, extrema_type: ['max', 'min'], extrema_radius):
    if extrema_type == 'min':
        image = -image

    # Initialize service variables
    ndims = len(image.shape)
    window_sizes = (2 * extrema_radius + 1,) * ndims
    center_ix = (extrema_radius,) * ndims
    window_axes = tuple(range(-1, -1 - ndims, -1))
    padding_cval = image.min()

    # Search for non-strict extrema with the separable extrema filter, and then check and filter detected points
    extrema_map = image >= scipy.ndimage.maximum_filter(image, size=window_sizes, mode='constant', cval=padding_cval)
    patches = np.lib.stride_tricks.sliding_window_view(np.pad(image, extrema_radius, 'constant', constant_values=padding_cval), window_sizes)

    ram_limit = int(RAM_LIMIT * 1024**3)

    # Process non-strict local extrema in batches, check for strictness and remove not strict ones
    extrema_ix = np.where(extrema_map)
    extrema_ix = list(zip(*extrema_ix))
    batch_size = max(1, int(np.floor(ram_limit / np.prod(window_sizes, dtype=float) / patches.itemsize)))
    extrema_ix = [extrema_ix[i : i + batch_size] for i in range(0, len(extrema_ix), batch_size)]
    extrema_ix = [tuple(map(np.array, zip(*ix))) for ix in extrema_ix]
    for ix in extrema_ix:
        patches_ = patches[ix]  # Here the program may crash by RAM

        windowed_comparison_map = image[ix][(...,) + (np.newaxis,) * ndims] > patches_
        windowed_comparison_map[(...,) + center_ix] = True
        true_map = windowed_comparison_map.all(axis=window_axes)

        extrema_map[tuple(x[~true_map] for x in ix)] = False

    return extrema_map


def fabemd(
    image: np.ndarray, max_modes: typing.Optional[int]=None,
    initial_extrema_radius: int=1,
    smooth_by_which_distance: ['max', 'min']='min',
    extrema_radius_grows_monotonically: bool=False,
    debug: bool=False
) -> tuple[np.ndarray, list[int]]:
    """
    Apply Fast and Adaptive Bidimensional Empirical Mode Decomposition [1, 2] algorithm to the image.

    The implementation is based on the description from [3].

    References
    ----------
    [1] S. M. A. Bhuiyan, R. R. Adhami and J. F. Khan, "A novel approach of fast and adaptive bidimensional empirical mode decomposition," 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, Las Vegas, NV, USA, 2008, pp. 1313-1316. https://doi.org/10.1109/ICASSP.2008.4517859.

    [2] Bhuiyan, S.M.A., Adhami, R.R. & Khan, J.F. Fast and Adaptive Bidimensional Empirical Mode Decomposition Using Order-Statistics Filter Based Envelope Estimation. EURASIP J. Adv. Signal Process. 2008, 728356 (2008). https://doi.org/10.1155/2008/728356

    [3] M. U. Ahmed and D. P. Mandic, "Image fusion based on Fast and Adaptive Bidimensional Empirical Mode Decomposition," 2010 13th International Conference on Information Fusion, Edinburgh, UK, 2010, pp. 1-6. https://doi.org/10.1109/ICIF.2010.5711841.

    Parameters
    ----------
    image : np.ndarray
        Input N-dimensional image.
    max_modes : int or None, optional
        Maximum number of intrinsic mode functions (IMFs) to compute, besides the residue. Unlimited if None. The default is None.
    initial_extrema_radius : int, optional
        Initial radius to scan for local extrema. The default is 1.
    smooth_by_which_distance : ['max', 'min'], optional
        Which distance between the nearest extrema to use for smoothing. The default is 'min'.
    extrema_radius_grows_monotonically : bool, optional
        Update the local extrema scan radius with the calculated smoothing window size, or always use the initial radius. The default is False.
    debug : bool, optional
        Print progress report during steps execution. The default is False.

    Returns
    -------
    imfs : np.ndarray
        The array of IMFs and the residue, with the shape (number of IMFs + 1, <input image shape>).
    smoothing_window_sizes : list[int]
        The list of envelope smoothing window sizes for each iteration.
    """
    assert initial_extrema_radius > 0
    assert smooth_by_which_distance in ['max', 'min']

    extrema_radius = initial_extrema_radius

    imfs = []
    smoothing_window_sizes = []

    residue = image
    while max_modes is None or len(imfs) < max_modes:
        if debug:
            print(f'Calculating the {len(imfs)}-th IMF.')
            print(f'    The local extrema search radius is {extrema_radius}.')

        # Find local extrema
        max_map = _find_local_extrema(residue, 'max', extrema_radius)
        if debug:
            print(f'    {max_map.sum()} local maxima have been found.')
        min_map = _find_local_extrema(residue, 'min', extrema_radius)
        if debug:
            print(f'    {min_map.sum()} local minima have been found.')

        # Check for the stopping criterion
        # if (max_map.sum() < 2) or (min_map.sum() < 2):
        if max_map.sum() + min_map.sum() < 3:
            if debug:
                print('\nFinished.')
            break

        # Calculate the distances between the local extrema
        if debug:
            print('    Calculating the distances between the local extrema... ', end='')
        smoothing_distance = []
        for map_ in [max_map, min_map]:
            coords = np.array(np.where(map_)).T
            if coords.shape[0] > 1:
                distances = scipy.spatial.KDTree(coords).query(coords, k=2, workers=-1)[0][:, -1]  # Without KDTree may crash by RAM
                smoothing_distance.append(distances.max() if smooth_by_which_distance == 'max' else distances.min())
            if debug:
                print('Done', end=' and... ' if map_ is max_map else '\n')

        smoothing_distance = max(smoothing_distance) if smooth_by_which_distance == 'max' else min(smoothing_distance)
        smoothing_distance = 2 * int(np.ceil(smoothing_distance / 2)) + 1
        smoothing_window_sizes.append(smoothing_distance)
        if debug:
            print(f'    The new smoothing window size is {smoothing_distance}.')

        # We can choose to update the extrema search radius with the calculated smoothing radius
        if extrema_radius_grows_monotonically:
            extrema_radius = smoothing_distance // 2

        # Calculate coarse envolopes
        if debug:
            print('    Calculating the coarse upper envelope... ', end='')
        upper_envelope = scipy.ndimage.maximum_filter(residue, smoothing_distance, mode='nearest')
        if debug:
            print('Done')
        if debug:
            print('    Calculating the coarse lower envelope... ', end='')
        lower_envelope = scipy.ndimage.minimum_filter(residue, smoothing_distance, mode='nearest')
        if debug:
            print('Done')

        # Calculate smooth envelopes (may have some issues on the border of the image)if debug:
        if debug:
            print('    Smoothing the envelopes... ', end='')
        smooth_envelopes = []
        for envelope in [upper_envelope, lower_envelope]:
            smooth_envelopes.append(scipy.ndimage.uniform_filter(envelope, smoothing_distance, mode='nearest'))
            if debug:
                print('Done', end=' and... ' if envelope is upper_envelope else '\n')

        new_residue = np.mean(smooth_envelopes, axis=0)
        new_imf = residue - new_residue

        imfs.append(new_imf)
        residue = new_residue

        if debug:
            print()

    imfs.append(residue)
    imfs = np.array(imfs)
    return imfs, smoothing_window_sizes
