import math
import numpy as np
import scipy
import skimage


def _find_local_extrema(image, extrema_type: ['max', 'min'], extrema_radius, mode: ['strict_by_filtering', 'strict_by_comparison', 'nonstrict_then_update'], submode):
    if extrema_type == 'min':
        image = -image
    
    ndims = len(image.shape)
    window_sizes = (2 * extrema_radius + 1,) * ndims
    center_ix = (extrema_radius,) * ndims
    window_axes = tuple(range(-1, -1 - ndims, -1))
    padding_cval = image.min() #np.finfo(image.dtype).min
    
    if mode == 'strict_by_filtering':  # Проверка на строгий экстремум через фильтрацию с выколотой точкой
        if submode == 1:
            footprint = np.full(window_sizes, True)
            footprint[center_ix] = False
            extrema_map = image > scipy.ndimage.maximum_filter(image, footprint=footprint, mode='constant', cval=padding_cval)  # Здесь может вылететь по памяти
            
        elif submode == 2:  # Серия частично избыточных сепарабельных фильтров
            partial_maps = []
            for dim in range(ndims):
                footprint1 = np.full(tuple(window_sizes[i] if i == dim else 1 for i in range(ndims)), True)
                footprint1[tuple(center_ix[i] if i == dim else 0 for i in range(ndims))] = False
                pmap = scipy.ndimage.maximum_filter(image, footprint=footprint1, mode='constant', cval=padding_cval)
                footprint2 = np.full(tuple(window_sizes[i] if i != dim else 1 for i in range(ndims)), True)
                pmap = scipy.ndimage.maximum_filter(pmap, footprint=footprint2, mode='constant', cval=padding_cval)
                partial_maps.append(pmap)
            
            extrema_map = image > np.max(partial_maps, axis=0)

        elif submode == 3:  # Серия частично избыточных сепарабельных фильтров
            partial_maps = []
            for dim in range(ndims):
                footprint = np.full(window_sizes[dim], True)
                footprint[center_ix[dim]] = False
                pmap = scipy.ndimage.maximum_filter(image, footprint=footprint, mode='constant', cval=padding_cval, axes=(dim,))
                pmap = scipy.ndimage.maximum_filter(pmap, size=tuple(window_sizes[i] for i in range(ndims) if i != dim), mode='constant', cval=padding_cval, axes=tuple(i for i in range(ndims) if i != dim))
                partial_maps.append(pmap)
            
            extrema_map = image > np.max(partial_maps, axis=0)
        
    elif mode == 'strict_by_comparison':  # Проверка на строгий экстремум через сравнение с пикселями окрестности
        # patches = skimage.util.view_as_windows(np.pad(image, extrema_radius, 'constant', constant_values=padding_cval), window_sizes)
        patches = np.lib.stride_tricks.sliding_window_view(np.pad(image, extrema_radius, 'constant', constant_values=padding_cval), window_sizes)
        windowed_comparison_map = image[(...,) + (np.newaxis,) * ndims] > patches  # Здесь может вылететь по памяти
        
        if submode == 1:  # Проверка на "больше всех" через сумму True
            extrema_map = windowed_comparison_map.sum(axis=window_axes) == np.prod(window_sizes) - 1
            
        elif submode == 2:  # Проверка на "больше всех" через all() после замены центрального элемента окна отношений на True с помощью бинарной операции
            footprint = np.full(window_sizes, False)
            footprint[center_ix] = True
            footprint = footprint[(np.newaxis,) * (len(windowed_comparison_map.shape) - ndims) + (...,)]
            extrema_map = (windowed_comparison_map | footprint).all(axis=window_axes)
            
        elif submode == 3:  # Проверка на "больше всех" через all() после замены центрального элемента окна отношений на True в самом массиве
            windowed_comparison_map[(...,) + center_ix] = True
            extrema_map = windowed_comparison_map.all(axis=window_axes)
    
    elif mode == 'nonstrict_then_update':  # Проверка на нестрогий экстремум через сепарабельную фильтрацию и дальнейшую корректировку найденных элементов 
        extrema_map = image >= scipy.ndimage.maximum_filter(image, size=window_sizes, mode='constant', cval=padding_cval)
        # patches = skimage.util.view_as_windows(np.pad(image, extrema_radius, 'constant', constant_values=padding_cval), window_sizes)
        patches = np.lib.stride_tricks.sliding_window_view(np.pad(image, extrema_radius, 'constant', constant_values=padding_cval), window_sizes)

        ram_limit = int(0.5 * 1024**3)
        
        extrema_ix = np.where(extrema_map)
        extrema_ix = list(zip(*extrema_ix))
        batch_size = max(1, int(np.floor(ram_limit / np.prod(window_sizes, dtype=float) / patches.itemsize)))
        extrema_ix = [extrema_ix[i : i + batch_size] for i in range(0, len(extrema_ix), batch_size)]
        extrema_ix = [tuple(map(np.array, zip(*ix))) for ix in extrema_ix]
        for ix in extrema_ix:
            patches_ = patches[ix]  # Здесь может вылететь по памяти
    
            if submode == 1:
                patches_ = patches_.copy()
                patches_[(...,) + center_ix] = np.nan
                true_map = image[ix] > np.nanmax(patches_, axis=window_axes)
                    
            elif submode in [2, 3, 4]:
                windowed_comparison_map = image[ix][(...,) + (np.newaxis,) * ndims] > patches_
                if submode == 2:
                    true_map = windowed_comparison_map.sum(axis=window_axes) == np.prod(window_sizes) - 1
                    
                elif submode == 3:
                    footprint = np.full(window_sizes, False)
                    footprint[center_ix] = True
                    footprint = footprint[(np.newaxis,) * (len(windowed_comparison_map.shape) - ndims) + (...,)]
                    true_map = (windowed_comparison_map | footprint).all(axis=window_axes)
                    
                elif submode == 4:
                    windowed_comparison_map[(...,) + center_ix] = True
                    true_map = windowed_comparison_map.all(axis=window_axes)
    
            extrema_map[tuple(x[~true_map] for x in ix)] = False
        
    return extrema_map


def fabemd(image, max_modes=None,
           initial_extrema_radius=1, smooth_by_which_distance: ['max', 'min']='min', update_extrema_radius=True,
           speed_over_memory=True, debug=False):
    '''Implementation of Fast and Adaptive Bidimensional Empirical Mode Decomposition [1, 2]. Based on the description from [3].
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (Height, Width).
            
    max_modes : int or None, optional
        Maximum number of intrinsic mode functions (IMFs) to compute, besides the residue. Use `None` to find all IMFs until the residue has less than 2 maxima or less than 2 minima.

    initial_extrema_radius : int, optional
        Initial radius to scan for local extrema.
        
    smooth_by_which_distance : string, optional
        Which distance between the nearest extrema to use, either `min` or `max`.
            
    update_extrema_radius : bool, optional
        `True` to use the distance between extrema from the previous iteration for finding extrema on the current iteration. `False` to always use the radius of 1 pixel (may not converge).

    speed_over_memory : bool, optional
        `True` to use more memory expensive, but faster method.

    debug : bool, optional
        Print progress marks, extrema counts and window size.

    Returns
    -------
    imfs : List[numpy.ndarray]
        The list of IMFs and the residue.

    smoothing_windows : List[int]
        The list of envelope smoothing window sizes for each iteration.
        
    References:
    [1] S. M. A. Bhuiyan, R. R. Adhami and J. F. Khan, "A novel approach of fast and adaptive bidimensional empirical mode decomposition," 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, Las Vegas, NV, USA, 2008, pp. 1313-1316, doi: 10.1109/ICASSP.2008.4517859.
    [2] Bhuiyan, S.M.A., Adhami, R.R. & Khan, J.F. Fast and Adaptive Bidimensional Empirical Mode Decomposition Using Order-Statistics Filter Based Envelope Estimation. EURASIP J. Adv. Signal Process. 2008, 728356 (2008). https://doi.org/10.1155/2008/728356
    [3] M. U. Ahmed and D. P. Mandic, "Image fusion based on Fast and Adaptive Bidimensional Empirical Mode Decomposition," 2010 13th International Conference on Information Fusion, Edinburgh, UK, 2010, pp. 1-6, doi: 10.1109/ICIF.2010.5711841.
    '''
    assert initial_extrema_radius > 0
    assert smooth_by_which_distance in ['max', 'min']

    # image = image.astype(np.float32)
    
    extrema_radius = initial_extrema_radius
    cmap = 'gray'
        
    imfs = []
    smoothing_windows = []
    residue = image
    while max_modes is None or len(imfs) < max_modes:
        if debug:
            print(f'{len(imfs)}) {extrema_radius}', end=' ')

        # Поиск экстремумов
        max_map = _find_local_extrema(residue, 'max', extrema_radius, *(('nonstrict_then_update', 4) if speed_over_memory else ('strict_by_filtering', 3)))
        if debug:
            print('.', end=' ')
        min_map = _find_local_extrema(residue, 'min', extrema_radius, *(('nonstrict_then_update', 4) if speed_over_memory else ('strict_by_filtering', 3)))
        if debug:
            print('.', end=' ')

        # Проверка на монотонность (должно быть "сумма < 3", но было лень делать обработку случая, когда минимум или максимум только один)
        if (max_map.sum() < 2) or (min_map.sum() < 2):
            if debug:
                print()
            break

        # Вычисление дистанций между экстремумами
        smoothing_distance = []
        for map_ in [max_map, min_map]:
            coords = np.array(np.where(map_)).T
            if debug:
                print(coords.shape, end=' ')

            distances = scipy.spatial.KDTree(coords).query(coords, k=2, workers=-1)[0][:, -1]  # Если просто считать расстояния крест-накрест, то может вылететь по памяти
            if debug:
                print(',', end=' ')
            smoothing_distance.append(distances.max() if smooth_by_which_distance == 'max' else distances.min())
        smoothing_distance = max(smoothing_distance) if smooth_by_which_distance == 'max' else min(smoothing_distance)
        smoothing_distance = 2 * math.ceil(smoothing_distance / 2) + 1
        smoothing_windows.append(smoothing_distance)
        if debug:
            print(smoothing_distance, end=' ')

        # Опционально можно задать для будущих экстремумов новый размер окна (как в белой презентации)
        if update_extrema_radius:
            extrema_radius = smoothing_distance // 2

        # Грубые огибающие
        upper_envelope = scipy.ndimage.maximum_filter(residue, smoothing_distance, mode='nearest')
        if debug:
            print('.', end=' ')
        lower_envelope = scipy.ndimage.minimum_filter(residue, smoothing_distance, mode='nearest')
        if debug:
            print('.', end=' ')

        # Гладкие огибающие (на краях не идеально, но сойдёт)
        smooth_envelopes = []
        for envelope in [upper_envelope, lower_envelope]:
            smooth_envelopes.append(scipy.ndimage.uniform_filter(envelope, smoothing_distance, mode='nearest'))
            if debug:
                print(';', end=' ')

        # Вычисление моды и остатка
        new_residue = np.mean(smooth_envelopes, axis=0)
        # new_residue = scipy.ndimage.uniform_filter((upper_envelope + lower_envelope) / 2, smoothing_distance, mode='nearest')
        new_imf = residue - new_residue
        
        imfs.append(new_imf)
        residue = new_residue
        
        if debug:
            print()
    
    imfs.append(residue)
    return imfs, smoothing_windows


def _fabemd(image, max_modes=None,
           initial_extrema_radius=1, smooth_by_which_distance: ['max', 'min']='min', update_extrema_radius=True,
           strict_extrema=False, eliminate_excessive_extrema=True,
           show_images=False, debug=False):
    '''Implementation of Fast and Adaptive Bidimensional Empirical Mode Decomposition [1, 2]. Based on the description from [3].
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (Height, Width).
            
    max_modes : int or None, optional
        Maximum number of intrinsic mode functions (IMFs) to compute, besides the residue. Use `None` to find all IMFs until the residue has less than 2 maxima or less than 2 minima.

    initial_extrema_radius : int, optional
        Initial radius to scan for local extrema.
        
    smooth_by_which_distance : string, optional
        Which distance between the nearest extrema to use, either `min` or `max`.
            
    update_extrema_radius : bool, optional
        `True` to use the distance between extrema from the previous iteration for finding extrema on the current iteration. `False` to always use the radius of 1 pixel (may not converge).
                            
    strict_extrema : bool, optional
        `True` to calculate strict extrema (very slow due to non-separability of the filter). `False` to find non-strict extrema (theoretically must be inaccurate).
                     
    eliminate_excessive_extrema : bool, optional
        Active only if `strict_extrema` is also set to `True`. `True` to revise the non-strict extrema and leave only the strict ones. May be slow or even cause a memory crash if the extrema count is very large.
                                  
    show_images : bool, optional
        Show the IMFs and the residue for debugging purposes.

    debug : bool, optional
        Print progress marks, extrema counts and window size.

    Returns
    -------
    imfs : List[numpy.ndarray]
        The list of IMFs and the residue.

    smoothing_windows : List[int]
        The list of envelope smoothing window sizes for each iteration.
        
    References:
    [1] S. M. A. Bhuiyan, R. R. Adhami and J. F. Khan, "A novel approach of fast and adaptive bidimensional empirical mode decomposition," 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, Las Vegas, NV, USA, 2008, pp. 1313-1316, doi: 10.1109/ICASSP.2008.4517859.
    [2] Bhuiyan, S.M.A., Adhami, R.R. & Khan, J.F. Fast and Adaptive Bidimensional Empirical Mode Decomposition Using Order-Statistics Filter Based Envelope Estimation. EURASIP J. Adv. Signal Process. 2008, 728356 (2008). https://doi.org/10.1155/2008/728356
    [3] M. U. Ahmed and D. P. Mandic, "Image fusion based on Fast and Adaptive Bidimensional Empirical Mode Decomposition," 2010 13th International Conference on Information Fusion, Edinburgh, UK, 2010, pp. 1-6, doi: 10.1109/ICIF.2010.5711841.
    '''
    assert initial_extrema_radius > 0
    assert smooth_by_which_distance in ['max', 'min']

    # image = image.astype(np.float32)
    
    extrema_radius = initial_extrema_radius
    cmap = 'gray'
    
    if show_images:
        plt.figure(figsize=(3, 2))
        plt.imshow(image, cmap=cmap)
        plt.tight_layout()
        plt.show()
    
    imfs = []
    smoothing_windows = []
    residue = image
    while max_modes is None or len(imfs) < max_modes:
        if debug:
            print(f'{len(imfs)}) {extrema_radius}', end=' ')

        # Поиск экстремумов
        # Максимумы
        if strict_extrema:  # Настоящие строгие экстремумы
            footprint = np.full((2 * extrema_radius + 1,) * 2, True)
            footprint[extrema_radius, extrema_radius] = False
            max_map = residue > scipy.ndimage.maximum_filter(residue, footprint=footprint, mode='constant', cval=residue.min())
        else:  # Нестрогие, т.к. у строгих в центре ядра фильтра дыра, из-за которой фильтр становится несепарабельным
            max_map = residue >= scipy.ndimage.maximum_filter(residue, size=(2 * extrema_radius + 1,) * 2, mode='constant', cval=residue.min())
            if eliminate_excessive_extrema:  # Но можно затем пройтись медленным фильтром только по найденным кандидатам. Проверено: результат совпадает со строгим фильтром
                patches = skimage.util.view_as_windows(np.pad(residue, extrema_radius, 'constant', constant_values=residue.min()), (2 * extrema_radius + 1,) * 2)[max_map].copy()  # Здесь может вылететь по памяти
                patches[:, extrema_radius, extrema_radius] = np.nan
                true_map = residue[max_map] > np.nanmax(patches, axis=(-1, -2))
                max_map_ = max_map.copy()
                max_map_[np.where(max_map_)[0][~true_map], np.where(max_map)[1][~true_map]] = False
                max_map = max_map_
                del max_map_
        
        if debug:
            print('.', end=' ')
            
        # Минимумы
        if strict_extrema:  # Настоящие строгие экстремумы
            footprint = np.full((2 * extrema_radius + 1,) * 2, True)
            footprint[extrema_radius, extrema_radius] = False
            min_map = residue < scipy.ndimage.minimum_filter(residue, footprint=footprint, mode='constant', cval=residue.max())
        else:  # Нестрогие, т.к. у строгих в центре ядра фильтра дыра, из-за которой фильтр становится несепарабельным
            min_map = residue <= scipy.ndimage.minimum_filter(residue, size=(2 * extrema_radius + 1,) * 2, mode='constant', cval=residue.max())
            if eliminate_excessive_extrema:  # Но можно затем пройтись медленным фильтром только по найденным кандидатам. Проверено: результат совпадает со строгим фильтром
                patches = skimage.util.view_as_windows(np.pad(residue, extrema_radius, 'constant', constant_values=residue.max()), (2 * extrema_radius + 1,) * 2)[min_map].copy()  # Здесь может вылететь по памяти
                patches[:, extrema_radius, extrema_radius] = np.nan
                true_map = residue[min_map] < np.nanmin(patches, axis=(-1, -2))
                min_map_ = min_map.copy()
                min_map_[np.where(min_map_)[0][~true_map], np.where(min_map_)[1][~true_map]] = False
                min_map = min_map_
                del min_map_
                
        if debug:
            print('.', end=' ')

        # Проверка на монотонность (должно быть "сумма < 3", но было лень делать обработку случая, когда минимум или максимум только один)
        if (max_map.sum() < 2) or (min_map.sum() < 2):
            if debug:
                print()
            break

        # Вычисление дистанций между экстремумами
        smoothing_distance = []
        for map_ in [max_map, min_map]:
            coords = np.array(np.where(map_)).T
            if debug:
                print(coords.shape, end=' ')

            distances = scipy.spatial.KDTree(coords).query(coords, k=2, workers=-1)[0][:, -1]  # Если просто считать расстояния крест-накрест, то может вылететь по памяти
            if debug:
                print(',', end=' ')
            smoothing_distance.append(distances.max() if smooth_by_which_distance == 'max' else distances.min())
        smoothing_distance = max(smoothing_distance) if smooth_by_which_distance == 'max' else min(smoothing_distance)
        smoothing_distance = 2 * math.ceil(smoothing_distance / 2) + 1
        smoothing_windows.append(smoothing_distance)
        if debug:
            print(smoothing_distance, end=' ')

        # Опционально можно задать для будущих экстремумов новый размер окна (как в белой презентации)
        if update_extrema_radius:
            extrema_radius = smoothing_distance // 2

        # Грубые огибающие
        upper_envelope = scipy.ndimage.maximum_filter(residue, smoothing_distance, mode='nearest')
        if debug:
            print('.', end=' ')
        lower_envelope = scipy.ndimage.minimum_filter(residue, smoothing_distance, mode='nearest')
        if debug:
            print('.', end=' ')

        # Гладкие огибающие (на краях не идеально, но сойдёт)
        smooth_envelopes = []
        for envelope in [upper_envelope, lower_envelope]:
            smooth_envelopes.append(scipy.ndimage.uniform_filter(envelope, smoothing_distance, mode='nearest'))
            if debug:
                print(';', end=' ')

        # Вычисление моды и остатка
        new_residue = np.mean(smooth_envelopes, axis=0)
        new_imf = residue - new_residue
    
        if show_images:
            plt.figure(figsize=(5, 2))
            plt.subplot(1, 2, 1)
            plt.imshow(new_imf, cmap=cmap)
            plt.subplot(1, 2, 2)
            plt.imshow(new_imf + (image.min() + image.max()) / 2, cmap='gray', vmin=image.min(), vmax=image.max())
            plt.tight_layout()
            plt.show()
        
            plt.figure(figsize=(3, 2))
            plt.imshow(new_residue, cmap=cmap)
            plt.tight_layout()
            plt.show()
        
        imfs.append(new_imf)
        residue = new_residue
        
        if debug:
            print()
            
    imfs.append(residue)
    return imfs, smoothing_windows