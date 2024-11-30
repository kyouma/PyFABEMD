import math
import numpy as np
import scipy
import skimage


def fabemd(image, max_modes=None, smooth_by_which_distance: ['max', 'min']='min', update_extrema_radius=True, strict_extrema=False, eliminate_excessive_extrema=True, show_images=False, debug=False, ):
    '''A realization of Fast and Adaptive Bidimensional Empirical Mode Decomposition [1, 2]. Based on the description from [3].
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (Height, Width).
            
    max_modes : int or None, optional
        Maximum number of intrinsic mode functions (IMFs) besides the residue to compute. `None` to find all IMFs until the residue has less than 2 maxima or less than 2 minima.
            
    smooth_by_which_distance : string, optional
        Which distance between th extrema to use. Either `min` or `max`.
            
    update_extrema_radius : bool, optional
        `True` to use the distance between extrema from the previous iteration for finding extrema on the current iteration. `False` to always use the radius of 1 pixel (may not converge).
                            
    strict_extrema : bool, optional
        `True` to calculate strict extrema (very slow due to non-separability of the filter). `False` to find non-strict extrema (must be inaccurate).
                     
    eliminate_excessive_extrema : bool, optional
        Active if `strict_extrema` is set to `True`. `True` to revise the non-strict extrema and leave only the strict ones. May be slow or even cause a memory crash due if the extrema count is very large.
                                  
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
    assert smooth_by_which_distance in ['max', 'min']

    image = image.astype(float)
    
    extrema_radius = 1
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
            print(f'{len(imfs)})', end=' ')

        # Поиск экстремумов
        # Максимумы
        if strict_extrema:  # Настоящие строгие экстремумы
            footprint = np.full((2 * extrema_radius + 1,) * 2, True)
            footprint[extrema_radius, extrema_radius] = False
            max_map = residue > scipy.ndimage.maximum_filter(residue, footprint=footprint, mode='mirror')
        else:  # Нестрогие, т.к. у строгих в центре ядра фильтра дыра, из-за которой фильтр становится несепарабельным
            max_map = residue >= scipy.ndimage.maximum_filter(residue, size=(2 * extrema_radius + 1,) * 2, mode='mirror')
            if eliminate_excessive_extrema:  # Но можно затем пройтись медленным фильтром только по найденным кандидатам. Проверено: результат совпадает со строгим фильтром
                patches = skimage.util.view_as_windows(np.pad(residue, extrema_radius, 'reflect'), (2 * extrema_radius + 1,) * 2)[max_map].copy()  # Здесь может вылететь по памяти
                patches[:, extrema_radius, extrema_radius] = np.nan
                true_map = residue[max_map] > np.nanmax(patches, axis=(-1, -2))
                max_map_ = max_map.copy()
                max_map_[np.where(max_map_)[0][~true_map], np.where(max_map)[1][~true_map]] = False
                max_map = max_map_
        
        if debug:
            print('.', end=' ')
            
        # Минимумы
        if strict_extrema:  # Настоящие строгие экстремумы
            footprint = np.full((2 * extrema_radius + 1,) * 2, True)
            footprint[extrema_radius, extrema_radius] = False
            min_map = residue < scipy.ndimage.minimum_filter(residue, footprint=footprint, mode='mirror')
        else:  # Нестрогие, т.к. у строгих в центре ядра фильтра дыра, из-за которой фильтр становится несепарабельным
            min_map = residue <= scipy.ndimage.minimum_filter(residue, size=(2 * extrema_radius + 1,) * 2, mode='mirror')
            if eliminate_excessive_extrema:  # Но можно затем пройтись медленным фильтром только по найденным кандидатам. Проверено: результат совпадает со строгим фильтром
                patches = skimage.util.view_as_windows(np.pad(residue, extrema_radius, 'reflect'), (2 * extrema_radius + 1,) * 2)[min_map].copy()  # Здесь может вылететь по памяти
                patches[:, extrema_radius, extrema_radius] = np.nan
                true_map = residue[min_map] < np.nanmin(patches, axis=(-1, -2))
                min_map_ = min_map.copy()
                min_map_[np.where(min_map_)[0][~true_map], np.where(min_map_)[1][~true_map]] = False
                min_map = min_map
                
        if debug:
            print('.', end=' ')

        # Проверка на монотонность (должно быть "сумма < 3", но было лень делать обработку случая, когда минимум или максимум только один)
        if (max_map.sum() < 2) or (min_map.sum() < 2):
            break
            
        # if show_images:
        #     plt.figure(figsize=(10, 2))
        #     plt.subplot(1, 4, 1)
        #     plt.imshow(max_map, cmap=cmap)
        #     plt.subplot(1, 4, 2)
        #     plt.imshow(np.nanmax(patches, axis=(-1, -2)), cmap=cmap, vmin=image.min(), vmax=image.max())
        #     plt.subplot(1, 4, 3)
        #     plt.imshow(min_map, cmap=cmap)
        #     plt.subplot(1, 4, 4)
        #     plt.imshow(np.nanmin(patches, axis=(-1, -2)), cmap=cmap, vmin=image.min(), vmax=image.max())
        #     plt.tight_layout()
        #     plt.show()

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
    
        # if show_images:
        #     plt.figure(figsize=(5, 2))
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(upper_envelope, cmap=cmap, vmin=image.min(), vmax=image.max())
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(lower_envelope, cmap=cmap, vmin=image.min(), vmax=image.max())
        #     plt.tight_layout()
        #     plt.show()

        # Гладкие огибающие (на краях не идеально, но сойдёт)
        smooth_envelopes = []
        for envelope in [upper_envelope, lower_envelope]:
            smooth_envelopes.append(scipy.ndimage.uniform_filter(envelope, smoothing_distance, mode='nearest'))
            if debug:
                print(';', end=' ')
    
        # if show_images:
        #     plt.figure(figsize=(5, 2))
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(smooth_envelopes[0], cmap=cmap, vmin=image.min(), vmax=image.max())
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(smooth_envelopes[1], cmap=cmap, vmin=image.min(), vmax=image.max())
        #     plt.tight_layout()
        #     plt.show()

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