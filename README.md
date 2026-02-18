# PyFABEMD

Implementation of Fast and Adaptive Bidimensional Empirical Mode Decomposition (FABEMD) \[1, 2\] for Python. Based on the description from \[3\]. \
Supports 2D images with shapes (Height, Width).

### Installation:

Download the module and put it into the folder with your Python script or notebook.

### Required packages:

NumPy, SciPy, scikit-image.

### Usage:

**Calculate the components:**

```python
# Load an image
import skimage
img = skimage.data.camera()

import pyfabemd  # Import the module

# Get all of the intrinsic mode functions (IMFs), the residue, and the envelope smoothing window sizes for each iteration
imfs, residue, smoothing_windows = pyfabemd.fabemd(
    img,
    max_modes=None,
    increase_extrema_scan_radius_monotonically=True,  # Make the extrema scan window grow as the smoothing window grows
)
```

**Plot the IMFs and the residue:**

```python
import matplotlib.pyplot as plt

nrows = (len(imfs) + 3) // 4
plt.figure(figsize=(14, 3 * nrows))
for i, x in enumerate(imfs + [residue]):
    plt.subplot(nrows, 4, i + 1)
    plt.imshow(x, cmap='gray')
    plt.colorbar()
    plt.title(
        (
            f'IMF #{i} (window size: {smoothing_windows[i]})'
            if i < len(imfs)
            else 'Residue'
        )
    )
plt.tight_layout()
plt.show()
```

![IMFs and residue](https://github.com/user-attachments/assets/e3366dd8-79dd-43bd-9195-3fe16e98e09c)

**Plot some statistics:**

```python
plt.figure(figsize=(5, 7))

ax1 = plt.subplot(2, 1, 1)
plt.plot(smoothing_windows, marker='x')
plt.xticks(range(len(imfs)))
plt.grid()
plt.title('The envelope smoothing windows size')

ax2 = plt.subplot(2, 1, 2)
plt.plot([x.var() for x in imfs + [residue]], marker='x', color='tab:orange')
plt.xticks(range(len(imfs) + 1), list(map(str, range(len(imfs)))) + ['Residue'])
plt.grid()
plt.title('The component variance')

ax1.sharex(ax2)
plt.tight_layout()
plt.show()
```

![IMF statistics](https://github.com/user-attachments/assets/8ca366d4-74f4-49ea-96e3-126ce48f0e37)

### References:

\[1\] S. M. A. Bhuiyan, R. R. Adhami and J. F. Khan, "A novel approach of fast and adaptive bidimensional empirical mode decomposition," 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, Las Vegas, NV, USA, 2008, pp. 1313-1316. https://doi.org/10.1109/ICASSP.2008.4517859

\[2\] Bhuiyan, S.M.A., Adhami, R.R. & Khan, J.F. Fast and Adaptive Bidimensional Empirical Mode Decomposition Using Order-Statistics Filter Based Envelope Estimation. EURASIP J. Adv. Signal Process. 2008, 728356 (2008). https://doi.org/10.1155/2008/728356

\[3\] M. U. Ahmed and D. P. Mandic, "Image fusion based on Fast and Adaptive Bidimensional Empirical Mode Decomposition," 2010 13th International Conference on Information Fusion, Edinburgh, UK, 2010, pp. 1-6. https://doi.org/10.1109/ICIF.2010.5711841
