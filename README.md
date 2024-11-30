# PyFABEMD
Fast and Adaptive Bidimensional Empirical Mode Decomposition for Python.

**Installation:**

Download the module and put into the folder with your Python script or notebook.

**Required packages:**

NumPy, SciPy, scikit-image.

**Usage:**

```python
import pyfabemd  # Import the module

img = skimage.data.camera()  # Load an image

# Get all of the intrinsic mode functions (IMFs) and the residue, and the envelope smoothing window sizes for each iteration
imfs, smoothing_windows = pyfabemd.fabemd(img, max_modes=None)

# Get 25 IMFs and the residue, and the envelope smoothing window sizes for each iteration
imfs, smoothing_windows = pyfabemd.fabemd(img, max_modes=25)
```

**References:**

\[1\] S. M. A. Bhuiyan, R. R. Adhami and J. F. Khan, "A novel approach of fast and adaptive bidimensional empirical mode decomposition," 2008 IEEE International Conference on Acoustics, Speech and Signal Processing, Las Vegas, NV, USA, 2008, pp. 1313-1316, doi: 10.1109/ICASSP.2008.4517859.

\[2\] Bhuiyan, S.M.A., Adhami, R.R. & Khan, J.F. Fast and Adaptive Bidimensional Empirical Mode Decomposition Using Order-Statistics Filter Based Envelope Estimation. EURASIP J. Adv. Signal Process. 2008, 728356 (2008). https://doi.org/10.1155/2008/728356
