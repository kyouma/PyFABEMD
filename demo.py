import pathlib

import skimage

img = skimage.data.camera()

import pyfabemd  # Import the module

# Get all of the intrinsic mode functions (IMFs), the residue, and the envelope smoothing window sizes for each iteration
imfs, residue, smoothing_windows = pyfabemd.fabemd(
    img,
    max_modes=None,
    increase_extrema_scan_radius_monotonically=True,  # Make the extrema scan window grow as the smoothing window grows
)

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
plt.savefig(pathlib.Path('~/Documents/imfs.png').expanduser())
plt.show()


plt.figure(figsize=(7, 7))

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
plt.savefig(pathlib.Path('~/Documents/stats.png').expanduser())
plt.show()
