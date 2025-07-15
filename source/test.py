import numpy as np

if not hasattr(np, 'bool'):
    np.bool = np.bool_

from imgaug import augmenters as iaa
from PIL import Image
import matplotlib.pyplot as plt

seq = iaa.Sequential([
    iaa.Dropout(p=0.25, per_channel=True)
])

img = Image.open("000463.png")
aug_img = seq.augment_image(np.asarray(img))

fig = plt.figure(frameon=False)
fig.set_size_inches(9, 3)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(aug_img)
fig.savefig("dp-2.png", dpi=1000)
