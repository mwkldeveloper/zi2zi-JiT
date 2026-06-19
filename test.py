import numpy as np
test_data = np.load("data/sample_dataset/test.npz")
font_labels_all = test_data['font_labels']
char_labels_all = test_data['char_labels']
style_images_all = test_data['style_images']      # (N, 3, 128, 128) uint8
content_images_all = test_data['content_images']

from font2data.utils import show_images
content_images_hwc = np.transpose(content_images_all, (0, 2, 3, 1))
print(content_images_hwc[0].shape)
show_images(content_images_hwc, show=False)