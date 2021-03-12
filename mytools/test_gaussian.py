# %%
import matplotlib.pyplot as plt
import torch
from mmdet.models.utils import gen_gaussian_target
img = torch.zeros(10, 10, dtype=torch.float32)
centers = [[0, 0], [3, 3], [7, 8]]
for center in centers:
    img = gen_gaussian_target(img, center, 1)
img = img.numpy()
print(img)
plt.imshow(img)
plt.show

# %%
