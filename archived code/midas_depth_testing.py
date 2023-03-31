import open3d as o3d
import numpy as np

import cv2 as cv
import time

import imageio.v3 as iio


import torch
import matplotlib.pyplot as plt

pathToImage ='./img_sets/square_dataset/images/'
exportPath = './outputs/depthmaps/'
savePrependTag = 'depth_'
IMAGE_NAME = '1677066767959_L'
IMAGE_TYPE = '.jpeg'
model_type = "MiDaS"
# select a version of the MiDaS model
# model_type = "DPT_Large"
# model_type = "DPT_Hybrid"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# select gpu if it is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# load transforms to resize/normalise image depending on the model
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Hybrid" or model_type == "DPT_Hybrid" or model_type == "MiDaS":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# load in the image and convert colour and apply transforms
input_image = cv.imread('%s%s%s'%(pathToImage, IMAGE_NAME, IMAGE_TYPE))
img = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
input_batch = transform(img).to(device)

# predict depth and then resize to the original resolution
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# present results
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 2),
                                    sharex=True, sharey=True)
for aa in (ax1, ax2):
    aa.set_axis_off()

ax1.imshow(input_image)
ax1.set_title('Source image')
ax2.imshow(output)
ax2.set_title('Depth image')

plt.savefig(("%s_%s.jpeg"%("midas_depth", IMAGE_NAME)))
plt.show()

# depth_image = cv.cvtColor(output, cv.COLOR_BGR2GRAY)

# buildModel(input_image, output)

# Read depth and color image:
rgb_image = input_image

cv.imwrite(exportPath + savePrependTag + IMAGE_NAME + IMAGE_TYPE, output)

# Display depth and grayscale image:
fig, axs = plt.subplots(1, 2)
axs[0].imshow(output, cmap="gray")
axs[0].set_title('Depth image')
axs[1].imshow(rgb_image)
axs[1].set_title('Source image')
plt.savefig(exportPath + savePrependTag + IMAGE_NAME + IMAGE_TYPE, dpi=1200)
plt.show()

# Raw plot
# plt.plot(output)
# plt.show()