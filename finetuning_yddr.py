import glob
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from typing import List, Dict, Any
import random


### Load images
image_path = "/home/yddr/data/SlicedPlane"
image_path_list = glob.glob(os.path.join(image_path,'*/*_seg.png'))

### Visualization funtion
def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])

    sorted_maskgen_data = sorted(mask, key=(lambda x: x['area']), reverse=True)
    ex_mask = mask[0]["segmentation"]
    img_h, img_w = ex_mask.shape[0:2] 
    overlay_img = np.ones((img_h, img_w, 4))
    overlay_img[:,:,3] = 0
    
    alpha_channel = np.zeros((img_h, img_w, 3),dtype='uint8')

    for each_gen in sorted_maskgen_data:
        # Generate outlines based on the boolean mask images
        boolean_mask = each_gen["segmentation"]
        uint8_mask = 255 * np.uint8(boolean_mask)
        mask_contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(mask_contours) == 0:
            continue

        overlay_img[boolean_mask] = [0,0,0,0.5]
        outline_opacity = 0.5
        outline_thickness = 2
        outline_color = np.concatenate([np.random.random(3), [outline_opacity]])
        cv2.polylines(overlay_img, mask_contours, True, outline_color, outline_thickness, cv2.LINE_AA)
        score = str(round(each_gen['predicted_iou'],2))
        x= int(each_gen['bbox'][0]+each_gen['bbox'][2]/2.)
        y = int(each_gen['bbox'][1]+each_gen['bbox'][3]/2.)
        cv2.putText(overlay_img,score,(x,y),cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
        cv2.putText(alpha_channel,score,(x,y),cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)

    m = cv2.cvtColor(alpha_channel,cv2.COLOR_BGR2GRAY)
    oc_mask = (m == 255)     ## boolean (W, H, 3)
    overlay_img[oc_mask] = (0,0,0,0)

    # Draw the overlay image
    ax.imshow(overlay_img)

### Make the G.T dict
dataset = {}
for ip in image_path_list:
    im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    gt_mask = (im == 255) ## If 255, particle => True    Else 0, background => False
    dataset[ip] = gt_mask


### Load Pretrained Model weight
model_type = 'vit_h'
checkpoint = '/home/yddr/code/segment-anything/weights/sam_vit_h_4b8939.pth'
device = 'cuda:0'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train();

### Preprocess the images
transformed_data = defaultdict(dict)
for k in dataset.keys():
  image = cv2.imread(k)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  transform = ResizeLongestSide(sam_model.image_encoder.img_size)
  input_image = transform.apply_image(image)
  input_image_torch = torch.as_tensor(input_image, device=device)
  transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
  
  input_image = sam_model.preprocess(transformed_image)
  original_image_size = image.shape[:2]
  input_size = tuple(transformed_image.shape[-2:])

  transformed_data[k]['image'] = input_image
  transformed_data[k]['input_size'] = input_size
  transformed_data[k]['original_image_size'] = original_image_size

### Set up the Hyperparameter 
lr = 1e-3
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

loss_fn = torch.nn.MSELoss()
keys = list(dataset.keys())

### Training
num_epochs = 300
losses = []

for epoch in range(num_epochs):
  epoch_losses = []
  # Just train on shuffled key list
  random.shuffle(keys)

  for k in keys[:30]:
    print(k)
    input_image = transformed_data[k]['image'].to(device)
    input_size = transformed_data[k]['input_size']
    original_image_size = transformed_data[k]['original_image_size']

    # No grad here as we don't want to optimise the encoders
    with torch.no_grad():
      image_embedding = sam_model.image_encoder(input_image)

      sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
    
    low_res_masks, iou_predictions = sam_model.mask_decoder(
      image_embeddings=image_embedding,
      image_pe=sam_model.prompt_encoder.get_dense_pe(),
      sparse_prompt_embeddings=sparse_embeddings,
      dense_prompt_embeddings=dense_embeddings,
      multimask_output=False,
    )
    
    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

    gt_mask_resized = torch.from_numpy(np.resize(dataset[k], (1, 1, dataset[k].shape[0], dataset[k].shape[1]))).to(device)
    gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

    loss = loss_fn(binary_mask, gt_binary_mask)
    print("one loss : ", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_losses.append(loss.item())

  losses.append(epoch_losses)
  print("==================================================================================")
  print(f'EPOCH: {epoch}')
  print(f'Mean loss: {mean(epoch_losses) }')

### Save tuned model weight
torch.save(sam_model.state_dict(), '/home/yddr/code/segment-anything/weights/tuned/tuned_model_weight.pt')
