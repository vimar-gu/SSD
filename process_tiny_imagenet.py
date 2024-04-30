import os
import pickle
import numpy as np
from PIL import Image


with open('wnids.txt', 'r') as fp:
    wnids = fp.readlines()
wnids = [wnid.strip() for wnid in wnids]
wnid2index = {wnid: index for index, wnid in enumerate(wnids)}

train_data = {'data': [], 'target': []}
for wnid in wnids:
    image_list = os.listdir(f'train/{wnid}/images')
    for image_name in image_list:
        image = np.asarray(Image.open(f'train/{wnid}/images/{image_name}').convert('RGB')).transpose(2, 0, 1)
        train_data['data'].append(image)
        train_data['target'].append(wnid2index[wnid])
train_data['data'] = np.stack(train_data['data'])
train_data['target'] = np.array(train_data['target'])
pickle.dump(train_data, open('train.pkl', 'wb'))

val_data = {'data': [], 'target': []}
with open('val/val_annotations.txt', 'r') as fp:
    val_annos = fp.readlines()
    for val_anno in val_annos:
        image_name, wnid = val_anno.split('\t')[:2]
        image = np.asarray(Image.open(f'val/images/{image_name}').convert('RGB')).transpose(2, 0, 1)
        val_data['data'].append(image)
        val_data['target'].append(wnid2index[wnid])
val_data['data'] = np.stack(val_data['data'])
val_data['target'] = np.array(val_data['target'])
pickle.dump(val_data, open('val.pkl', 'wb'))
