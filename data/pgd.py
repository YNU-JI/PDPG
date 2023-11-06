import os

import numpy as np

cat_path = '/home/special/user/jijun/Adv_img/cifar10/test/resnet50/pgd/'
count = 0
for file_name in os.listdir(cat_path):
    file_name = file_name.split('_')[2]
    print(file_name)

    count +=1
    data = np.load(cat_path+file_name)
    # split_arr = np.split(data, 2, axis=0)
    img = np.squeeze(data)
    path_dir = f'/home/special/user/jijun/Adv_img/cifar10/test/resnet50/pgdg'
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    sav_filename = f'{path_dir}/{file_name}'
    # np.save(sav_filename, img)
    # print(img.shape)
print(count)