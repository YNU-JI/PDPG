import os

data_dir = 'experiments/test_inpainting_places2_230328_003036/results/test/0/'
out_dir = 'experiments/test_inpainting_places2_230328_003036/results/test/out_directory'
gt_dir = 'experiments/test_inpainting_places2_230327_144535/results/test/gt_directory'


cifar_test = '/home/special/user/jijun/Adv_img/cifar10/test/resnet50/test/resnet50'
clean_dir = '/home/special/user/jijun/Adv_img/cifar10/test/resnet50/test/'

pgd_dir = '/home/special/user/jijun/Adv_img/cifar10/test/resnet50/pgd'
pgd_out_dir = '/home/special/user/jijun/Adv_img/cifar10/test/resnet50/pgd'

# for file_name in os.listdir(data_dir):
#     # print("start")
#     if file_name.startswith('Out'):
#         src_path = os.path.join(data_dir, file_name)
#         dst_path = os.path.join(out_dir, file_name[4])
#         if not os.path.exists(dst_path):
#             os.makedirs(dst_path)
#         dst_path = os.path.join(dst_path, file_name[6:])
#         os.rename(src_path, dst_path)
#     elif file_name.startswith('GT'):
#         src_path = os.path.join(data_dir, file_name)
#         dst_path = os.path.join(gt_dir, file_name[3])
#         if not os.path.exists(dst_path):
#             os.makedirs(dst_path)
#         dst_path = os.path.join(dst_path, file_name[5:])
#         os.rename(src_path, dst_path)
file_path = "/home/special/user/jijun/PyTorchProject/Save_Adv_image/filenames.txt"  # 文件的路径
with open(file_path, "r") as file:
    # 读取文件内容
    content = file.read()
# print(content)
count =0
for file_name in os.listdir(pgd_dir):
    file_name_parts = file_name.split('.')
    file_name = file_name_parts[0]
    dir_name = file_name.split('_')[2]


    src_path = os.path.join(pgd_dir, file_name)
    dst_path = os.path.join(pgd_out_dir)
    # print(file_name[0])
    # if not os.path.exists(dst_path):
    #     os.makedirs(dst_path)
    dst_path = os.path.join(dst_path, file_name)
    # os.rename(src_path, dst_path)


    # 打开文件

    if dir_name in ('1679'):

        # os.rename(src_path+'.npy', dst_path+'.npy')
        print(file_name, file_name_parts[0],file_name_parts[1], dir_name, file_name)
        count += 1
        # print(dst_path)
    # break
print(count)
# import os

# path = '../CIFAR_data/train/3_33858.png'


