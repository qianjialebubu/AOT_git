import os
dir = "D:\\file\program_data\deeplearning\AOT-GAN-for-Inpainting-master\src\dir_image\celeba-64"
filenames=os.listdir(dir)
print(filenames)
print(len(filenames))
for i in range(len(filenames)):
    print(filenames[i])

