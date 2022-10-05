from PIL import Image
import os
def make_mask(dir_image,image_name,dir_image_save):
    mask_size = 256

    dir = os.path.join(dir_image_save, image_name)
    image = Image.open(dir_image)
    image_copy = image.copy()
    # image_copy.show()
    image_new = Image.new('RGB', (int(2.5 * mask_size), int(1.25 * mask_size)), (0, 0, 0))
    # image_new2 = Image.new('1', (32, 32), '#646464')
    image_new2 = Image.new("L", (int(2.5 * mask_size), int(1.25 * mask_size)), 255)

    image_copy.paste(image_new, (int(1.125 * mask_size), int(2.5 * mask_size)), mask=image_new2)
    image_copy.save(dir)
    # image_save = Image.open(dir)
    # print(image_save.format, image_save.mode)
def main():
    # dir_project = "/data3/qianjiale_dataset/"
    # dir_image = "/data3/qianjiale_dataset/celeba-512"
    # dir_image_t = os.path.join(dir_project, dir_image)
    dir_image_t = "/data3/qianjiale_dataset/celeba-1024"
    dir_image_save = "/data3/qianjiale_dataset/celeba_mask_1024"
    # dir_image_save_t = os.path.join(dir_project, dir_image_save)
    filenames = os.listdir(dir_image_t)
    for i in range(len(filenames)):
        dir = os.path.join(dir_image_t, filenames[i])

        make_mask(dir,filenames[i],dir_image_save)
        if i%1000 ==0:
            print(filenames[i])
    print("succeed")
if __name__ == '__main__':
    main()
