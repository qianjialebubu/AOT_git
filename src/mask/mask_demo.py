from PIL import Image

if __name__ == '__main__':


    image = Image.open("1.jpg")
    image_copy = image.copy()
    # image_copy.show()
    image_new = Image.new('RGB', (32, 32), (0, 0, 0))
    # image_new2 = Image.new('1', (32, 32), '#646464')
    image_new2 = Image.new("L", (32, 32), 255)

    image_copy.paste(image_new, (16, 16), mask=image_new2)
    image_copy.save('1.png')
    # image_save = Image.open('1.png')
    # print(image_save.format, image_save.mode)
    # image_copy.show()
