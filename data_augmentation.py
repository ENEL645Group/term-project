import os
from os import listdir
from pathlib import Path
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pylab as plt
import time

path = Path("dataset/")

for dir in os.listdir(path):

    onlyfiles = [f for f in listdir(path / dir)]

    count = 0
    for file in onlyfiles:    
        file_path = path / dir / file
        print(file_path)
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

        if img is not None:

            if "jose" not in file:

                print(img.shape)
                img = cv2.resize(img, (256, 256))
                im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # padding the images because some of them have the values and the suit really close to the border
                # and they could be lost when rotating
                im_rgb = np.pad(im_rgb, [(20, 20),(20, 20),(0, 0)], 'constant', constant_values=(255))

                #define the augmentation params
                data_generator = ImageDataGenerator(brightness_range=(1.0, 3.0), rotation_range=180)
                # data_generator = ImageDataGenerator(rotation_range=90, shear_range=15.0)

                imgs = np.expand_dims(im_rgb, axis=0)
                data_generator.fit(imgs)
                image_iterator = data_generator.flow(imgs, batch_size=1)

                print("transforming {}".format(file))

                # print(imgs.shape)
                # plt.imshow(np.squeeze(imgs))
                # plt.show()

                #for each image in the folder, generate 9 new augmented images
                for x in range(9):
                    img_transformed = image_iterator.next()
                    img_transformed = np.squeeze(img_transformed)
                    # print(img_transformed.shape)
                    filename = "jose_{}.jpg".format(count)
                    path_to_save = str(path / dir / filename)
                    print("saving img to {}".format(path_to_save))
                    im_rgb = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(path_to_save, im_rgb)
                    count = count + 1
                    # time.sleep(2)
