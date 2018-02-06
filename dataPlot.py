import dataLoad
import matplotlib.pyplot as plt
import numpy as np

def pic_show(picture, number = 25):

    fig, axes = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(30,8))
    imgs = picture[:number]

    for image, row in zip([imgs[:5],
                           imgs[5:10],
                           imgs[10:15],
                           imgs[15:20],
                           imgs[20:25]], axes):

        for img, ax in zip(image, row):
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout(pad=0.1)
    plt.show()

pic = dataLoad.picture
pic_show(pic)
plt.show()