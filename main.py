
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
from skimage import img_as_uint


def testGenerator(train_path, target_size=(512,512)):
    for i in range(1, 19):
        if i<10:
            img = io.imread(os.path.join(train_path,"0%d_h.jpg"%i))
        else:
            img = io.imread(os.path.join(train_path,"%d_h.jpg"%i))
        img = trans.resize(img,target_size)
        img = np.reshape(img,(1,)+img.shape)
        yield img


def saveResult(save_path, npyfile):
    for i,item in enumerate(npyfile):
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img_as_uint(item))


if __name__ == '__main__':
    model = load_model('unet_eye4.hdf5')
    testGene = testGenerator("eye")
    results = model.predict(testGene, steps=18, verbose=1)
    saveResult("results", results)

