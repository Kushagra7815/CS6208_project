#test_attackcarlini--- It tests the l2 attack of Carlini,Wagner paper
# on VGG19 model.

import random
import tensorflow as tf
import numpy as np
import time
import cv2

from keras.applications import vgg19
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from l2_attackcarlini import CarliniL2
from setup_VGG19 import ModelVGG19

img = cv2.imread("/Users/kushagrachatterjee/ML4D2E/CS6208_class_project/Image1.png")
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input
image = load_img(path="/Users/kushagrachatterjee/ML4D2E/CS6208_class_project/Image1.png", target_size=inputShape)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess(image)

model = ModelVGG19()
model.predict(image)
if __name__ == "__main__":
    with tf.compat.v1.Session() as sess:
        attack = CarliniL2(sess, model, batch_size=1, max_iterations=1000, confidence=0)

        timestart = time.time()
        adv = attack.attack_batch(image, "cock")
        timeend = time.time()

        # classify the perturbed image
        print("[INFO] classifying perturbed image with VGG19")
        adv = preprocess(adv)
        preds = model.predict(adv)


