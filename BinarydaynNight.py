import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import os
import random

def load_dataset(img_dir):
    im_list = []
    image_types = ["day","night"]

    for im in image_types:
        for filex in glob.glob(os.path.join(img_dir, im, "*")):
            
            img = mpimg.imread(filex)

            if not img is None:
                im_list.append((img, im))
    
    return im_list 

def encode(label):
    num = 0
    if(label == 'day'):
        num = 1

    return num

def standardize_img(img):
    img = cv2.resize(img, (1100, 600))
        
    return img   

def standardize(image_list):
    standard_list = []
    for item in Image_list:
        image = item[0]
        label = item[1]

        standard_img = standardize_img(image)
        encoded_label = encode(label)

        standard_list.append((standard_img, encoded_label))
    
    return standard_list
    
def avgbrightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sum_v = np.sum(hsv[:,:,2])
    area = 1100*600
    avg = sum_v/area

    return avg

def estimate_label(img):
    avg_bright = avgbrightness(img)
    predicted_label = 0
    threshold = 101
    if(avg_bright >= threshold):
        predicted_label = 1

    return predicted_label

def get_misclassified_imgs(test_img_list):
    misclassified_imgs = []
    for img in test_img_list:
        image = img[0]
        true_label = img[1]
        predicted_label = estimate_label(image)
        if(predicted_label != true_label):
            misclassified_imgs.append((image, true_label, predicted_label))

    return misclassified_imgs        

train_dir = "day_night_images/training/"
test_dir = "day_night_images/test/"

Image_list = load_dataset(train_dir)
standard_set = standardize(Image_list)

test_image_list = load_dataset(test_dir)
#print(len(test_image_list))
test_standard_set = standardize(test_image_list)
random.shuffle(test_standard_set)

MISCLASSIFIED = get_misclassified_imgs(test_standard_set)
total = len(standard_set)
accuracy = (total - len(MISCLASSIFIED))/total 
print(accuracy)