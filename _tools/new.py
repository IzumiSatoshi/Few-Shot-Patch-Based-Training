import argparse
import cv2
import time
import os
import shutil
parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--videofile', type=str, help='path to your video file, for example --videofile C:\file\video\extract\video.mp4')
parser.add_argument('--projectname', type=str, help='name of the project to create the directories')
args = parser.parse_args()

import cv2
import numpy as np
import glob
from skimage.filters import gaussian
from skimage import img_as_ubyte
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as F
#select the path
#path = 'C:/Users' + '/' + %USERNAME%+ '/'+ '/Documents/Visions of Chaos/fewshot/Data' + '/' + args.projectname_train + '/' + '/' 'input_filtered'
path = 'C:\\Users\\Gebruiker\\Documents\\Visions of Chaos\\fewshot\\data\\' + '\\' + args.projectname + '_train\\input_filtered'
for file in glob.glob(path):
    print(file)
    im=Image.open(file, 0)
    im3 = ImageEnhance.Brightness(im)
    im1 = im1.save('C:\\Users\\Gebruiker\\Documents\\Visions of Chaos\\fewshot\\data\\' + '\\' + args.projectname + '_train\\mask'+str(file)+".png", image)
    img_number +=1