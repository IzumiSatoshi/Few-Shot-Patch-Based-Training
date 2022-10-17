import os
import argparse
import multiprocessing as mp
import torch
import numpy as np
import time
import random
import numba
from numba import cuda

parser = argparse.ArgumentParser(description='arguments')

##tool_disflow
#parser.add_argument('--inputdir', type=str, help='put a projectname_gen/input_directory path here')
parser.add_argument('--extension', type=str, help='png or jpg')
#parser.add_argument('--flowFwdDir', type=str, help='directory for flow_fwd')
#parser.add_argument('--flowBwdDir', type=str, help='directory for flow_bwd')
parser.add_argument('--frames', type=int, help='number of frames')
#tool_bilateraladv
#parser.add_argument('--inputdir', type=str, help='put an inputdirectory path here')
#parser.add_argument('--extension', type=str, help='png or jpg')
#parser.add_argument('--flowFwdDir', type=str, help='directory for flow_fwd')
#parser.add_argument('--flowBwdDir', type=str, help='directory for flow_bwd')
#parser.add_argument('--outputdirfilter', type=str, help='directory for input_filtered')
#parser.add_argument('--frames', type=int, help='number of frames')

##tool_gauss
#parser.add_argument('--geninputfilteredmask', type=str, help='paste _train\input_filtered directory')
#parser.add_argument('--extension', type=str, help='png or jpg')
#parser.add_argument('--flowFwdDir', type=str, help='directory for flow_fwd')
#parser.add_argument('--flowBwdDir', type=str, help='directory for flow_bwd')
#parser.add_argument('--outputdirfilter', type=str, help='directory for input_filtered output')
#parser.add_argument('--frames', type=int, help='number of frames')
parser.add_argument('--projectname', type=str, help='name of the project_')
#parser.add_argument('--gdisko15dir', type=str, help='paste location for diskodir')
parser.add_argument('--cpu'            , action='store_true'                  )
args = parser.parse_args()

data_path = os.path.expanduser('~\Documents\\visionsofchaos\\fewshot\\data')
gen = data_path + "\\" + args.projectname + "_gen"
train = data_path + "\\" + args.projectname + "_train"
flow = data_path + "\\" + args.projectname + "_flow"
##TOOL_DISFLOW##   
################ MAKE CHANGES HERE #################
inputdir = ""              # path to the input sequence PNGs
inputFileFormat = "%03d"        # name of input files, e.g., %03d if files are named 001.png, 002.png
inputFileExt = "png"            # extension of input files (without .), e.g., png, jpg
flowFwdDir = flow + "\\" + "flowfwd"         # path to the output forward flow files
flowBwdDir = flow + "\\" + "flowbwd"         # path to the output backward flow files
FIRST = 1                       # number of the first PNG file in the input folder
#frames = --frames                      # number of the last PNG file in the input folder
####################################################

if not os.path.exists(flowFwdDir):
    os.mkdir(flowFwdDir)
    
if not os.path.exists(flowBwdDir):
    os.mkdir(flowBwdDir)

inputFiles = gen + "\\" + "input_filtered" + "\\" + inputFileFormat + "." + args.extension
flwFwdFile = flow + "\\" + "flowfwd" + "\\" + inputFileFormat + ".A2V2f"
flwBwdFile = flow + "\\" + "flowbwd" + "\\" + inputFileFormat + ".A2V2f"

firstFrame = FIRST+1
lastFrame  = args.frames
frameStep  = +1

for frame in range(firstFrame,lastFrame+frameStep,frameStep):
    os.system("disflow %s %s %s"%(inputFiles%(frame),inputFiles%(frame-frameStep),flwFwdFile%(frame)))

firstFrame = args.frames-1
lastFrame  = FIRST
frameStep  = -1

for frame in range(firstFrame,lastFrame+frameStep,frameStep):
    os.system("disflow %s %s %s"%(inputFiles%(frame),inputFiles%(frame-frameStep),flwBwdFile%(frame)))
    
 
  
##TOOL_BILATERALADV
################ MAKE CHANGES HERE #################
inputFileFormat = "%03d"            # name of input files, e.g., %03d if files are named 001.png, 002.png
imageFormat   = train + "\\" + "input_filtered" + "\\" + inputFileFormat + "." + args.extension
flowFwdFormat = data_path + "\\" + "flowfwd" + "\\" + inputFileFormat + ".A2V2f"  # path to the forward flow files (computed by _tools/disflow)
flowBwdFormat = data_path + "\\" + "flowbwd" + "\\" + inputFileFormat + ".A2V2f"  # path to the backward flow files (computed by _tools/disflow)
outputFormat   = gen + "\\" + "input_filtered" + "\\" + inputFileFormat + "." + args.extension  # path to the result filtered sequence
frameFirst = 1                      # number of the first PNG file in the input folder
frameLast = args.frames                     # number of the last PNG file in the input folder
####################################################


os.makedirs(os.path.dirname(outputFormat),exist_ok=True)

for frame in range(firstFrame,lastFrame+frameStep,frameStep):
        filter = "bilateralAdv.exe "+imageFormat+" "+flowFwdFormat+" "+flowBwdFormat+(" %d "%(frame))+" 15 16 "+(outputFormat%(frame))
        #print(filter)
        os.system(filter)
  
firstFrame = frameFirst
lastFrame= frameLast
frameStep = +1

commands = [(frame) for frame in range(firstFrame, lastFrame + frameStep, frameStep)]




##TOOL_GAUSS##
################ MAKE CHANGES HERE #################
inputFileFormat = "%03d"    # name of input files, e.g., %03d if files are named 001.png, 002.png
maskDir = train + "\\" "input_filtered"          # mask dir, essentially leading frames from where the gaussians will be propagated
## fix maskdir so it works with args.geninputfilteredmask
#maskDir = str(args.geninputfilteredmask) ??
maskFiles = train + "\\" "input_filtered" + "\\" + inputFileFormat + "." + args.extension
flowFwdFiles = flow + "\\" + "flowfwd" + "\\" + inputFileFormat + ".A2V2f"  # path to the forward flow files (computed by _tools/disflow)
flowBwdFiles = flow + "\\" + "flowbwd" + "\\" + inputFileFormat + ".A2V2f"   # path to the backward flow files (computed by _tools/disflow)
#frameFirst = "001"                  # name of the first PNG file in the input folder (without extension)
#frameLast = "116"                  # number of the last PNG file in the input folder (without extension) 
##fix that it recognizes frames automatically
#frameLast = str(args.frames)  ??
gdisko_gauss_r10_s10_dir = gen + "\\" + "input_gdisko_gauss_r10_s10"    # path to the result gauss r10 s10 sequence
gdisko_gauss_r10_s15_dir = gen + "\\" + "input_gdisko_gauss_r10_s15"    # path to the result gauss r10 s15 sequence
#fix dirs for automatic pasting to project_gen and duplicating the selected frames to train
#gdisko_gauss_r10_s10_dir = 'C:\Users\Gebruiker\Documents\visionsofchaos\fewshot\data\' + "\" + str(args.projectname) + '_gen' + '\input_gdisko_gauss_r10_s10'   # path to the result gauss r10 s10 sequence
#gdisko_gauss_r10_s15_dir = 'C:\Users\Gebruiker\Documents\visionsofchaos\fewshot\data\' + "\" + str(args.projectname) + '_gen' + '\input_gdisko_gauss_r10_s15'    # path to the result gauss r10 s15 sequence
gdisko_gauss_r10_s10_files = gdisko_gauss_r10_s10_dir + "\\" + inputFileFormat + "." + args.extension 
gdisko_gauss_r10_s15_files = gdisko_gauss_r10_s15_dir + "\\" + inputFileFormat + "." + args.extension 
####################################################


if not os.path.exists(gdisko_gauss_r10_s10_dir):
    os.mkdir(gdisko_gauss_r10_s10_dir)
    
if not os.path.exists(gdisko_gauss_r10_s15_dir):
    os.mkdir(gdisko_gauss_r10_s15_dir)

masks_str = ""
masks_list_dir = os.listdir(maskDir)
for mask in masks_list_dir:
    masks_str += mask.replace(".png", "").replace(".jpg", "")
    masks_str += " "
    
frameFirst = 1 
lastFrame= args.frames

os.system(f"gauss.exe {maskFiles} {flowFwdFiles} {flowBwdFiles} {frameFirst} {lastFrame} {len(masks_list_dir)} {masks_str} 10 10 {gdisko_gauss_r10_s10_files}")
os.system(f"gauss.exe {maskFiles} {flowFwdFiles} {flowBwdFiles} {frameFirst} {lastFrame} {len(masks_list_dir)} {masks_str} 10 15 {gdisko_gauss_r10_s15_files}")


import os
import shutil

parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--videofile', type=str, help='path to your video file, for example --videofile C:\file\video\extract\video.mp4')
parser.add_argument('--projectname', type=str, help='name of the project to create the directories')
parser.add_argument('--framegap', type=int, help='name of the project to create the directories')

args = parser.parse_args()
doc_path = os.path.expanduser('~\Documents')
data_path = os.path.expanduser('~\Documents\\visionsofchaos\\fewshot\\data')

print (" ")
print ("making frames with your --framegap value to gen_filtered folder")

train_filtered = data_path+'\\'+str(args.projectname)+'_train'+'\\'+'input_filtered'
gen_filtered = data_path + '\\' + args.projectname + '_gen\\input_filtered'


video_length = len(os.listdir(gen_filtered))
print ("Number of frames: ", video_length)


if video_length <100:
    for i in range(1, 9, args.framegap):
        print (i)  
        shutil.copy2(gen_filtered+'\\'+'00'+str(i)+'.png', train_filtered)
    for i in range(10, video_length, args.framegap):
        print (i)  
        shutil.copy2(gen_filtered+'\\'+'0'+str(i)+'.png', train_filtered)
        
if video_length >100:
    for i in range(1, 9, args.framegap):
        print (i)  
        shutil.copy2(gen_filtered+'\\'+'00'+str(i)+'.png', train_filtered)
    for i in range(10, 99, args.framegap):
        print (i)  
        shutil.copy2(gen_filtered+'\\'+'0'+str(i)+'.png', train_filtered)
    for i in range(100, video_length, args.framegap):
        print (i)  
        shutil.copy2(gen_filtered+'\\'+str(i)+'.png', train_filtered)
print ("exported all frames to " "gen_filtered")        
