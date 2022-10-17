import os

import argparse

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--inputdir', type=str, help='put a projectname_gen/input_directory path here')
parser.add_argument('--extension', type=str, help='png or jpg')
parser.add_argument('--flowFwdDir', type=str, help='directory for flow_fwd')
parser.add_argument('--flowBwdDir', type=str, help='directory for flow_bwd')
parser.add_argument('--frames', type=int, help='number of frames')
args = parser.parse_args()
    
################ MAKE CHANGES HERE #################
inputdir = ""              # path to the input sequence PNGs
inputFileFormat = "%03d"        # name of input files, e.g., %03d if files are named 001.png, 002.png
inputFileExt = "png"            # extension of input files (without .), e.g., png, jpg
flowFwdDir = "C:\deepdream-test\Few-Shot-Patch-Based-Training-master\\flow_fwd"         # path to the output forward flow files
flowBwdDir = "C:\deepdream-test\Few-Shot-Patch-Based-Training-master\\flow_bwd"         # path to the output backward flow files
FIRST = 1                       # number of the first PNG file in the input folder
#frames = --frames                      # number of the last PNG file in the input folder
####################################################


if not os.path.exists(args.flowFwdDir):
    os.mkdir(args.flowFwdDir)
    
if not os.path.exists(args.flowBwdDir):
    os.mkdir(args.flowBwdDir)

inputFiles = args.inputdir + "/" + inputFileFormat + "." + args.extension
flwFwdFile = args.flowFwdDir + "/" + inputFileFormat + ".A2V2f"
flwBwdFile = args.flowBwdDir + "/" + inputFileFormat + ".A2V2f"

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

