import os
import argparse

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--inputdir', type=str, help='put an inputdirectory path here')
parser.add_argument('--extension', type=str, help='png or jpg')
parser.add_argument('--flowFwdDir', type=str, help='directory for flow_fwd')
parser.add_argument('--flowBwdDir', type=str, help='directory for flow_bwd')
parser.add_argument('--outputdirfilter', type=str, help='directory for input_filtered')
parser.add_argument('--frames', type=int, help='number of frames')
args = parser.parse_args()
    
################ MAKE CHANGES HERE #################
inputFileFormat = "%03d"            # name of input files, e.g., %03d if files are named 001.png, 002.png
imageFormat   = args.inputdir + inputFileFormat + ".png"
flowFwdFormat = args.flowFwdDir + inputFileFormat + ".A2V2f"  # path to the forward flow files (computed by _tools/disflow)
flowBwdFormat = args.flowBwdDir + inputFileFormat + ".A2V2f"  # path to the backward flow files (computed by _tools/disflow)
outputFormat   = args.outputdirfilter + inputFileFormat + ".png"  # path to the result filtered sequence
frameFirst = 1                      # number of the first PNG file in the input folder
#frameLast = 109                     # number of the last PNG file in the input folder
####################################################

firstFrame = frameFirst
lastFrame= args.frames
frameStep = +1

os.makedirs(os.path.dirname(outputFormat),exist_ok=True)

for frame in range(firstFrame,lastFrame+frameStep,frameStep):  	
  filter = "bilateralAdv.exe "+imageFormat+" "+flowFwdFormat+" "+flowBwdFormat+(" %d "%(frame))+" 15 16 "+(outputFormat%(frame))
  #print(filter)
  os.system(filter)
