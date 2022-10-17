import os
import argparse

parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--geninputfilteredmask', type=str, help='paste _train\input_filtered directory')
parser.add_argument('--extension', type=str, help='png or jpg')
parser.add_argument('--flowFwdDir', type=str, help='directory for flow_fwd')
parser.add_argument('--flowBwdDir', type=str, help='directory for flow_bwd')
parser.add_argument('--outputdirfilter', type=str, help='directory for input_filtered output')
parser.add_argument('--frames', type=int, help='number of frames')
parser.add_argument('--projectname', type=str, help='paste _train\input_filtered directory')
parser.add_argument('--gdisko15dir', type=str, help='paste location for diskodir')
args = parser.parse_args()

################ MAKE CHANGES HERE #################
inputFileFormat = "%03d"    # name of input files, e.g., %03d if files are named 001.png, 002.png
maskDir ="C:/Users/Gebruiker/Documents/visionsofchaos/fewshot/data/project2_train/input_filtered"           # mask dir, essentially leading frames from where the gaussians will be propagated
## fix maskdir so it works with args.geninputfilteredmask
#maskDir = str(args.geninputfilteredmask) ??
maskFiles = maskDir + "/" + inputFileFormat + ".png"
flowFwdFiles = "flow_fwd" + "/" + inputFileFormat + ".A2V2f"  # path to the forward flow files (computed by _tools/disflow)
flowBwdFiles = "flow_bwd" + "/" + inputFileFormat + ".A2V2f"   # path to the backward flow files (computed by _tools/disflow)
frameFirst = "001"                  # name of the first PNG file in the input folder (without extension)
frameLast = "116"                  # number of the last PNG file in the input folder (without extension) 
##fix that it recognizes frames automatically
#frameLast = str(args.frames)  ??
gdisko_gauss_r10_s10_dir = "input_gdisko_gauss_r10_s10"    # path to the result gauss r10 s10 sequence
gdisko_gauss_r10_s15_dir = "input_gdisko_gauss_r10_s15"    # path to the result gauss r10 s15 sequence
#fix dirs for automatic pasting to project_gen and duplicating the selected frames to train
#gdisko_gauss_r10_s10_dir = 'C:\Users\Gebruiker\Documents\visionsofchaos\fewshot\data\' + ''\'' + str(args.projectname) + '_gen' + '\input_gdisko_gauss_r10_s10'   # path to the result gauss r10 s10 sequence
#gdisko_gauss_r10_s15_dir = 'C:\Users\Gebruiker\Documents\visionsofchaos\fewshot\data\' + ''\'' + str(args.projectname) + '_gen' + '\input_gdisko_gauss_r10_s15'    # path to the result gauss r10 s15 sequence
gdisko_gauss_r10_s10_files = gdisko_gauss_r10_s10_dir + "/" + inputFileFormat + ".png" 
gdisko_gauss_r10_s15_files = gdisko_gauss_r10_s15_dir + "/" + inputFileFormat + ".png" 
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


os.system(f"gauss.exe {maskFiles} {flowFwdFiles} {flowBwdFiles} {frameFirst} {frameLast} {len(masks_list_dir)} {masks_str} 10 10 {gdisko_gauss_r10_s10_files}")
os.system(f"gauss.exe {maskFiles} {flowFwdFiles} {flowBwdFiles} {frameFirst} {frameLast} {len(masks_list_dir)} {masks_str} 10 15 {gdisko_gauss_r10_s15_files}")
