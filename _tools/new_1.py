import argparse
import cv2
import time
import os
import shutil
import numpy
parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--videofile', type=str, help='path to your video file, for example --videofile C:\file\video\extract\video.mp4')
parser.add_argument('--projectname', type=str, help='name of the project to create the directories')
parser.add_argument('--framegap', type=int, help='name of the project to create the directories')

args = parser.parse_args()
doc_path = os.path.expanduser('~\Documents')
data_path = os.path.expanduser('~\Documents\\visionsofchaos\\fewshot\\data')
other_output_loc = data_path + '\\' + args.projectname + '_train\\input_filtered'

train_filtered = data_path+str(args.projectname)+'_train'+'\\'+'input_filtered'
framegap = args.framegap
#take every #nth frame
#convert video
def video_to_frames(input_loc, output_loc, framegap, other_output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#03d.png" % (count+1), frame)

        # if our count is divisible by the framegap, copy to other folder location
        if count % args.framegap == 1:
            cv2.imwrite(other_output_loc, "/%#03d.png" % (count+1), frame)        
        count = count + 1

        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break
if __name__=="__main__":
    input_loc = args.videofile
    output_loc = data_path + '\\' + args.projectname + '_gen\\input_filtered'
    other_output_loc = data_path + '\\' + args.projectname + '_train\\input_filtered'
    video_to_frames(input_loc, output_loc, framegap, other_output_loc)
