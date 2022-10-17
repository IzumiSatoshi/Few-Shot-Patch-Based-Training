import argparse
import cv2
import time
import os
import shutil

parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--videofile', type=str, help='path to your video file, for example --videofile C:\file\video\extract\video.mp4')
parser.add_argument('--projectname', type=str, help='name of the project to create the directories')
parser.add_argument('--framegap', type=int, help='number of how many skip frames')

args = parser.parse_args()
doc_path = os.path.expanduser('~\Documents')
data_path = os.path.expanduser('~\Documents\\visionsofchaos\\fewshot\\data')

if not os.path.exists(doc_path+'\\'+'visionsofchaos'):
    path10= doc_path
    os.chdir(path10)
    newfolder_10='visionsofchaos'
    os.makedirs(newfolder_10)
if not os.path.exists(doc_path+'\\'+'visionsofchaos'+'\\'+'fewshot'):
    path11= doc_path+'\\'+'visionsofchaos'
    os.chdir(path11)
    newfolder_11='fewshot'
    os.makedirs(newfolder_11)
if not os.path.exists(doc_path+'\\'+'visionsofchaos'+'\\'+'fewshot'+'\\'+'data'):
    path12= doc_path+'\\'+'visionsofchaos'+'\\'+'fewshot'
    os.chdir(path12)
    newfolder_12='data'
    os.makedirs(newfolder_12)



#make subdirectories
path1= data_path
os.chdir(path1)
newfolder=str(args.projectname)+'_train'
os.makedirs(newfolder)
    #to make folders inside the other folders
path2=path1+'\\'+newfolder
os.chdir(path2)
newfolder_2='input_filtered'
os.makedirs(newfolder_2)

path3=path1+'\\'+newfolder
os.chdir(path2)
newfolder_3='mask'
os.makedirs(newfolder_3)    

path4=path1+'\\'+newfolder
os.chdir(path2)
newfolder_4='output'
os.makedirs(newfolder_4)    


path5= data_path
os.chdir(path5)
newfolder_5=str(args.projectname)+'_gen'
os.makedirs(newfolder_5)

 
path6= data_path + '\\' + args.projectname + '_gen'
os.chdir(path6)
newfolder_6='input'
os.makedirs(newfolder_6)

path13= data_path + '\\' + args.projectname + '_gen'
os.chdir(path13)
newfolder_13='input_filtered'
os.makedirs(newfolder_13)

path14= data_path + '\\' + args.projectname + '_gen'
os.chdir(path14)
newfolder_14='input_gdisko_gauss_r10_s10'
os.makedirs(newfolder_14)

path15= data_path + '\\' + args.projectname + '_gen'
os.chdir(path15)
newfolder_15='input_gdisko_gauss_r10_s15'
os.makedirs(newfolder_15)

path16= data_path + '\\' + args.projectname + '_train'
os.chdir(path16)
newfolder_16='input_gdisko_gauss_r10_s10'
os.makedirs(newfolder_16)

path17= data_path + '\\' + args.projectname + '_train'
os.chdir(path17)
newfolder_17='input_gdisko_gauss_r10_s15'
os.makedirs(newfolder_17)



path7= data_path + '\\' + args.projectname + '_gen'
os.chdir(path6)
newfolder_7='mask'
os.makedirs(newfolder_7)    

path8= data_path + '\\' + args.projectname + '_gen'
os.chdir(path6)
newfolder_8='output'
os.makedirs(newfolder_8)   

path9= data_path
os.chdir(path9)
newfolder_9=str(args.projectname)+'_flow'
os.makedirs(newfolder_9)


train_filtered = data_path+str(args.projectname)+'_train'+'\\'+'input_filtered'

#convert video
def video_to_frames(input_loc, output_loc):
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
    video_to_frames(input_loc, output_loc)

import os
import shutil

print (" ")
print ("making frames with your --framegap value to gen_filtered folder")

train_filtered = data_path+'\\'+str(args.projectname)+'_train'+'\\'+'input_filtered'
gen_filtered = data_path + '\\' + args.projectname + '_gen\\input_filtered'


video_length = len(os.listdir(gen_filtered))
print ("Number of frames: ", video_length)

if int(args.framegap) <9:
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
if int(args.framegap) >9:
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
            
        for i in range(100-args.framegap, video_length, args.framegap):
            print (i)  
            shutil.copy2(gen_filtered+'\\'+str(i)+'.png', train_filtered)
print ("exported all frames to " "gen_filtered")        
