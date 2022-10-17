import argparse
import cv2
import time
import os
import shutil

parser = argparse.ArgumentParser(description='arguments')

parser.add_argument('--videofile', type=str, help='path to your video file, for example --videofile C:\file\video\extract\video.mp4',required=True)
parser.add_argument('--projectname', type=str, help='name of the project to create the directories',required=True)
parser.add_argument('--framegap', type=int,default='5', help='number of how many skip frames',required=True)
parser.add_argument('--precision', type=str, help='detailed gives less styletransfer but more accurate',choices=['detailed','normal'],required=True)
parser.add_argument('--W', type=str, help='width',required=True)
parser.add_argument('--H', type=str, help='height',required=True)
parser.add_argument('--logpath', type=str, help='path where your training happens',default = 'logs')
args = parser.parse_args()

cwd = os.getcwd()
tools_all = cwd + '/_tools/tools_all.py'
trainur = cwd + '/train.py'
disco1010 = cwd + '/_config/reference_P_disco1010.yaml'
disco1015 = cwd + '/_config/reference_P_disco1015.yaml'

doc_path = os.path.expanduser('~\Documents')
if args.logpath:
    data_path = cwd+'/'+args.logpath
    if not os.path.exists(data_path):
        path999= cwd
        os.chdir(path999)
        newfolder_999=data_path
        os.makedirs(newfolder_999)
else:
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
path2=path1+'/'+newfolder
os.chdir(path2)
newfolder_2='input_filtered'
os.makedirs(newfolder_2)

path3=path1+'/'+newfolder
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

path9= data_path
os.chdir(path9)
newfolder_30=str(args.projectname)+'_flow/flowfwd'
os.makedirs(newfolder_30)

path9= data_path
os.chdir(path9)
newfolder_31=str(args.projectname)+'_flow/flowbwd'
os.makedirs(newfolder_31)


train_filtered = data_path+str(args.projectname)+'_train'+'\\'+'input_filtered'

#convert video
def video_to_frames(input_loc, output_loc):
    #Function to extract frames from input video file
    #and save them as separate frames in an output directory.
    #Args:
    #    input_loc: Input video file.
    #    output_loc: Output directory to save the frames.
    #Returns:
    #    None
    
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

train_filtered = data_path+'\\'+str(args.projectname)+'_train'+'\\'+'input_filtered/'
gen_filtered = data_path + '\\' + args.projectname + '_gen\\input_filtered/'
gen_filtered_batch = data_path + '\\' + args.projectname + '_gen\\input_filtered/*'
gen_mask = data_path + '\\' + args.projectname + '_gen\\mask/'
train_output = data_path+'\\'+str(args.projectname)+'_train'+'\\'+'output/'
train_output_batch = data_path+'\\'+str(args.projectname)+'_train'+'\\'+'output/*'
train_filtered = data_path + '\\' + args.projectname + '_train\\input_filtered/'
train_filtered_batch = data_path + '\\' + args.projectname + '_train\\input_filtered/*'
train_mask = data_path + '\\' + args.projectname + '_train\\mask/'
train_root = data_path + '\\' + args.projectname + '_train/'
disco1010path = data_path + '\\' + args.projectname + '_gen\\res__P_disco1010'
disco1015path = data_path + '\\' + args.projectname + '_gen\\res__P_disco1015'


video_length = len(os.listdir(gen_filtered))
print ("Number of frames: ", video_length)
print ("exporting framegap frames")

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
print ("exported framegap frames to " ,train_filtered)        

import subprocess


resizesize = args.W + 'x' + args.H + '!'
imageread = gen_filtered +'001.png'
import cv2
img = cv2.imread(imageread)
print(img.shape)
if( img.shape != (args.H, args.W, 3) ):
    subprocess.run(["magick", "mogrify", "-resize", resizesize, "-quality", "100", gen_filtered_batch])#, "*.png", "-quality", "100", gen_filtered])
    print ("frames in ",gen_filtered, "resized") 
    subprocess.run(["magick", "mogrify", "-resize", resizesize, "-quality", "100", train_filtered_batch])#, "*.png", "-quality", "100", train_filtered])
    print ("frames in ",train_filtered, "resized")
if( img.shape == (args.H, args.W, 3) ):
    print('no resizing needed')

     
subprocess.run(['magick', 'mogrify', '-brightness-contrast', '200x0', '-path', gen_mask, '-format','png', gen_filtered_batch])
print ("masks in " ,gen_mask, "made") 
subprocess.run(['magick', 'mogrify', '-brightness-contrast', '200x0', '-path', train_mask, '-format','png', train_filtered_batch])
print ("masks in " ,train_mask, "made") 

prjnm = str(args.projectname)
frmgp = str(args.framegap)
video_length2 = str(video_length)
logpath = str(data_path)

print("")
print("")
print("")
print("")
print("preparation done, apply desired effects on the images that are in '",train_filtered,"' and export those to '",train_output,"'")
print("")
print("")
print("the flowframe prediction takes a while to render but it renders on cpu so u can create the export frames with effects with your GPU in the meanwhile")
print("")
print("")
frowframe_run1 = (input("have your read the above and understand? press ENTER if you do"))
print("")
print("")
if frowframe_run1:
    print("")
    print("")
frowframe_run2 = (input("are you sure u read it? press ENTER"))
print("")
print("")
if frowframe_run2:
    print("")
    print("")    
frowframe_run = (input("press ENTER to start rendering the flowframes"))
if frowframe_run:
    print("")
    print("")  
subprocess.run(['python', tools_all, '--projectname', prjnm, '--frames', video_length2, '--extension', 'png','--framegap', frmgp, '--precision', 'detailed','--logpath',logpath]) #add choice for precision and add '--export_path', args.export_path


print("")
print("")
print("a reminder to put the styled export frames into", train_output)
export_done = (input("are you done with creating the styled frames? press ENTER to start patch based training"))

if export_done:
    print("")
    
    print("")
imageread1 = train_output +'001.png'
import cv2
img1 = cv2.imread(imageread1)
if args.precision == 'detailed':
    print("results will appear in ",disco1010path,"every 10000 steps")
else:
    print("results will appear in ",disco1015path,"every 10000 steps")
if( img1.shape != (args.H, args.W, 3) ):
        subprocess.run(["magick", "mogrify", "-resize", resizesize, "-quality", "100", train_output_batch]) 
    
if args.precision == 'detailed':
        print('python', '-B', trainur, '--config', disco1010, '--data_root', train_root, '--log_interval', '10000', '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath)
        subprocess.run(['python', '-B', trainur, '--config', disco1010, '--data_root', train_root, '--log_interval', '10000', '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath])
        
else:
        print('python', '-B', trainur, '--config', disco1015, '--data_root', train_root, '--log_interval', '10000', '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath)
        subprocess.run(['python', '-B', trainur, '--config', disco1015, '--data_root', train_root, '--log_interval', '10000', '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath])
        
