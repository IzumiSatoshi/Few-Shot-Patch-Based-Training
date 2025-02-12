import argparse
import cv2
import time
import os
import shutil
from gooey import Gooey, GooeyParser
import PySimpleGUI as sg 

@Gooey(program_name="Few-Shot UI")
def parse_args():
    parser = GooeyParser(description='this script is based on FSPBT')#argparse.ArgumentParser(description='arguments')
    files = parser.add_argument_group('fileselection', gooey_options={'show_border': bool,'show_underline': bool,'label_color': '#FF9900','columns': 2})
    size = parser.add_argument_group('size', gooey_options={'show_border': bool,'show_underline': bool,'label_color': '#FF9900','columns': 1})
    settings = parser.add_argument_group('settings', gooey_options={'show_border': bool,'show_underline': bool,'label_color': '#FF9900','columns': 3})
    mysterysettings = parser.add_argument_group('mystery settings', gooey_options={'show_border': bool,'show_underline': bool,'label_color': '#FF9900','columns': 3})
    
    files.add_argument('--inputfile', widget="FileChooser", type=str, help='path to your input video or gif, only up to 1000 frames for now',required=True)
    files.add_argument('--maskfile', widget="FileChooser", type=str, help='path to your mask video (optional)(should be an alpha mask)')
    files.add_argument('--projectname', type=str, help='name of the project to create the directories',required=True)
    settings.add_argument('--framegap',default=5, widget="IntegerField", gooey_options={'min': 0, 'max': 1000, 'increment': int}, type=int, help='number of how many frames are inbetween the style images')
    settings.add_argument('--precision', type=str, help='_flow takes more time to prepape but is much better and faster at training',choices=['detailed_flow','undetailed_flow', 'normal', 'normal_slow','webcam_test'],required=True,nargs='*')
    size.add_argument('--W', type=str,metavar='width', widget="IntegerField", gooey_options={'min': 0, 'max': 1024, 'increment': int},default=512)
    size.add_argument('--H', type=str,metavar='height', widget="IntegerField", gooey_options={'min': 0, 'max': 1024, 'increment': int},default=512)
    files.add_argument('--logpath', type=str, help='name of the path where your training happens',default = 'logs')
    settings.add_argument('--log_interval', type=str, help='path where your training happens',default = '10000', widget="IntegerField", gooey_options={'min': 100, 'max': 80000, 'increment': int})
    
    mysterysettings.add_argument('--perception_loss_weight', help='test', default=6.0, widget="DecimalField")
    mysterysettings.add_argument('--reconstruction_weight', help='test', default=4., widget="DecimalField")
    mysterysettings.add_argument('--adversarial_weight', help='test', default=0.5, widget="DecimalField")
    mysterysettings.add_argument('--append_smoothers', type=str,nargs='*', help='test',default = 'True', choices=['True', 'False'])
    mysterysettings.add_argument('--filters_layers', type=str,nargs='*', help='test',default = '[32, 64, 128, 128, 128, 64]', choices=['[32, 64, 128, 128, 128, 64]', '[32, 32, 32, 32, 32, 32]','[32, 64, 64, 64, 64, 64]'])
    mysterysettings.add_argument('--patch_size', type=str,nargs='*', help='test',default = '32', choices=['16','32','64','128'])
    mysterysettings.add_argument('--use_normalization', type=str,nargs='*', help='test',default = 'False', choices=['True', 'False'])
    mysterysettings.add_argument('--use_image_loss', type=str,nargs='*', help='test',default = 'True', choices=['True', 'False'])
    mysterysettings.add_argument('--tanh', type=str,nargs='*', help='test',default = 'True', choices=['True', 'False'])
    mysterysettings.add_argument('--use_bias', type=str,nargs='*', help='test',default = 'True', choices=['True', 'False'])
    return parser.parse_args()

args = parse_args()

cwd = os.getcwd()
tools_all = cwd + '/_tools/tools_all.py'
trainur = cwd + '/train.py'
disco1010 = cwd + '/_config/reference_P_disco1010.yaml'
disco1015 = cwd + '/_config/reference_P_disco1015.yaml'

webcam = cwd + '/_config/reference_webcam.yaml'
normal = cwd + '/_config/reference_P.yaml'
normal2 = cwd + '/_config/reference_F.yaml'

doc_path = os.path.expanduser('~\Documents')
if args.logpath:
    data_path = cwd+'/'+args.logpath
    if not os.path.exists(data_path):
        path999= cwd
        os.chdir(path999)
        newfolder_999=data_path
        os.makedirs(newfolder_999)
else:
    data_path = os.path.expanduser('~\Documents/visionsofchaos/fewshot/data')

    if not os.path.exists(doc_path+'/'+'visionsofchaos'):
        path10= doc_path
        os.chdir(path10)
        newfolder_10='visionsofchaos'
        os.makedirs(newfolder_10)
    if not os.path.exists(doc_path+'/'+'visionsofchaos'+'/'+'fewshot'):
        path11= doc_path+'/'+'visionsofchaos'
        os.chdir(path11)
        newfolder_11='fewshot'
        os.makedirs(newfolder_11)
    if not os.path.exists(doc_path+'/'+'visionsofchaos'+'/'+'fewshot'+'/'+'data'):
        path12= doc_path+'/'+'visionsofchaos'+'/'+'fewshot'
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

path4=path1+'/'+newfolder
os.chdir(path2)
newfolder_4='output'
os.makedirs(newfolder_4)    


path5= data_path
os.chdir(path5)
newfolder_5=str(args.projectname)+'_gen'
os.makedirs(newfolder_5)

 
path6= data_path + '/' + args.projectname + '_gen'
os.chdir(path6)
#newfolder_6='input'
#os.makedirs(newfolder_6)

path13= data_path + '/' + args.projectname + '_gen'
os.chdir(path13)
newfolder_13='input_filtered'
os.makedirs(newfolder_13)


path7= data_path + '/' + args.projectname + '_gen'
os.chdir(path6)
newfolder_7='mask'
os.makedirs(newfolder_7)    

path8= data_path + '/' + args.projectname + '_gen'
os.chdir(path6)
#newfolder_8='output'
#os.makedirs(newfolder_8)   

if args.precision == ['detailed_flow']:
    flow = True
elif args.precision == ['undetailed_flow']:
    flow = True
else:
    flow = False
    
if flow == True: 
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
    if args.precision == ['detailed_flow']:
        path14= data_path + '/' + args.projectname + '_gen'
        os.chdir(path14)
        newfolder_14='input_gdisko_gauss_r10_s10'
        os.makedirs(newfolder_14)

        path16= data_path + '/' + args.projectname + '_train'
        os.chdir(path16)
        newfolder_16='input_gdisko_gauss_r10_s10'
        os.makedirs(newfolder_16)
    elif args.precision == ['undetailed_flow']:
        path15= data_path + '/' + args.projectname + '_gen'
        os.chdir(path15)
        newfolder_15='input_gdisko_gauss_r10_s15'
        os.makedirs(newfolder_15)

        path17= data_path + '/' + args.projectname + '_train'
        os.chdir(path17)
        newfolder_17='input_gdisko_gauss_r10_s15'
        os.makedirs(newfolder_17)


train_filtered = data_path+str(args.projectname)+'_train'+'/'+'input_filtered'

import moviepy.editor as mp
inputfile = args.inputfile
if inputfile.endswith('.gif'):
    clip = mp.VideoFileClip(args.inputfile)
    clip.write_videofile("myvideo.mp4")

#convert video
def video_to_frames(input_loc, output_loc):
    args = parse_args()
    #Function to extract frames from input video file
    #and save them as separate frames in an output directory.
    #Args:
    #    input_loc: Input video file.
    #    output_loc: Output directory to save the frames.
    #Returns:
    #    None
    import moviepy.editor as mp
    clip = mp.VideoFileClip(input_loc1)
    clip_resized = clip.resize((args.W,args.H)) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
    clip_resized.write_videofile("movie_resized.mp4")
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
    if inputfile.endswith('.gif'):
        input_loc1 = "myvideo.mp4"
        input_loc = "movie_resized.mp4"
    else:
        input_loc1 = args.inputfile
        input_loc = "movie_resized.mp4"
    output_loc = data_path + '/' + args.projectname + '_gen/input_filtered'
    video_to_frames(input_loc, output_loc)


               
import os
import shutil

print (" ")
print ("making frames with your --framegap value to gen_filtered folder")

gen = data_path + '/' + args.projectname + '_gen/'
train = data_path + '/' + args.projectname + '_train/'
train_filtered = data_path+'/'+str(args.projectname)+'_train'+'/'+'input_filtered/'
gen_filtered = data_path + '/' + args.projectname + '_gen/input_filtered/'
gen_filtered_batch = data_path + '/' + args.projectname + '_gen/input_filtered/*'
gen_mask = data_path + '/' + args.projectname + '_gen/mask/'
gen_mask2 = data_path + '/' + args.projectname + '_gen/mask'
train_output = data_path+'/'+str(args.projectname)+'_train'+'/'+'output/'
train_output_batch = data_path+'/'+str(args.projectname)+'_train'+'/'+'output/*'
train_filtered = data_path + '/' + args.projectname + '_train/input_filtered/'
train_filtered_batch = data_path + '/' + args.projectname + '_train/input_filtered/*'
train_mask = data_path + '/' + args.projectname + '_train/mask/'
train_root = data_path + '/' + args.projectname + '_train/'

if flow == True:
    disco1010path = data_path + '/' + args.projectname + '_gen/res__P_disco1010'
    disco1015path = data_path + '/' + args.projectname + '_gen/res__P_disco1015'
resppath = data_path + '/' + args.projectname + '_gen/res__P'
resfpath = data_path + '/' + args.projectname + '_gen/res__F'


video_length = len(os.listdir(gen_filtered))
print ("Number of frames: ", video_length)

if args.framegap:
    print ("exporting framegap frames")

    if video_length <100:
        for i in range(1, 9, args.framegap):
            print (i)  
            shutil.copy2(gen_filtered+'/'+'00'+str(i)+'.png', train_filtered)
        for i in range(10, video_length, args.framegap):
            print (i)  
            shutil.copy2(gen_filtered+'/'+'0'+str(i)+'.png', train_filtered)
        
    if video_length >100:
        for i in range(1, 9, args.framegap):
            print (i)  
            shutil.copy2(gen_filtered+'/'+'00'+str(i)+'.png', train_filtered)
        for i in range(10, 99, args.framegap):
            print (i)  
            shutil.copy2(gen_filtered+'/'+'0'+str(i)+'.png', train_filtered)
        for i in range(100, video_length, args.framegap):
            print (i)  
            shutil.copy2(gen_filtered+'/'+str(i)+'.png', train_filtered)
    print ("exported framegap frames to " ,train_filtered)      
else:
    print("you didn't provide a framegap value so you'll have to choose the frames that you display the style on yourself")
    print("choose a couple frames in ",gen_filtered,"and put them in",train_filtered)
    window=sg.Window('READ',
                     [ [sg.Text("you didn't provide a framegap value so you'll have to choose the frames that you display the style on yourself")],
                       [sg.Text(f"choose a couple frames in {gen_filtered} and put them in {train_filtered}")],
                       [sg.Text('COPY THE PATHS IN THE STATUS WINDOW AND DO WHAT IT SAYS')],
                       [sg.Text('have your read the above and understand? press Submit if you do')],
                       [sg.Submit()]])
    event, values = window.Read()
    window.Close() 

import subprocess


resizesize = args.W + 'x' + args.H + '!'
imageread = gen_filtered +'001.png'
import cv2
img = cv2.imread(imageread)
print(img.shape)
'''if( img.shape != (args.H, args.W, 3) ):
    subprocess.run(["magick", "mogrify", "-resize", resizesize, "-quality", "100", gen_filtered_batch])#, "*.png", "-quality", "100", gen_filtered])
    print ("frames in ",gen_filtered, "resized") 
    subprocess.run(["magick", "mogrify", "-resize", resizesize, "-quality", "100", train_filtered_batch])#, "*.png", "-quality", "100", train_filtered])
    print ("frames in ",train_filtered, "resized")
if( img.shape == (args.H, args.W, 3) ):
    print('no resizing needed')'''

if args.maskfile:
    input_loc1 = args.maskfile
    input_loc = "movie_resized.mp4"
    output_loc = data_path + '/' + args.projectname + '_gen/mask'
    video_to_frames(input_loc, output_loc)
    if args.framegap:
        if video_length <100:
            for i in range(1, 9, args.framegap):
                print (i)  
                shutil.copy2(gen_mask2+'/'+'00'+str(i)+'.png', train_mask)
            for i in range(10, video_length, args.framegap):
                print (i)  
                shutil.copy2(gen_mask2+'/'+'0'+str(i)+'.png', train_mask)
        
        if video_length >100:
            for i in range(1, 9, args.framegap):
                print (i)  
                shutil.copy2(gen_mask2+'/'+'00'+str(i)+'.png', train_mask)
            for i in range(10, 99, args.framegap):
                print (i)  
                shutil.copy2(gen_mask2+'/'+'0'+str(i)+'.png', train_mask)
            for i in range(100, video_length, args.framegap):
                print (i)  
                shutil.copy2(gen_mask2+'/'+str(i)+'.png', train_mask)
        print ("exported framegap frames to " ,train_mask)      
    else:
        print("you didn't provide a framegap value so you'll have to choose the frames that you display the style on yourself")
        print("copy your previously chosen framenumbers but now in this folder = ",gen_mask2,"and put them in",train_mask)
        window=sg.Window('READ',
                        [ [sg.Text("you didn't provide a framegap value so you'll have to choose the frames that you display the style on yourself")],
                        [sg.Text(f"copy your previously chosen framenumbers but now in this folder = {gen_mask2} and pu them in {train_mask}")],
                        [sg.Text('COPY THE PATHS IN THE STATUS WINDOW AND DO WHAT IT SAYS')],
                        [sg.Text('have your read the above and understand? press Submit if you do')],
                        [sg.Submit()]])
        event, values = window.Read()
        window.Close() 
else:    
    import shutil
    import os
    if os.path.exists(gen_mask): 
        os.rmdir(gen_mask)
    if os.path.exists(train_mask):  
        os.rmdir(train_mask)
    # path to source directory
    src_dir = gen_filtered
 
    # path to destination directory
    dest_dir = gen_mask
 
    # getting all the files in the source directory
    files = os.listdir(src_dir)
 
    shutil.copytree(src_dir, dest_dir)
    
    src_dir2 = train_filtered
 
    
    dest_dir2 = train_mask
 
    files = os.listdir(src_dir2)
 
    shutil.copytree(src_dir2, dest_dir2)
    #import pytesseract
    from PIL import Image, ImageEnhance
    import glob
    images = glob.glob(gen_mask+'/*.png')

    for image in images:
        jpg_path = os.path.join(gen_mask, image)
        if os.path.isfile(jpg_path):
            #head_tail = os.path.basename(file)
            
            img = Image.open(jpg_path)
            contrast = ImageEnhance.Contrast(img)
            factor = 0
            img = contrast.enhance(factor)
            enhancer = ImageEnhance.Brightness(img)
            factor2 = 1000 #brightens the image
            img = enhancer.enhance(factor2)
            img.save(jpg_path)
            #print(jpg_path)
    print ("masks in " ,gen_mask, "made")        
    images2 = glob.glob(train_mask+'/*.png')

    for image in images2:
        jpg_path = os.path.join(train_mask, image)
        if os.path.isfile(jpg_path):
            #head_tail = os.path.basename(file)
            
            img = Image.open(jpg_path)
            contrast = ImageEnhance.Contrast(img)
            factor = 0
            img = contrast.enhance(factor)
            enhancer = ImageEnhance.Brightness(img)
            factor2 = 1000 #brightens the image
            img = enhancer.enhance(factor2)
            img.save(jpg_path)
            #print(jpg_path)
            #im_output.save(file)
        #subprocess.run(['magick', 'mogrify', '-brightness-contrast', '200x0', '-path', gen_mask, '-format','png', gen_filtered_batch])
        #print ("masks in " ,gen_mask, "made") 
        #subprocess.run(['magick', 'mogrify', '-brightness-contrast', '200x0', '-path', train_mask, '-format','png', train_filtered_batch])
    print ("masks in " ,train_mask, "made") 

prjnm = str(args.projectname)
frmgp = str(args.framegap)
video_length2 = str(video_length)
logpath = str(data_path)

if flow == True:
    deletevideo = train+"movie_resized.mp4"
else:
    deletevideo = gen+"movie_resized.mp4"
os.remove(deletevideo)

                    
print("")
print("")
print("")
print("!!!!!important!!!!!")
print("")
print("preparation done, apply desired effects on the images that are in '",train_filtered,"' and export those to '",train_output,"'")
print("")
print("")
print("")
print("")
print("")

if args.precision == ['detailed_flow']:  
    window=sg.Window('READ',
                     [ [sg.Text('the frame movement prediction takes a while to render but it renders on CPU, so u can create the export frames with effects with your GPU in the meanwhile')],
                       [sg.Text('COPY THE PATHS BELOW !!!!!important!!!!! IN THE STATUS WINDOW AND DO WHAT IT SAYS')],
                       #[sg.Text('preparation done, apply desired effects on the images that are in "',int(train_filtered),'" and export those to "',int(train_output),'"')],
                       [sg.Text('have your read the above and understand? press Submit if you do')],
                       [sg.Submit()]])
    event, values = window.Read()
    window.Close()   
    print("")
    print("")
    print("")
    print("")
    print("STARTING PREPARATION")
elif args.precision == ['undetailed_flow']:
    window=sg.Window('READ',
                     [ [sg.Text('the frame movement prediction takes a while to render but it renders on CPU, so u can create the export frames with effects with your GPU in the meanwhile')],
                       [sg.Text('COPY THE PATHS BELOW !!!!!important!!!!! IN THE STATUS WINDOW AND DO WHAT IT SAYS')],
                       #[sg.Text('preparation done, apply desired effects on the images that are in "',int(train_filtered),'" and export those to "',int(train_output),'"')],
                       [sg.Text('have your read the above and understand? press Submit if you do')],
                       [sg.Submit()]])
    event, values = window.Read()
    window.Close() 
    print("")
    print("")
    print("")
    print("")
    print("STARTING PREPARATION")
else:
    """R=fedit(title='READ',
            comment="webcam_test, normal, normal_slow don't use movement prediction but you still have to process the images",
            data= [(None, None),
                   (None, "COPY THE PATHS IN THE STATUS WINDOW AND DO WHAT IT SAYS"),
                   (None, None),
                   (None, 'have your read the above and understand? press OK if you do'),
                   ])"""
    window=sg.Window('READ',
                     [ [sg.Text("webcam_test, normal, normal_slow don't use movement prediction but you still have to process the images")],
                       [sg.Text('COPY THE PATHS BELOW !!!!!important!!!!! IN THE STATUS WINDOW AND DO WHAT IT SAYS')],
                       #[sg.Text('preparation done, apply desired effects on the images that are in "',int(train_filtered),'" and export those to "',int(train_output),'"')],
                       [sg.Text('have your read the above and understand? press Submit if you do')],
                       [sg.Submit()]])
    event, values = window.Read()
    window.Close() 
"""print("")
print("")
print("")
print("")
print("")
if args.precision == 'detailed_flow':  
    print("the frame movement prediction takes a while to render but it renders on CPU, so u can create the export frames with effects with your GPU in the meanwhile")
elif args.precision == 'undetailed_flow':  
    print("the frame movement prediction takes a while to render but it renders on CPU, so u can create the export frames with effects with your GPU in the meanwhile")
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
if args.precision == 'detailed_flow':    
    frowframe_run = (input("press ENTER to start rendering the movement prediction frames"))
elif args.precision == 'undetailed_flow':
    frowframe_run = (input("press ENTER to start rendering the movement prediction frames"))
else:
    print("")
    print("")
if args.precision == 'detailed_flow':      
    if frowframe_run:
        print("")
        print("") 
elif args.precision == 'undetailed_flow':      
    if frowframe_run:
        print("")
        print("") """
if args.precision == ['detailed_flow']:
        if args.framegap:
            if args.maskfile:
                subprocess.run(['python', tools_all, '--projectname', prjnm, '--frames', video_length2, '--extension', 'png','--framegap', frmgp, '--precision', 'detailed_flow','--logpath',logpath,'--mask', '1'], shell=True)
            else:
                subprocess.run(['python', tools_all, '--projectname', prjnm, '--frames', video_length2, '--extension', 'png','--framegap', frmgp, '--precision', 'detailed_flow','--logpath',logpath], shell=True) #add choice for precision and add '--export_path', args.export_path
        else:
            if args.maskfile:
                subprocess.run(['python', tools_all, '--projectname', prjnm, '--frames', video_length2, '--extension', 'png', '--precision', 'detailed_flow','--logpath',logpath,'--mask', '1'], shell=True)
            else:
                subprocess.run(['python', tools_all, '--projectname', prjnm, '--frames', video_length2, '--extension', 'png', '--precision', 'detailed_flow','--logpath',logpath]) #add choice for precision and add '--export_path', args.export_path
elif args.precision == ['undetailed_flow']:
        if args.framegap:
            if args.maskfile:
                subprocess.run(['python', tools_all, '--projectname', prjnm, '--frames', video_length2, '--extension', 'png','--framegap', frmgp, '--precision', 'undetailed_flow','--logpath',logpath,'--mask', '1'], shell=True)
            else:
                subprocess.run(['python', tools_all, '--projectname', prjnm, '--frames', video_length2, '--extension', 'png','--framegap', frmgp, '--precision', 'undetailed_flow','--logpath',logpath], shell=True) #add choice for precision and add '--export_path', args.export_path
        else:
            if args.maskfile:
                subprocess.run(['python', tools_all, '--projectname', prjnm, '--frames', video_length2, '--extension', 'png', '--precision', 'undetailed_flow','--logpath',logpath,'--mask', '1'], shell=True)
            else:
                subprocess.run(['python', tools_all, '--projectname', prjnm, '--frames', video_length2, '--extension', 'png', '--precision', 'undetailed_flow','--logpath',logpath], shell=True) #add choice for precision and add '--export_path', args.export_path
else:
    print("webcam_test, normal, normal_slow don't use movement prediction, skipping..")



print("")
print("")
print("!!!!!important!!!!!")
print("")
print("a reminder to put the styled export frames into", train_output)
print("")
print("")
print("")
print("")
print("")

if args.precision == ['detailed_flow']:
    window=sg.Window('READ',
                     [ [sg.Text('the frame movement prediction takes a while to render but it renders on CPU, so u can create the export frames with effects with your GPU in the meanwhile')],
                       [sg.Text('COPY THE PATHS IN THE STATUS WINDOW AND DO WHAT IT SAYS')],
                       #[sg.Text('preparation done, apply desired effects on the images that are in "',int(train_filtered),'" and export those to "',int(train_output),'"')],
                       [sg.Text('have your read the above and understand? press Submit if you do')],
                       [sg.Submit()]])
    event, values = window.Read()
    window.Close() 
elif args.precision == ['undetailed_flow']:
    window=sg.Window('READ',
                     [ [sg.Text('the frame movement prediction takes a while to render but it renders on CPU, so u can create the export frames with effects with your GPU in the meanwhile')],
                       [sg.Text('COPY THE PATHS IN THE STATUS WINDOW AND DO WHAT IT SAYS')],
                       #[sg.Text('preparation done, apply desired effects on the images that are in "',int(train_filtered),'" and export those to "',int(train_output),'"')],
                       [sg.Text('have your read the above and understand? press Submit if you do')],
                       [sg.Submit()]])
    event, values = window.Read()
    window.Close() 
#export_done = (input("are you done with creating the styled frames? press ENTER to start patch based training"))
print("")
print("")
print("")
print("")
print("")
"""if export_done:
    print("")
    
    print("")"""
imageread1 = train_output +'001.png'
import cv2
img1 = cv2.imread(imageread1)
log_interval = args.log_interval
reconstruction_weight = args.reconstruction_weight
adversarial_weight = args.adversarial_weight
perception_loss_weight = args.perception_loss_weight
if args.append_smoothers == ['True'] :
    append_smoothers = 'True'
elif args.append_smoothers == ['False'] :
    append_smoothers = 'False'
    
if args.use_normalization == ['False']:
    use_normalization = 'False'
elif args.use_normalization == ['True']:
    use_normalization = 'True'
if args.use_image_loss == ['False']:
    use_image_loss = 'False'
elif args.use_image_loss == ['True']:
    use_image_loss = 'True' 
if args.tanh == ['False']:
    tanh = 'False'
elif args.tanh == ['True']:
    tanh = 'True'
if args.use_bias == ['False']:
    use_bias = 'False'
elif args.use_bias == ['True']:
    use_bias = 'True'
    

if args.filters_layers == ['[32, 64, 128, 128, 128, 64]'] :
    filters_layers = '326412812812864'
elif args.filters_layers == ['[32, 32, 32, 32, 32, 32]'] :
    filters_layers = '323232323232'
elif args.filters_layers == ['[32, 64, 64, 64, 64, 64]'] :
    filters_layers = '326464646464'
    
if args.patch_size == ['128']:
    patch_size = '128'
elif args.patch_size == ['16']:
    patch_size = '16'
elif args.patch_size == ['32']:
    patch_size = '32'
elif args.patch_size == ['64']:
    patch_size = '64'
    
#print(img1.shape)
#print(args.H, args.W, "3")
if( img1.shape != (args.H, args.W, 3) ):
    subprocess.run(["magick","mogrify", "-resize", resizesize, "-quality", "100", train_output_batch], shell=True) # magick mogrify -resize 512x1024! -quality 100 C:\deepdream-test\Few-Shot-Patch-Based-Training-master\logs\kind_train\output/*
else:
    print("W and H are the same, skipping resize")
if args.precision == ['detailed_flow']:
        print("results will appear in ",disco1010path,"every" ,log_interval,"steps")
        print("")
        print("")
        print('python', '-B', trainur, '--config', disco1010, '--data_root', train_root, '--log_interval', log_interval, '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath,'--perception_loss_weight',perception_loss_weight,'--reconstruction_weight',reconstruction_weight,'--adversarial_weight',adversarial_weight,'--append_smoothers',append_smoothers,'--filters_layers',filters_layers,'--patch_size',patch_size,'--use_normalization',use_normalization, '--use_image_loss',use_image_loss, '--tanh',tanh, '--use_bias',use_bias)
        print("")
        print("")
        subprocess.run(['python', '-B', trainur, '--config', disco1010, '--data_root', train_root, '--log_interval', log_interval, '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath,'--perception_loss_weight',perception_loss_weight,'--reconstruction_weight',reconstruction_weight,'--adversarial_weight',adversarial_weight,'--append_smoothers',append_smoothers,'--filters_layers',filters_layers,'--patch_size',patch_size,'--use_normalization',use_normalization, '--use_image_loss',use_image_loss, '--tanh',tanh, '--use_bias',use_bias], shell=True)
elif args.precision == ['webcam_test']:
        print("results will appear in ",resppath,"every" ,log_interval,"steps")
        print("")
        print("")
        print('python', '-B', trainur, '--config', webcam, '--data_root', train_root, '--log_interval', log_interval, '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath,'--perception_loss_weight',perception_loss_weight,'--reconstruction_weight',reconstruction_weight,'--adversarial_weight',adversarial_weight,'--append_smoothers',append_smoothers,'--filters_layers',filters_layers,'--patch_size',patch_size,'--use_normalization',use_normalization, '--use_image_loss',use_image_loss, '--tanh',tanh, '--use_bias',use_bias)
        print("")
        print("")
        subprocess.run(['python', '-B', trainur, '--config', webcam, '--data_root', train_root, '--log_interval', log_interval, '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath,'--perception_loss_weight',perception_loss_weight,'--reconstruction_weight',reconstruction_weight,'--adversarial_weight',adversarial_weight,'--append_smoothers',append_smoothers,'--filters_layers',filters_layers,'--patch_size',patch_size,'--use_normalization',use_normalization, '--use_image_loss',use_image_loss, '--tanh',tanh, '--use_bias',use_bias], shell=True)
elif args.precision == ['undetailed_flow']:
        print("results will appear in ",disco1015path,"every" ,log_interval,"steps")
        print("")
        print("")
        print('python', '-B', trainur, '--config', disco1015, '--data_root', train_root, '--log_interval', log_interval, '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath,'--perception_loss_weight',perception_loss_weight,'--reconstruction_weight',reconstruction_weight,'--adversarial_weight',adversarial_weight,'--append_smoothers',append_smoothers,'--filters_layers',filters_layers,'--patch_size',patch_size,'--use_normalization',use_normalization, '--use_image_loss',use_image_loss, '--tanh',tanh, '--use_bias',use_bias)
        print("")
        print("")
        subprocess.run(['python', '-B', trainur, '--config', disco1015, '--data_root', train_root, '--log_interval', log_interval, '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath,'--perception_loss_weight',perception_loss_weight,'--reconstruction_weight',reconstruction_weight,'--adversarial_weight',adversarial_weight,'--append_smoothers',append_smoothers,'--filters_layers',filters_layers,'--patch_size',patch_size,'--use_normalization',use_normalization, '--use_image_loss',use_image_loss, '--tanh',tanh, '--use_bias',use_bias], shell=True)
elif args.precision == ['normal']:
        print("results will appear in ",resppath,"every" ,log_interval,"steps")
        print("")
        print("")
        print('python', '-B', trainur, '--config', normal, '--data_root', train_root, '--log_interval', log_interval, '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath,'--perception_loss_weight',perception_loss_weight,'--reconstruction_weight',reconstruction_weight,'--adversarial_weight',adversarial_weight,'--append_smoothers',append_smoothers,'--filters_layers',filters_layers,'--patch_size',patch_size,'--use_normalization',use_normalization, '--use_image_loss',use_image_loss, '--tanh',tanh, '--use_bias',use_bias)
        print("")
        print("")
        subprocess.run(['python', '-B', trainur, '--config', normal, '--data_root', train_root, '--log_interval', log_interval, '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath,'--perception_loss_weight',perception_loss_weight,'--reconstruction_weight',reconstruction_weight,'--adversarial_weight',adversarial_weight,'--append_smoothers',append_smoothers,'--filters_layers',filters_layers,'--patch_size',patch_size,'--use_normalization',use_normalization, '--use_image_loss',use_image_loss, '--tanh',tanh, '--use_bias',use_bias], shell=True)
elif args.precision == ['normal_slow']:
        print("results will appear in ",resfpath,"every" ,log_interval,"steps")
        print("")
        print("")
        print('python', '-B', trainur, '--config', normal2, '--data_root', train_root, '--log_interval', log_interval, '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath,'--perception_loss_weight',perception_loss_weight,'--reconstruction_weight',reconstruction_weight,'--adversarial_weight',adversarial_weight,'--append_smoothers',append_smoothers,'--filters_layers',filters_layers,'--patch_size',patch_size,'--use_normalization',use_normalization, '--use_image_loss',use_image_loss, '--tanh',tanh, '--use_bias',use_bias)
        print("")
        print("")
        subprocess.run(['python', '-B', trainur, '--config', normal2, '--data_root', train_root, '--log_interval', log_interval, '--log_folder', 'logs_reference_P','--projectname', prjnm,'--logpath',logpath,'--perception_loss_weight',perception_loss_weight,'--reconstruction_weight',reconstruction_weight,'--adversarial_weight',adversarial_weight,'--append_smoothers',append_smoothers,'--filters_layers',filters_layers,'--patch_size',patch_size,'--use_normalization',use_normalization, '--use_image_loss',use_image_loss, '--tanh',tanh, '--use_bias',use_bias], shell=True)
        