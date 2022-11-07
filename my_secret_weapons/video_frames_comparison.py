# !!!IMPORTANT!!! The output must be encoded once, e.g. with a video editor, as it may contain errors that make it unplayable as it is.
import cv2
import glob

#### change here
left_frames_dir_path ="logs/7_gen/res__P_disco1010/0020000"
right_frames_dir_path =  "logs/7_gen/input_filtered"
fps = 24
output_video_path = "miku_rotate.mp4" #mp4 only
input_frames_height = 512
input_frames_width = 512
#### 


fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(output_video_path,fourcc, fps, (input_frames_width*2, input_frames_height))

left_frames_path_list = glob.glob(left_frames_dir_path+"/*")
right_frames_path_list = glob.glob(right_frames_dir_path+"/*")

for left_frame_path, right_frame_path in zip(left_frames_path_list, right_frames_path_list):
    left_frame = cv2.imread(left_frame_path)
    right_frame = cv2.imread(right_frame_path)
    output_frame = cv2.hconcat([left_frame, right_frame])
    video.write(output_frame)

video.release()
print('done. enjoy:D')