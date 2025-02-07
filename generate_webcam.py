import argparse
import os
from custom_transforms import *
from PIL import Image
import numpy as np
import torch.utils.data
import torch
import torch.nn as nn
import cv2


# Main to generate images
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="checkpoint location", required=True)
    parser.add_argument("--device", help="device", required=True)
    parser.add_argument("--resolution", type=int, nargs=2, metavar=('width', 'height'), default=(480, 640))
    parser.add_argument("--show_original", type=int, default=0)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--webcam_number", type=int, default=0)
    parser.add_argument("--video_path", type=str)
    args = parser.parse_args()
    
    
    generator = (torch.load(args.checkpoint, map_location=lambda storage, loc: storage))
    generator.eval()

    device = args.device
    print("device: " + device, flush=True)

    generator = generator.to(device)
    if device.lower() != "cpu":
        generator = generator.type(torch.half)

    transform = build_transform()
    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(args.webcam_number)
    width, height = args.resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #cap.set(cv2.CAP_PROP_CHANNEL, 1)
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            exit()
        frame_counter += 1
        #If the last frame is reached, reset the capture and the frame_counter
        if args.video_path:
            if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0 #Or whatever as long as it is the same as next line
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        x = int(frame.shape[0] / 2)
        y = int(frame.shape[1] / 2)
        res = min(x, y)
        frame = frame[x-res:x+res, y-res:y+res, :]
        frame_resized = cv2.resize(frame, (args.resize, args.resize))
        frame_resized = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)) #convert to PIL.Image for torchvision transforms
        net_in = transform(frame_resized).to(args.device).unsqueeze(0)

        if device.lower() != "cpu":
            net_in = net_in.type(torch.half)
        net_out = generator(net_in)
        
        im = ((net_out[0].clamp(-1, 1) + 1) * 127.5).permute((1, 2, 0)).cpu().data.numpy().astype(np.uint8)
        im = cv2.cvtColor(cv2.resize(im, (2*res, 2*res)), cv2.COLOR_RGB2BGR)
        if args.show_original == 1:
            im = np.concatenate((frame, im), axis=1)

        cv2.imshow("press q to exit", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
            