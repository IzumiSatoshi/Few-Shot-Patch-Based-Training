import argparse
import os
from PIL import Image
from custom_transforms import *
import numpy as np
import torch.utils.data
import time
from data import DatasetFullImages



# Main to generate images
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="checkpoint location", required=True)
    #parser.add_argument("--data_root", help="data root", required=False)
    #parser.add_argument("--dir_input", help="dir input", required=False)
    parser.add_argument("--dir_x1", help="dir extra 1", required=False)
    parser.add_argument("--dir_x2", help="dir extra 2", required=False)
    parser.add_argument("--dir_x3", help="dir extra 3", required=False)
    parser.add_argument("--outdir", help="output directory", required=True)
    parser.add_argument("--device", help="device", required=True)
    parser.add_argument("--channels", help="if you didn't use tools_all.py u can just use --channels 1, if you did use it use --channels 2", required=True)
    parser.add_argument('--projectname', type=str, help='name of the project_', required=True)
    args = parser.parse_args()
    
    data_path = os.path.expanduser('~\Documents\\visionsofchaos\\fewshot\\data')
    data_root = data_path + "\\" + args.projectname+"_gen"
    dir_input = "input_filtered"
    checkpoint = data_path + "\\" + "\\"+ args.projectname+"_train"+"\\"+"logs_reference_P"+"\\"+args.checkpoint
    
    generator = (torch.load(checkpoint, map_location=lambda storage, loc: storage))
    generator.eval()
    

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    device = args.device
    print("device: " + device, flush=True)

    generator = generator.to(device)
    if device.lower() != "cpu":
        generator = generator.type(torch.half)
    transform = build_transform()
    dataset = DatasetFullImages(data_root + "/" + dir_input, "ignore", "ignore", device,
                      dir_x1=data_root + "/" + args.dir_x1 if args.dir_x1 is not None else None,
                      dir_x2=data_root + "/" + args.dir_x2 if args.dir_x2 is not None else None,
                      dir_x3=data_root + "/" + args.dir_x3 if args.dir_x3 is not None else None,
                      dir_x4=None, dir_x5=None, dir_x6=None, dir_x7=None, dir_x8=None, dir_x9=None)

    imloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, drop_last=False)  # num_workers=4

    generate_start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(imloader):
            print('Batch %d / %d' % (i, len(imloader)))

            net_in = batch['pre'].to(args.device)
            if device.lower() != "cpu":
                net_in = net_in.type(torch.half)
            net_out = generator(net_in)

            #image_space_in = to_image_space(batch['image'].cpu().data.numpy())

            #image_space = to_image_space(net_out.cpu().data.numpy())
            image_space = ((net_out.clamp(-1, 1) + 1) * 127.5).permute((0, int(args.channels), 3, 1))
            image_space = image_space.cpu().data.numpy().astype(np.uint8)

            for k in range(0, len(image_space)):
                im = image_space[k] #image_space[k].transpose(1, 2, 0)
                Image.fromarray(im).save(os.path.join(args.outdir, batch['file_name'][k]))


    print(f"Generating took {(time.time() - generate_start_time)}", flush=True)

