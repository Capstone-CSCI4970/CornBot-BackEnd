import io
import tarfile
import cv2
import torch.nn.functional as F
import torchvision.utils as utils
import numpy as np
import torch
import os
import shutil

import boto3
import gzip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def visualize_attn_softmax(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    N,C,W,H = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    #print('###############',a.size())
    #if up_factor > 1:
    #    a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None,scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    # if not os.path.exists(checkpoint):
    #     raise("File doesn't exist {}".format(checkpoint))
    
    path_local_checkpoint = "static/API/new"


   
    
    
    # s3_client = boto3.client("s3")
    
    # s3_object = s3_client.get_object(Bucket="cornleafimagebucket", Key="best.pth.tar")
    # wholefile = s3_object['Body'].read()
    # fileobj = io.BytesIO(wholefile)
    # tar_file = tarfile.open(fileobj=fileobj)
    s3 = boto3.resource('s3')   
 
    s3.Bucket('cornleafimagebucket').download_file(
        'best.pth.tar', 
         path_local_checkpoint
    )
    
    print("\nDownloading the model from s3\n")
    
    checkpoint = torch.load(path_local_checkpoint,map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Download Complete")
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #optimizer2.load_state_dict(checkpoint['optimizer_closs_state_dict'])
    
    if scheduler:
      scheduler.load_state_dict(checkpoint["scheduler_save"])

    return checkpoint