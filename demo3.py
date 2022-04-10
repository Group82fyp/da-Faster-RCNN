import torch, os, cv2
from PIL import Image

from model.model import parsingNet
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
row_anchor = culane_row_anchor
import os
griding_num = 200
cls_num_per_lane = 18




if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    dist_print('start testing...')

    test_folder = '/home/jiaxi/da-Faster-RCNN/results'

    net = parsingNet(pretrained = False, backbone='18' ,cls_dim = (griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load("culane_18.pth", map_location='cpu')['model']
    # state_dict = torch.load("ep049.pth", map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    list_files = os.listdir(test_folder)

    for count,img in enumerate(list_files):
        test_img = os.path.join(test_folder,img)
        frame = cv2.imread(test_img)

        img_h, img_w= frame.shape[0], frame.shape[1]
        # frame = cv2.resize(frame, (288, 800))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img_transforms(img)
        img = img.unsqueeze(0)  #

        imgs = img.cuda()
        with torch.no_grad():
            out = net(imgs)

        col_sample = np.linspace(0, 800 - 1, griding_num)
        col_sample_w = col_sample[1] - col_sample[0]


        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == griding_num] = 0
        out_j = loc

        # import pdb; pdb.set_trace()
        # vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
        colourno = 0
        for i in range(out_j.shape[1]):
            colour = [(0, 255, 0), (255, 0, 0), (0,0,255), (255, 255, 255)]

            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        # print("ppp:")
                        # print(ppp)
                        image = cv2.circle(frame,ppp,5,colour[colourno],-1)
            colourno = colourno + 1
            if colourno > 4:
                colourno = 0
        cv2.imwrite("output" + count+ ".png", image)
        # cv2.imshow("show", image)


        
