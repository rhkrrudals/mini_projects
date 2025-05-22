import os
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from models.craft import CRAFT
import models.craft_utils as craft_utils
import models.file_utils as file_utils
import models.imgproc as imgproc
import uuid

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


# 텍스트 감지 및 블러링 설정 값
config = {
    'trained_model': './static/pths/craft_mlt_25k.pth',
    'text_threshold': 0.7,
    'low_text': 0.4,
    'link_threshold': 0.4,
    'cuda': False,
    'canvas_size': 1280,
    'mag_ratio': 1.5,
    'poly': False,
    'show_time': False,
    'test_folder': './data/test_img',
    'refine': False,
    'refiner_model': './static/craft_refiner_CTW1500.pth',
}

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, config['canvas_size'], interpolation=cv2.INTER_LINEAR, mag_ratio=config['mag_ratio'])
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if config['show_time'] : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def blur_text_regions(frame, boxes):
    '''
    텍스트 영역 블러 처리
    '''
    if boxes is None:
        return frame

    blurred_frame = frame.copy()
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        
        #print(pts)

        # 블록 다각형 마스크 생성
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # 블러 처리
        blurred_roi = cv2.medianBlur(frame, 99)
        blurred_roi = cv2.bitwise_and(blurred_roi, blurred_roi, mask=mask)

        # 블러 처리된 ROI를 원본 프레임에 합성
        inv_mask = cv2.bitwise_not(mask)
        blurred_frame = cv2.bitwise_and(blurred_frame, blurred_frame, mask=inv_mask)
        blurred_frame = cv2.add(blurred_frame, blurred_roi)
        
    return blurred_frame

class VideoProcessor:
    def __init__(self, total_frames):
        self.job_id = str(uuid.uuid4())
        self.total_frames = total_frames
        self.current_frame = 0
        self.status = 'processing'
        self.error = None
        
    def videoRun(self, input_path, ouput_path):
        try: 
            # input_video = './static/video/video1.mp4'
            # output_video = './static/processed/blurred_video_1m.mp4'
            
            input_video = input_path
            output_video = ouput_path
            
            net = CRAFT()
            print(f"Loading weights from checkpoint ({config['trained_model']})")
            net.load_state_dict(copyStateDict(
                torch.load(config['trained_model'], map_location='cuda' if config['cuda'] else 'cpu')))
            if config['cuda']:
                net = net.cuda()
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = False
            net.eval()

            # 비디오 캡처 및 설정
            cap = cv2.VideoCapture(input_video)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            file = os.path.basename(input_video)
            filename, ext = os.path.splitext(file)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if ext == 'MOV' : fourcc = cv2.VideoWriter_fourcc(*'MOV')
            
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

            # 프레임 처리
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                bboxes, polys,score_text = test_net(net, frame,  config['text_threshold'], config['link_threshold'], config['low_text'], config['cuda'], config['poly'], refine_net=None)
                blurred_frame = blur_text_regions(frame, bboxes)
                out.write(blurred_frame)

                frame_count += 1
                print(f'Processing frame {frame_count}', end='\r')

            cap.release()
            out.release()
            print(f'\nProcessed {frame_count} frames')
            
            # 바꿔주기
            self.status = 'completed'
        except Exception as e:
            self.status = 'error'
            self.error = str(e)


if __name__ == '__main__':
    processor = VideoProcessor()
    processor.videoRun()
