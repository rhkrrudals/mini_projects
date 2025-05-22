# 동영상 프레임 받아서 처리하는 부분
import cv2
from PIL import Image, ImageDraw
from torchvision import transforms
from model import EAST

# 모델 처리 부분
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import EAST
import os
from dataset import get_rotate_mat
import numpy as np
import lanms
import time


# 텍스트 detect 함수 부분 
def resize_img(img):
	'''resize image to be divisible by 32
	'''
	w, h = img.size
	resize_w = w
	resize_h = h

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	'''convert PIL Image to torch.Tensor
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(img, model, device):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''
	img, ratio_h, ratio_w = resize_img(img)
	with torch.no_grad():
		score, geo = model(load_pil(img).to(device))
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
	return img


def detect_dataset(model, device, test_img_path, submit_path):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		model        : detection model
		device       : gpu if gpu is available
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])
	
	for i, img_file in enumerate(img_files):
		print('evaluating {} image'.format(i), end='\r')
		boxes = detect(Image.open(img_file), model, device)
		seq = []
		if boxes is not None:
			seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
		with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg','.txt')), 'w') as f:
			f.writelines(seq)

       
# 영상처리 부분 
# text 부분 detect해주기
def detect_text_boxes(frame, model, device):
    '''
	프레임 내 텍스트 감지
    '''
    # PIL 이미지 변환
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # detect 하기
    boxes = detect(pil_frame,model,device)
    
    # print("Dectected boxes", boxes)
    return boxes


def blur_text_regions(frame,boxes):
    '''
	텍스트 영역 블러 처리
    '''		
    if boxes is None:
        return frame
    
    blurred_frame = frame.copy()
    for box in boxes:
        # 좌표 추출 - 정수(int) 변환 
        pts = np.array([
			[box[0], box[1]],
			[box[2], box[3]],
			[box[4], box[5]],
			[box[6], box[7]]
		], np.int32) 
        
        # 블록 다각형 마스크 생성
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        # ROI 추출
        # roi = cv2.bitwise_and(frame,frame,mask=mask)
        
        # 블러 처리 
        blurred_roi = cv2.medianBlur(frame, 99)
        blurred_roi = cv2.bitwise_and(blurred_roi, blurred_roi, mask=mask)
        
        # 블러 처리된 ROI를 원본 프레임에 합성
        inv_mask = cv2.bitwise_not(mask)
        blurred_frame = cv2.bitwise_and(blurred_frame, blurred_frame, mask=inv_mask)
        blurred_frame = cv2.add(blurred_frame, blurred_roi)
    
    return blurred_frame

def process_video(input_path, output_path, model, device):
    '''
	비디오의 모든 프레임에서 텍스트 블러 처리
    '''
        
    # 비디오 캡쳐 및 설정
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 비디오 writer 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width,height))
    
    # 프레임 처리
    frame_count = 0
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        
        # 텍스트 박스 감지
        boxes = detect_text_boxes(frame, model, device)
        
        # 텍스트 블러 처리
        blurred_frame = blur_text_regions(frame, boxes)
        
        # 출력 비디오에 쓰기
        out.write(blurred_frame)
        
        # frame 개수 확인
        frame_count += 1
        print(f'Processing frmae {frame_count}', end='\r')
        
    # 리소스 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'\nProcessed {frame_count} frames')  

     

if __name__ == '__main__':
	# img_path    = './ICDAR_2015/test_img/img_4.jpg'
	model_path  = './pths/east_vgg16.pth'
	# res_img     = './res3.bmp'
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	model = EAST().to(device)
	model.load_state_dict(torch.load(model_path, map_location='cpu'))
	model.eval()
 
	input_video = './video/video1.mp4'
	ouput_video = 'blurred_video_median.mp4'
	process_video(input_video, ouput_video, model, device)
 
 
	# img = Image.open(img_path)
	# boxes = detect(img, model, device)
	# plot_img = plot_boxes(img, boxes)	
	# plot_img.save(res_img)

# print(time.process_time())
# print(boxes)


