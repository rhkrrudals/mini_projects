# Flask Framework에서 Webserver 구동 파일 
# - 파일명 : __init__.py

# 모듈 로딩 
from flask import Flask, render_template, request

# 모델 모듈 로딩 
import os 
import argparse

# 동영상 프레임 받아서 처리하는 부분
import cv2
from PIL import Image, ImageDraw
from torchvision import transforms

# 스레드 처리 부분 - 비디오 처리량(?)
import tqdm
import threading
import queue
import uuid

# 파일 저장
import jsonify


# 모델 처리 부분
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import time

from models.craft import CRAFT
from models.test_video import copyStateDict, test_net, blur_text_regions, VideoProcessor


# 사용자 정의 함수
def create_app():
    
    # 전역변수
    # Flask Web Server 인스턴스 생성
    APP = Flask(__name__)
    
    # 라우팅 기능 함수 : 입력한 URL를 해당 URL로 보내주는 기능
    @APP.route('/process-video', methods=['POST'])
    def upload_video():
        file = request.files['file']
        
        # 비디오 총 프레임 수 계산 
        temp_video = cv2.VideoCapture(file)
        total_frames = int(temp_video.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_video.release()
        
        # 파일 저장 
        file_path = f'./static/video/{file.filename}'
        file.save(file_path)
        
        # 비디오 처리기 생성 
        processor = VideoProcessor(total_frames=total_frames)
        
        # input_path = './static/video/video1.mp4'
        output_path = './static/processed/blurred_video_1m.mp4'
        
        # processor.videoRun(input_path, output_path)
        
        # 비동기 처리
        thread = threading.Thread(
            target=processor.videoRun,
            args=(file_path,output_path)
        )
        
        thread.start()
        
        return jsonify({
            'jobld': processor.job_id,
            'totalFrames' : total_frames
        })
        
    @APP.route('/process-status/<job_id>', methods=['GET'])
    def get_process_status(job_id):
        processor = processor.get(job_id)
        if not processor:
            return jsonify({'status':'not_found'}), 404
        
        return jsonify({
            'jobId':processor.job_id,
            'status':processor.status,
            'progress' : processor.current_frame,
            'currentFrames': processor.current_frame,
            'totalFrames' : processor.total_frames,
            'error':processor.error
            
        })
        

    return APP

if __name__ == "__main__":
    app = create_app()
    app.run()