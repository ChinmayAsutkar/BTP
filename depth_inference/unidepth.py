import cv2
import glob
import os
import torch
import pandas as pd
import numpy as np
from models.depth_models.Unidepth.UniDepth.models.unidepthv2 import UniDepthV2
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation

class UniDepthInference:
    def __init__(self, input_path, model_name='lpiccinelli/unidepth-v2-vitl14', 
                 outdir='results/unidepth', max_depth=80, savenumpy=False, 
                 colormap='', eval=False, depth_path='', stream=False):
        
        self.input_path = input_path
        self.outdir = outdir
        self.model_name = model_name
        self.max_depth = max_depth
        self.evaluate = eval
        self.depth_path = depth_path
        self.stream = stream
        self.savenumpy = savenumpy
        self.colormap = colormap
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Load UniDepth model - handles all preprocessing internally
        self.depth_model = UniDepthV2.from_pretrained(self.model_name)
        self.depth_model = self.depth_model.to(self.DEVICE).eval()

        # YOLO and depth approximation setup (same as your original)
        self.model = Yolo2DObjectDetection('models/detection_models/bounding_box/yolov8n.pt')
        self.depth_approximator = DepthApproximation(max_depth=self.max_depth)
        
        self.process_files()
    
    def preprocess_image_for_unidepth(self, raw_image):
        """
        UniDepth preprocessing:
        - Accepts raw BGR/RGB [0-255] from cv2.imread
        - Automatically handles normalization and camera estimation
        - No manual preprocessing required
        """
        # Convert BGR to RGB (cv2.imread loads as BGR)
        if len(raw_image.shape) == 3 and raw_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = raw_image
            
        # Convert to tensor format [C, H, W] - UniDepth expects this format
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()
        
        return rgb_tensor
    
    def infer_depth(self, raw_image):
        """
        UniDepth depth inference with automatic preprocessing
        """
        with torch.no_grad():
            # Preprocess for UniDepth
            rgb_tensor = self.preprocess_image_for_unidepth(raw_image)
            rgb_tensor = rgb_tensor / 255.0                # normalize to [0, 1]
            rgb_tensor = rgb_tensor.unsqueeze(0).to(self.DEVICE)  # add batch dim
            
            # UniDepth inference
            predictions = self.depth_model.infer(rgb_tensor)
            
            # Extract metric depth in meters (remove batch dimension)
            depth = predictions["depth"].squeeze().cpu().numpy()

            
            return depth

    
    def process_files(self):
        if os.path.isfile(self.input_path):
            self.filenames = [self.input_path]
        else:
            self.filenames = glob.glob(os.path.join(self.input_path, '/*'), recursive=True)
        os.makedirs(self.outdir, exist_ok=True)
    
    def process_images(self):
        for k, filename in enumerate(self.filenames):
            print(f'Progress {k+1}/{len(self.filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            
            # UniDepth depth inference
            depth = self.infer_depth(raw_image)
            
            output_path = os.path.join(self.outdir,'frame_{}.png'.format(k))
            
            # YOLO predictions and depth extraction (same as your original)
            predictions = self.model.predict(raw_image)
            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_image, predictions)
            
            if self.evaluate:
                depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)
            
            cv2.imwrite(output_path, depth_frame)
            
            # Save options (same as your original)
            if self.savenumpy:
                output_path = os.path.join(self.savenumpy, os.path.splitext(os.path.basename(filename))[0] + '_numpy_matrix' + '_raw_depth_meter_frame.npy')
                np.save(output_path, depth)
                
        print(f'Output saved to {self.outdir}')

    def process_video(self, fps=30):
        for k, filename in enumerate(self.filenames):
            raw_video = cv2.VideoCapture(filename)
            length = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))/3
            output_path = os.path.join(self.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
            frame=1
            print(f'Progress {k+1}/{len(self.filenames)}: {filename}')
            while raw_video.isOpened():
                ret, raw_frame = raw_video.read()
                if not ret:
                    break

                depth = self.infer_depth(raw_frame, self.input_size)

                predictions = self.model.predict(raw_frame)
                predictions = self.depth_approximator.depth(predictions, depth)
                depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)
                if self.evaluate:
                    depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)
                out.write(depth_frame)
                print(f'Frame: {frame}/{length} complete')
                frame+=1
            
            raw_video.release()
            out.release()
        print(f'Output saved to {self.outdir}')

    def eval_stereo(self, fps=4):
        # output_path = os.path.join(self.outdir,'video.mp4')
        # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
        data = []
        df = pd.read_csv(r"F:\data\timestamps.csv")
        for k, filename in enumerate(self.filenames):
            file_name = os.path.join(self.input_path, 'frame_{}.jpg'.format(k+4201))
            print(f'Progress {k+1}/{len(self.filenames)}: {file_name}')
            raw_frame = cv2.imread(file_name)
            depth = self.infer_depth(raw_frame)
            output_path = os.path.join(self.outdir,'frame_{}.png'.format(k+4201)) ## new change for getting output as annoted images instead of video file
            predictions = self.model.predict(raw_frame)
            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)
            depth_frame, predictions = self.depth_approximator.evaluate(self.depth_path, depth_frame, file_name, predictions)
            # out.write(depth_frame)
            cv2.imwrite(output_path,depth_frame)
            for pred in predictions:
                res = [df[df['frame_number']==k+4201]['frame_number'].to_string()[5:].strip(),df[df['frame_number']==k+4201]['utc_timestamp'].to_string()[5:].strip(),pred['x1'],pred['y1'],pred['x2'],pred['y2'], pred['class'], pred['estimated_depth'], pred['actual_depth']]               
                data.append(res)
            if(k+1==5):
                break
        df = pd.DataFrame(data, columns=['Frame','UTC_time','x1','y1','x2','y2','CLASS', 'predicted_depth', 'actual_depth'])
        df.to_csv('{}/result.csv'.format(self.outdir), index=False)
        # out.release()
        # out.release()
        print(f'Output saved to {self.outdir}')
        # out.release()
        # print(f'Output saved to {self.outdir}')
    
    # Include all your other methods (process_video, eval_stereo) with the same structure
    # Just replace self.depth_anything.infer_image() with self.infer_depth()