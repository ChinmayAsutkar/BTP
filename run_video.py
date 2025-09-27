import argparse

from depth_inference.depth_anything_v2 import DepthAnythingV2Inference
from depth_inference.monodepth2 import MonoDepth2Inference
from depth_inference.midas_inference import MiDaSInference
from depth_inference.unidepth import UniDepthInference
from depth_inference.zeodepth_inference import ZeoDepthInference
from depth_inference.hrdepth_inference import HRDepthInference
from depth_inference.depthfm_inference import DepthFMInference
from depth_inference.metric3d_inference import Metric3DInference
from depth_inference.marigold_inference import MarigoldInference

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Estimation Script')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input image or directory of images')

    args = parser.parse_args()

    # depth_anything = DepthAnythingV2Inference(input_path=args.input_path)
    # monodepth = MonoDepth2Inference(input_path=args.input_path)
    # midas = MiDaSInference(input_path=args.input_path)
    # unidepth = UniDepthInference(input_path=args.input_path)
    # zeodepth = ZeoDepthInference(input_path=args.input_path)
    # hrdepth = HRDepthInference(input_path=args.input_path)
    # depthfm = DepthFMInference(input_path=args.input_path)
    # metric3d = Metric3DInference(input_path=args.input_path)
    marigold = MarigoldInference(input_path=args.input_path)

    # depth_anything.process_video()
    # monodepth.process_video()
    # midas.process_video()
    # unidepth.process_video()
    # zeodepth.process_video()
    # hrdepth.process_video()
    # depthfm.process_video()
    # metric3d.process_video()
    marigold.process_video()
    