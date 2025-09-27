import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd

# ---- your project utilities (kept identical) -------------------------------
from utils.detection import Yolo2DObjectDetection
from utils.generic import DepthApproximation

# ---------------------------------------------------------------------------
# MonoViT wrapper
# ---------------------------------------------------------------------------

class MonoViTBackbone(torch.nn.Module):
    """
    Minimal runtime wrapper around the MonoViT depth network.
    Exposes a simple .infer_image(np.ndarray[h,w,3], input_size=(H,W)) -> depth(float32, meters-ish / relative)
    Assumes weights saved like monodepth2-style: encoder.pth and depth.pth under a folder.
    """

    WEIGHTS_FILENAMES = {
        "encoder": "encoder.pth",
        "depth": "depth.pth"
    }

    def __init__(self,
                 load_weights_folder: str,
                 mpvit_ckpt_path: str = "./ckpt/mpvit_small.pth",
                 device: str = None):
        super().__init__()

        # Device selection
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else \
                     'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)

        # --- Validate paths early
        if not os.path.isdir(load_weights_folder):
            raise FileNotFoundError(
                f"[MonoViT] load_weights_folder not found: {load_weights_folder}"
            )
        enc_pth = os.path.join(load_weights_folder, self.WEIGHTS_FILENAMES["encoder"])
        dep_pth = os.path.join(load_weights_folder, self.WEIGHTS_FILENAMES["depth"])
        if not os.path.isfile(enc_pth):
            raise FileNotFoundError(f"[MonoViT] Missing encoder weights: {enc_pth}")
        if not os.path.isfile(dep_pth):
            raise FileNotFoundError(f"[MonoViT] Missing depth weights: {dep_pth}")

        if not os.path.isfile(mpvit_ckpt_path):
            raise FileNotFoundError(
                f"[MonoViT] MPViT backbone checkpoint not found: {mpvit_ckpt_path}\n"
                f"Place the ImageNet-1K pretrained MPViT weights in ./ckpt/ "
                f"(see MonoViT README)."
            )

        # --- Imports from the MonoViT repo
        # Expect structure like monodepth2 with MPViT encoder + depth decoder
        # networks/monovit_encoder.py, networks/depth_decoder.py etc.
        try:
            from networks.monovit_encoder import MonoViTEncoderMPViT  # repo-provided
        except Exception:
            # Some forks name it mpvit_encoder.py; try a fallback
            try:
                from networks.mpvit_encoder import MonoViTEncoderMPViT
            except Exception as e:
                raise ImportError(
                    "[MonoViT] Could not import MonoViT MPViT encoder from the repo. "
                    "Ensure the official repository is on PYTHONPATH or this file lives inside it."
                ) from e

        try:
            from networks.depth_decoder import DepthDecoder
        except Exception as e:
            raise ImportError(
                "[MonoViT] Could not import DepthDecoder from the repo."
            ) from e

        # --- Build model
        # The exact constructor args follow MonoViT’s evaluation scripts (encoder + decoder)
        self.encoder = MonoViTEncoderMPViT()
        # Load MPViT ImageNet weights into the encoder (as per README)
        state_dict = torch.load(mpvit_ckpt_path, map_location="cpu")
        self.encoder.load_state_dict(state_dict, strict=False)

        # Depth decoder takes encoder output channels; common pattern from monodepth2/monovit
        # If the repo exposes encoder.num_ch_enc, use it; fallback to typical channels.
        if hasattr(self.encoder, "num_ch_enc"):
            num_ch_enc = self.encoder.num_ch_enc
        else:
            # Reasonable default; adjust if your encoder exposes other channels
            num_ch_enc = [32, 64, 128, 256, 512]

        self.depth_decoder = DepthDecoder(num_ch_enc=num_ch_enc, scales=range(4))

        # Load trained MonoViT weights (encoder finetune + decoder)
        encoder_dict = torch.load(enc_pth, map_location="cpu")
        depth_dict = torch.load(dep_pth, map_location="cpu")

        # For safety when keys are prefixed (e.g., 'module.')
        def strip_module_prefix(d):
            return {k.replace("module.", ""): v for k, v in d.items()}

        self.encoder.load_state_dict(strip_module_prefix(encoder_dict), strict=False)
        self.depth_decoder.load_state_dict(strip_module_prefix(depth_dict), strict=True)

        self.encoder.to(self.device).eval()
        self.depth_decoder.to(self.device).eval()

        # Normalization consistent with monodepth2/monovit eval
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Disable grads
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def infer_image(self, bgr_image: np.ndarray, input_size= (192, 640)) -> np.ndarray:
        """
        Args:
            bgr_image: np.uint8 (H, W, 3) in BGR (OpenCV)
            input_size: (H, W) expected by the trained MonoViT model (e.g., (192, 640), (320,1024), (384,1280))
        Returns:
            depth: np.float32 (H, W) disparity/depth map resized back to original size
        """
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("[MonoViT] Empty image passed to infer_image")

        # Resize & normalize (to RGB)
        H_in, W_in = int(input_size[0]), int(input_size[1])
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (W_in, H_in), interpolation=cv2.INTER_LINEAR)
        im = rgb_resized.astype(np.float32) / 255.0
        im = (im - self.mean) / self.std
        im = np.transpose(im, (2, 0, 1))  # CHW
        tens = torch.from_numpy(im).unsqueeze(0).to(self.device)  # 1x3xH×W

        # Encoder/decoder forward
        feats = self.encoder(tens)
        outputs = self.depth_decoder(feats)

        # Monodepth-style: use the highest-resolution scale (0) 'disp' prediction
        if "disp_0" in outputs:
            disp = outputs["disp_0"]
        elif "disp" in outputs:
            disp = outputs["disp"]
        else:
            # Fallback to the first tensor output
            disp = next(v for k, v in outputs.items() if isinstance(v, torch.Tensor))

        disp = torch.nn.functional.interpolate(
            disp, size=bgr_image.shape[:2], mode="bilinear", align_corners=False
        )
        disp = disp.squeeze().detach().cpu().numpy().astype(np.float32)

        # If you need metric depth, plug your calibration/scale here or let DepthApproximation do it
        # For now, return as-is (relative / normalized disparity-like)
        return disp


# ---------------------------------------------------------------------------
# MonoViT Inference class (mirrors your DepthAnythingV2Inference)
# ---------------------------------------------------------------------------

class MonoViTInference:
    def __init__(
        self,
        input_path,
        load_weights_folder='models/depth_models/MonoViT/tmp/mono_model/models/weights_19',
        mpvit_ckpt='./ckpt/mpvit_small.pth',
        input_size=(192, 640),
        outdir='results/monovit_depth',
        savenumpy=False,
        colormap_dir='',
        eval=False,
        depth_path='',
        stream=False,
        max_depth=80,
    ):
        """
        Args mirror your original file, adding:
        - load_weights_folder: folder containing {encoder.pth, depth.pth} (MonoViT-style)
        - mpvit_ckpt: ImageNet-1K pretrained MPViT weights path (as per MonoViT README)
        - input_size: (H, W) expected by the specific MonoViT model variant you use
        """
        self.input_path = input_path
        self.input_size = input_size
        self.outdir = outdir
        self.savenumpy = savenumpy
        self.colormap_dir = colormap_dir
        self.evalualte = eval
        self.depth_path = depth_path
        self.stream = stream
        self.max_depth = max_depth

        # Device selection (same policy as your file)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else \
                      'mps' if torch.backends.mps.is_available() else 'cpu'

        # Depth backbone
        self.monovit = MonoViTBackbone(
            load_weights_folder=load_weights_folder,
            mpvit_ckpt_path=mpvit_ckpt,
            device=self.DEVICE
        )

        # Your detector + depth utils
        self.model = Yolo2DObjectDetection('models/detection_models/bounding_box/test.pt')
        self.depth_approximator = DepthApproximation(max_depth=self.max_depth)

        self.process_files()

        # Optional colormap (matplotlib)
        self._cmap = None
        if self.colormap_dir:
            try:
                import matplotlib
                import matplotlib.cm as cm
                self._cmap = cm.get_cmap('inferno')
                os.makedirs(self.colormap_dir, exist_ok=True)
            except Exception:
                self._cmap = None

    # -----------------------------------------------------------------------
    def process_files(self):
        if os.path.isfile(self.input_path):
            self.filenames = [self.input_path]
        else:
            self.filenames = glob.glob(os.path.join(self.input_path, '**/*'), recursive=True)
            # Filter obvious non-media if needed
            self.filenames = [f for f in self.filenames if os.path.isfile(f)]
        os.makedirs(self.outdir, exist_ok=True)

    # -----------------------------------------------------------------------
    def _maybe_save_numpy_and_colormap(self, base_name: str, raw_depth: np.ndarray):
        # Save raw depth/disparity as .npy
        if self.savenumpy:
            os.makedirs(self.savenumpy, exist_ok=True)
            npy_path = os.path.join(
                self.savenumpy, f"{base_name}_raw_depth.npy"
            )
            np.save(npy_path, raw_depth.astype(np.float32))

        # Save colormap PNG if requested
        if self.colormap_dir and self._cmap is not None:
            depth = raw_depth.copy()
            # normalize to [0,1] for coloring
            dmin, dmax = float(depth.min()), float(depth.max())
            if dmax > dmin:
                depth = (depth - dmin) / (dmax - dmin)
            depth_u8 = (depth * 255.0).astype(np.uint8)
            # matplotlib returns RGB float [0..1]; convert to BGR uint8
            cm_img = (self._cmap(depth_u8)[:, :, :3] * 255.0).astype(np.uint8)
            cm_img = cm_img[:, :, ::-1]
            os.makedirs(self.colormap_dir, exist_ok=True)
            out_png = os.path.join(self.colormap_dir, f"{base_name}_colormap.png")
            cv2.imwrite(out_png, cm_img)

    # -----------------------------------------------------------------------
    def process_images(self):
        for k, filename in enumerate(self.filenames):
            print(f'Progress {k+1}/{len(self.filenames)}: {filename}')
            raw_image = cv2.imread(filename)
            if raw_image is None:
                print(f"[WARN] Could not read image: {filename}")
                continue

            # Inference
            depth = self.monovit.infer_image(raw_image, self.input_size)

            # Detection + depth attachment (your utilities)
            predictions = self.model.predict(raw_image)
            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_image, predictions)

            if self.evalualte:
                depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)

            # Save annotated frame
            base_name = f"frame_{k}"
            output_path = os.path.join(self.outdir, f"{base_name}.png")
            cv2.imwrite(output_path, depth_frame)

            # Optional extras
            self._maybe_save_numpy_and_colormap(os.path.splitext(os.path.basename(filename))[0], depth)

        print(f'Output saved to {self.outdir}')

    # -----------------------------------------------------------------------
    def process_video(self, fps=30):
        for k, filename in enumerate(self.filenames):
            raw_video = cv2.VideoCapture(filename)
            if not raw_video.isOpened():
                print(f"[WARN] Could not open video: {filename}")
                continue

            length = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = max(1.0, float(raw_video.get(cv2.CAP_PROP_FPS)) / 3.0)

            output_path = os.path.join(self.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

            frame = 1
            print(f'Progress {k+1}/{len(self.filenames)}: {filename}')
            while True:
                ret, raw_frame = raw_video.read()
                if not ret:
                    break

                depth = self.monovit.infer_image(raw_frame, self.input_size)
                predictions = self.model.predict(raw_frame)
                predictions = self.depth_approximator.depth(predictions, depth)
                depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)

                if self.evalualte:
                    depth_frame = self.depth_approximator.evaluate(self.depth_path, depth_frame, filename, predictions)

                out.write(depth_frame)
                print(f'Frame: {frame}/{length} complete')
                frame += 1

            raw_video.release()
            out.release()
        print(f'Output saved to {self.outdir}')

    # -----------------------------------------------------------------------
    def eval_stereo(self, fps=4):
        data = []
        # Adjust CSV path to yours
        df = pd.read_csv(r"F:\data\timestamps.csv")

        for k, _ in enumerate(self.filenames):
            file_name = os.path.join(self.input_path, f'frame_{k+4201}.jpg')
            print(f'Progress {k+1}/{len(self.filenames)}: {file_name}')

            raw_frame = cv2.imread(file_name)
            if raw_frame is None:
                print(f"[WARN] Could not read image: {file_name}")
                continue

            depth = self.monovit.infer_image(raw_frame, self.input_size)

            base_name = f"frame_{k+4201}"
            output_path = os.path.join(self.outdir, f'{base_name}.png')

            predictions = self.model.predict(raw_frame)
            predictions = self.depth_approximator.depth(predictions, depth)
            depth_frame = self.depth_approximator.annotate_depth_on_img(raw_frame, predictions)

            # For eval, you indicated evaluate returns (frame, predictions) in your stereo function
            maybe_eval = self.depth_approximator.evaluate(self.depth_path, depth_frame, file_name, predictions)
            if isinstance(maybe_eval, tuple) and len(maybe_eval) == 2:
                depth_frame, predictions = maybe_eval
            else:
                depth_frame = maybe_eval

            cv2.imwrite(output_path, depth_frame)

            for pred in predictions:
                row = [
                    df[df['frame_number'] == k+4201]['frame_number'].to_string(index=False).strip(),
                    df[df['frame_number'] == k+4201]['utc_timestamp'].to_string(index=False).strip(),
                    pred.get('x1'), pred.get('y1'), pred.get('x2'), pred.get('y2'),
                    pred.get('class'), pred.get('estimated_depth'), pred.get('actual_depth')
                ]
                data.append(row)

            if (k+1) == 20:
                break

        out_df = pd.DataFrame(data, columns=['Frame','UTC_time','x1','y1','x2','y2','CLASS','predicted_depth','actual_depth'])
        out_df.to_csv(os.path.join(self.outdir, 'result.csv'), index=False)
        print(f'Output saved to {self.outdir}')
