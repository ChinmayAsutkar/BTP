import argparse
import traceback
import os
import shutil
import csv

from depth_inference.depth_anything_v2 import DepthAnythingV2Inference
from depth_inference.monodepth2 import MonoDepth2Inference
from depth_inference.midas_inference import MiDaSInference
from depth_inference.unidepth import UniDepthInference
from depth_inference.zeodepth_inference import ZeoDepthInference
from depth_inference.hrdepth_inference import HRDepthInference
from depth_inference.depthfm_inference import DepthFMInference
from depth_inference.metric3d_inference import Metric3DInference
from depth_inference.marigold_inference import MarigoldInference


AVAILABLE_MODELS = [
    'depth_anything',
    'midas',
    'monodepth2',
    'unidepth',
    'zeodepth',
    'hrdepth',
    'depthfm',
    'metric3d',
    'marigold',
]

MODEL_OUTDIRS = {
    'depth_anything': 'results/depth_anything',
    'midas': 'results/midas',
    'monodepth2': 'results/monodepth2',
    'unidepth': 'results/unidepth',
    'zeodepth': 'results/zoedepth',
    'hrdepth': 'results/hrdepth',
    'depthfm': 'results/depthfm',
    'metric3d': 'results/metric3d',
    'marigold': 'results/marigold',
}

def build_infer(model_name: str, input_path: str, encoder: str, max_depth: int):
    if model_name == 'depth_anything':
        return DepthAnythingV2Inference(input_path=input_path, encoder=encoder, max_depth=max_depth)
    if model_name == 'midas':
        return MiDaSInference(input_path=input_path)
    if model_name == 'monodepth2':
        return MonoDepth2Inference(input_path=input_path)
    if model_name == 'unidepth':
        return UniDepthInference(input_path=input_path)
    if model_name == 'zeodepth':
        return ZeoDepthInference(input_path=input_path)
    if model_name == 'hrdepth':
        return HRDepthInference(input_path=input_path)
    if model_name == 'depthfm':
        return DepthFMInference(input_path=input_path)
    if model_name == 'metric3d':
        return Metric3DInference(input_path=input_path)
    if model_name == 'marigold':
        return MarigoldInference(input_path=input_path)
    raise ValueError(f'Unknown model: {model_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple depth models over images')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input image or directory of images')
    parser.add_argument('--models', type=str, default=','.join(AVAILABLE_MODELS),
                        help=f'Comma-separated subset of models to run. Options: {",".join(AVAILABLE_MODELS)}')
    parser.add_argument('--encoder', type=str, default='vitl', help='Encoder for DepthAnythingV2 (vits/vitb/vitl/vitg)')
    parser.add_argument('--max-depth', type=int, default=80, help='Max depth for annotation (meters)')
    parser.add_argument('--combined-outdir', type=str, default='results/combined', help='Directory to collect all outputs together')
    parser.add_argument('--summary-csv', type=str, default='', help='Path to combined summary CSV (defaults to <combined-outdir>/summary.csv)')
    parser.add_argument('--combined-metrics-csv', type=str, default='', help='Path to combined metrics CSV (defaults to <combined-outdir>/combined_results.csv)')
    args = parser.parse_args()

    os.makedirs(args.combined_outdir, exist_ok=True)
    summary_csv_path = args.summary_csv or os.path.join(args.combined_outdir, 'summary.csv')
    combined_metrics_csv_path = args.combined_metrics_csv or os.path.join(args.combined_outdir, 'combined_results.csv')
    summary_csv_exists = os.path.isfile(summary_csv_path)
    combined_metrics_exists = os.path.isfile(combined_metrics_csv_path)

    selected = [m.strip() for m in args.models.split(',') if m.strip()]
    # Prepare CSV writers
    csv_file = open(summary_csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not summary_csv_exists:
        csv_writer.writerow(['model', 'original_output_path', 'combined_output_path'])
    metrics_csv_file = open(combined_metrics_csv_path, 'a', newline='')
    metrics_csv_writer = csv.writer(metrics_csv_file)
    # Defer writing header for metrics until we read the first model result.csv
    wrote_metrics_header = combined_metrics_exists
    metrics_header = None

    for model_name in selected:
        print(f'\n===== Running model: {model_name} =====')
        try:
            model_outdir = MODEL_OUTDIRS.get(model_name)
            if model_outdir is None:
                raise ValueError(f'No default outdir mapping for model {model_name}')
            os.makedirs(model_outdir, exist_ok=True)
            before = set(os.listdir(model_outdir))

            infer = build_infer(model_name, args.input_path, args.encoder, args.max_depth)
            infer.process_images()

            after = set(os.listdir(model_outdir))
            new_files = sorted(list(after - before))
            for fname in new_files:
                src = os.path.join(model_outdir, fname)
                combined_name = f"{model_name}__{fname}"
                dst = os.path.join(args.combined_outdir, combined_name)
                try:
                    shutil.copy2(src, dst)
                    csv_writer.writerow([model_name, src, dst])
                except Exception as copy_err:
                    print(f"[WARN] Failed to copy {src} -> {dst}: {copy_err}")

            # Merge per-model metrics result.csv if present
            model_metrics_path = os.path.join(model_outdir, 'result.csv')
            if os.path.isfile(model_metrics_path):
                try:
                    with open(model_metrics_path, 'r', newline='') as mf:
                        reader = csv.reader(mf)
                        header = next(reader, None)
                        if header:
                            if not wrote_metrics_header:
                                metrics_csv_writer.writerow(['model'] + header)
                                wrote_metrics_header = True
                                metrics_header = header
                            for row in reader:
                                metrics_csv_writer.writerow([model_name] + row)
                except Exception as merge_err:
                    print(f"[WARN] Failed to merge metrics from {model_metrics_path}: {merge_err}")
            else:
                # Fallback: no model metrics CSV. Record discovered files as minimal metrics rows.
                try:
                    if not wrote_metrics_header:
                        # Create a simple header when no metrics file has defined one yet
                        metrics_header = ['file']
                        metrics_csv_writer.writerow(['model'] + metrics_header)
                        wrote_metrics_header = True
                    for fname in new_files:
                        combined_name = f"{model_name}__{fname}"
                        dst = os.path.join(args.combined_outdir, combined_name)
                        # Write a row placing file path into the first metrics column, pad others if needed
                        pad = [] if metrics_header is None else [''] * (len(metrics_header) - 1)
                        metrics_csv_writer.writerow([model_name, dst] + pad)
                except Exception as fb_err:
                    print(f"[WARN] Failed to write fallback metrics for {model_name}: {fb_err}")
        except Exception as e:
            print(f'[SKIP] {model_name}: {e}')
            traceback.print_exc()
            continue

    csv_file.close()
    metrics_csv_file.close()
    print(f'\nAll requested models processed. Combined outputs at: {args.combined_outdir}\nSummary CSV: {summary_csv_path}\nCombined metrics CSV: {combined_metrics_csv_path}')
