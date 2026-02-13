# DeepStream YOLOv13 Object Detection

Run **YOLOv13** object detection using [NVIDIA DeepStream SDK 8.0](https://developer.nvidia.com/deepstream-sdk) and [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo).


## Quick Start

### 1. Pull the Docker image

```bash
docker pull manasi1096/deepstream8-python:h100
```

### 2. Start the container with Jupyter Lab

```bash
docker run -d --gpus all -p 8888:8888 -v $(pwd):/workspace \
  manasi1096/deepstream8-python:h100 \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Get the Jupyter URL:

```bash
docker logs <container_id> 2>&1 | grep "http://127.0.0.1:8888/lab?token="
```

### 3. Open and run the notebook

Navigate to `DeepStream_YOLOv13_Detection.ipynb` in Jupyter Lab and run all cells.

The notebook handles everything automatically:
1. Clones [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) and [YOLOv13](https://github.com/iMoonLab/yolov13)
2. Installs the YOLOv13 package (fork of ultralytics with custom modules)
3. Downloads `yolov13s.pt` weights and exports to ONNX
4. Builds the DeepStream custom inference library
5. Runs detection on a sample video and saves `output_yolov13.mp4`

## Important Notes

- **First run takes ~10 minutes** (package installs + TensorRT engine build). Subsequent runs reuse the cached engine.
- **YOLOv13 must be installed as a package** (`pip install -e .` from the cloned repo). Vanilla `ultralytics` cannot load YOLOv13 models.
- **ONNX export**: Use `--opset 18` and do **not** use `--simplify`. PyTorch 2.10+ stores weights externally (`.onnx.data`), which must be inlined into a single `.onnx` file for TensorRT.
- **Headless mode**: The config uses file output (`sink type=3`) instead of display (`type=2`), so it works in containers without a display.

## Manual Steps (without the notebook)

```bash
# Inside the container
cd /workspace

# Clone repos
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
git clone --depth 1 https://github.com/iMoonLab/yolov13.git

# Install YOLOv13 + deps
cd yolov13
pip install -e .
pip install onnx onnxslim onnxruntime onnxscript huggingface_hub

# Download weights and export
wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13s.pt
cp ../DeepStream-Yolo/utils/export_yoloV13.py .
python3 export_yoloV13.py -w yolov13s.pt --dynamic --opset 18

# Inline ONNX weights
python3 -c "
import onnx
m = onnx.load('yolov13s.onnx', load_external_data=True)
onnx.save(m, '../DeepStream-Yolo/yolov13s.onnx')
"
cp labels.txt ../DeepStream-Yolo/

# Build custom lib
cd ../DeepStream-Yolo
export CUDA_VER=12.8
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo

# Run detection
deepstream-app -c deepstream_app_config_yolov13.txt
```

## Output

| File | Description |
|------|-------------|
| `output_yolov13.mp4` | Detection results (1280x720, 30fps, H.264) |
| `model_b1_gpu0_fp32.engine` | Cached TensorRT engine (reused on subsequent runs) |

## Model Info

| Property | Value |
|----------|-------|
| Model | YOLOv13s |
| Parameters | 9.0M |
| GFLOPs | 20.8 |
| Classes | 80 (COCO) |
| Input | 640x640 |
| Output | 8400 detections x 6 (bbox + score + label) |

## References

- [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) by marcoslucianops
- [YOLOv13](https://github.com/iMoonLab/yolov13) by iMoonLab
- [YOLOv13 DeepStream docs](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/docs/YOLOv13.md)
