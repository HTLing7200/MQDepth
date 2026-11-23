# MQDepth

This repository provides the inference code for MQDepth (ACM MM 2026), a metric depth estimation method for RGB images.

The indoor and outdoor variants mainly differ in their maximum encoded depth: 50 m for indoor scenes and 512 m for outdoor scenes. Compared with the earlier version, this release adopts Vision Banana-style depth color encoding instead of three-channel grayscale images. We use the original image aspect ratio instead of explicitly encoding the stretch ratio.

Outdoor inference is currently available. Indoor inference code and model checkpoints will be released separately.

## Installation

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Alternatively, run `bash script/install.sh`. The scripts use the active Python environment; set `PYTHON_BIN` to select another interpreter.

Inference uses CUDA by default. To run on CPU, add `--device cpu` to the commands below.

## Inference

Provide an input image, a model weights directory, and the camera focal length:

```bash
bash script/infer_outdoor.sh \
  --weights /path/to/model \
  --image image.jpg \
  --fx 886.81 \
  --output results/depth.npy
```

`--fx` is the focal length in pixels at the original image resolution. Image resizing and focal-length conversion are handled internally. The output is a float32 NumPy array containing metric depth in meters at the original image resolution.

The default uses 20 inference steps and the median of 4 ensemble predictions. The script also saves colored depth maps and side-by-side RGB/depth previews. Use `--steps` and `--ensemble` to adjust inference settings.

To process a directory of images with the same focal length:

```bash
bash script/infer_outdoor.sh \
  --weights /path/to/model \
  --input-dir images \
  --fx 886.81 \
  --output results
```

The Python entry point is also available directly:

```bash
python infer/outdoor.py --weights /path/to/model --image image.jpg --fx 886.81 --output depth.npy --visualize
```

To colorize an existing depth array:

```bash
bash script/convert_depth.sh --mode colorize --input depth.npy --output depth.color.png
```

The same conversion script supports inverse depth, uint16 depth PNGs, color decoding, and focal-length mapping. Run it with `--help` for the available options.

Checkpoints will be released.

## Repository Layout

```text
models/             Model components and depth encoding
infer/              Inference, depth conversion, and visualization
script/
  install.sh        Dependency installation
  infer_outdoor.sh  Outdoor inference with visualization
  convert_depth.sh  Depth conversion and colorization
README.md
requirements.txt
```

## Release Plan

- ☑️ Release outdoor inference code
- ⬜ Release indoor inference code
- ⬜ Release model checkpoints
