<h1 align="center">
    <p>LeRobot: State-of-the-art AI for real-world robotics</p>
</h1>

> [!NOTE]
> This repository supports converting datasets from OpenX format to LeRobot V2.0 dataset format.
> 
> Current script is now compatible with LeRobot V2.1.

## 🚀 What's New in This Script

In this dataset, we have made several key improvements:

- **OXE Standard Transformations** 🔄: We have integrated OXE's standard transformations to ensure uniformity across data.
- **Alignment of State and Action Information** 🤖: State and action information are now perfectly aligned, enhancing the clarity and coherence of the dataset.
- **Robot Type and Control Frequency** 📊: Annotations have been added for robot type and control frequency to improve dataset comprehensibility.
- **Joint Information** 🦾: Joint-specific details have been included to assist with fine-grained understanding.

Dataset Structure of `meta/info.json`:

```json
{
  "codebase_version": "v2.1", // lastest lerobot format
  "robot_type": "franka", // specific robot type, unknown if not provided
  "fps": 3, // control frequency, 10 if not provided
  // will add an additional key "control_frequency"
  "features": {
    "observation.images.image_key": {
      "dtype": "video",
      "shape": [128, 128, 3],
      "names": ["height", "width", "rgb"], // bgr to rgb if needed
      "info": {
        "video.fps": 3.0,
        "video.height": 128,
        "video.width": 128,
        "video.channels": 3,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "has_audio": false
      }
    },
    "observation.state": {
      "dtype": "float32",
      "shape": [8],
      "names": {
        "motors": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"] 
        // unified 8-dim vector: [xyz, state type, gripper], motor_x if not provided
      }
    },
    "action": {
      "dtype": "float32",
      "shape": [7],
      "names": {
        "motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"] 
        // unified 7-dim vector: [xyz, action type, gripper], motor_x if not provided
      }
    }
  }
}
```

## Installation

Download lerobot code:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

Install 🤗 LeRobot:

```bash
pip install -e .
```

## Get started

> [!IMPORTANT]  
> 1.Before running the following code, modify `save_episode()` function in lerobot.
> ```python
> def save_episode(self, episode_data: dict | None = None, keep_images: bool | None = False) -> None:
>     ...
>     # delete images
>     if not keep_images:
>         img_dir = self.root / "images"
>         if img_dir.is_dir():
>             shutil.rmtree(self.root / "images")
>     ...
> ```
> 2.for `bc_z` dataset, modify `encode_video_frames()` in `lerobot/common/datasets/video_utils.py`.
> 
> ```python
> # add the following content to line 141:
> vf: str = "pad=ceil(iw/2)*2:ceil(ih/2)*2",
> # Add the following content to line 171:
> ffmpeg_args["-vf"] = vf
> ```

> [!TIP]
> We recommend using `libsvtav1` as the vcodec for ffmpeg when encoding videos during dataset conversion.

Compile FFmpeg with libsvtav1 encoder (Optional):

`libsvtav1` is only supported in higher version of ffmpeg, so many users need to compile ffmpeg to enable it. You can follow this [link](https://trac.ffmpeg.org/wiki/CompilationGuide) for detailed compilation instructions..

Download source code:

```bash
git clone https://github.com/Tavish9/openx2lerobot.git
```

Modify path in `convert.sh`:

```bash
python openx_rlds.py \
    --raw-dir /path/to/droid/1.0.0 \
    --local-dir /path/to/LEROBOT_DATASET \
    --repo-id your_hf_id \
    --use-videos \
    --push-to-hub
```

Execute the script:

```bash
bash convert.sh
```

## Available OpenX_LeRobot Dataset

We have upload most of the OpenX datasets in [huggingface](https://huggingface.co/IPEC-COMMUNITY)🤗.

You can visualize the dataset in this [space](https://huggingface.co/spaces/IPEC-COMMUNITY/openx_dataset_lerobot_v2.0).

## Acknowledgment

Special thanks to the [Lerobot teams](https://github.com/huggingface/lerobot) for making this great framework.
