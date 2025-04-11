# OpenX to LeRobot 

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

1. Install LeRobot:
  Follow instructions in [official repo](https://github.com/huggingface/lerobot?tab=readme-ov-file#installation).

2. Install others:
  For reading tfds/rlds, we need to install `tensorflow-datasets`:
    ```bash
    pip install tensorflow
    pip install tensorflow-datasets
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
> We recommend using `libsvtav1` as the vcodec for ffmpeg when encoding videos during dataset conversion. If you can't use libsvtav1 after installing lerobot, you need to compile it yourself. Follow this [link](https://trac.ffmpeg.org/wiki/CompilationGuide) for detailed compilation instructions.


1. Download source code:

    ```bash
    git clone https://github.com/Tavish9/openx2lerobot.git
    ```

2. Modify path in `convert.sh`:

    ```bash
    python openx_rlds.py \
        --raw-dir /path/to/droid/1.0.0 \
        --local-dir /path/to/LEROBOT_DATASET \
        --repo-id your_hf_id \
        --use-videos \
        --push-to-hub
    ```

3. Execute the script:

    ```bash
    bash convert.sh
    ```

## Available OpenX_LeRobot Dataset

We have upload most of the OpenX datasets in [huggingface](https://huggingface.co/IPEC-COMMUNITY)🤗.

You can visualize the dataset in this [space](https://huggingface.co/spaces/IPEC-COMMUNITY/openx_dataset_lerobot_v2.0).

