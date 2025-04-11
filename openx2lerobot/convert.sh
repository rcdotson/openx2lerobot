export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/envs/vla_next/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/envs/vla_next/lib/python3.10/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"

python openx_rlds.py \
    --raw-dir /oss/vla_ptm_hwfile/DATA/fine_tune/kitchen_banana/0.1.0 \
    --local-dir /cpfs01/shared/optimal/vla_next/LEROBOT_DATASET/Franka \
    --repo-id your_hf_id \
    --use-videos
    # --push-to-hub
