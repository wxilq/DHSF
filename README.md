# DHSF Project Summary  
An enhanced video semantic segmentation framework based on SegFormer supports 4-level semantic hierarchy for video scene understanding.
## Core Architecture  
Base Model: SegFormer (transformer-based semantic segmentation)

Target: Video semantic segmentation with 4-level semantic hierarchy: Pixel → Object → Room → Scene

## Key features  
- 4 semantic levels: Pixel → Object → Room → Scene 
- Video Temporal Modeling: Specialized Mechanisms for Temporal Feature Extraction and Fusion with scene boundary detection and adaptive state resetting.
- Cross-level Interaction: Bidirectional state transfer and attention mechanisms between levels
- Progressive Training: A multi-stage training strategy that progressively optimizes model performance
- AI2Thor Integration: Complete indoor Scene dataset support  

## Requirements  
- Python >= 3.7 
- PyTorch == 1.7.0+cu110  
- CUDA >= 11.0
- MMSegmentation
- MMCV >= 1.1.4, <= 1.3.0  

## Install Dependencies  
pip install -r requirements

## Data Preparation  
python ai2thor/ai2thor_dataseg.py

## Training  
python train.py --config configs/your_config.py --work-dir work_dirs/experiment_name

## Distributed Training  
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/your_config.py --launcher pytorch

## Dataset Format

- `data/`
  - `ai2thor/`
    - `images/`
    - `annotations/`
    - `hierarchical_labels/`
      - `pixel/`
      - `object/`
      - `room/`
      - `scene/`
    - `video_sequences/`      
