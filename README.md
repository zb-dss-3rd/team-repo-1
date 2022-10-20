# ****도로 청소 로봇 개발을 위한 물 웅덩이 감지 프로젝트****

제로베이스 데이터 스쿨 3기 김영빈, 박재경

### 목차

1. [Description](#1-description)
2. [Information](#2-information)
3. [Models](#3-models)
4. [Performance](#4-performance)

## 1. Description

- 주제: 고여있는 물(형태가 일정하지 않은 물체) Segmentation
- 기획 의도
    - 도로 청소 로봇의 특성상 도로에 고여있는 물을 피해야하기 때문에 문제 발생
    - 청소를 해야하는 쓰레기와 물 웅덩이를 구분지어 물 웅덩이만을 피할 수 있도록 물 웅덩이를 감지하는 모델 개발
- 제작 기간
    - 2022.09.06 ~ 2022.10.19 총 43일간 진행
    
    | 일정 | 내용 |
    | --- | --- |
    | 09.06 ~ 09.30 | Data Labeling, Data Preprocessing |
    | 10.01 ~ 10.11 | 모델링, 학습 및 평가
    - Faster RCNN
    - Mask RCNN (epoch = 12)
    - UNet |
    | 10.13 ~ 10.17 | 모델링, 학습 및 평가
    - Mask RCNN (epoch = 24) |
    | 10.19 | 최종 발표 |

## 2. Information

### 1. Environment

> Colab Pro  
Python: 3.7.15  
GPU: K80, T4 or P100  
RAM: 24.45GB  
Pytorch: 1.12.1 cu113  
> 

### 2. Prerequisite

> MMCV
MMDetection  
import os  
import os.path as osp  
import mmcv  
from mmdet.apis import init_detector, inference_detector  
from mmdet.apis import set_random_seed  
from mmdet.apis import train_detector, show_results_pyplot  
from mmdet.apis import multi_gpu_test, single_gpu_test  
from mmdet.datasets.builder import DATASETS  
from mmdet.datasets.coco import CocoDataset  
from mmcv import Config  
from mmdet.datasets import build_dataset, build_dataloader,  
from mmdet.datasets import replace_ImageToTensor  
from mmdet.models import build_detector  
from mmcv.parallel import MMDataParallel   
from mmcv.parallel import MMDistributedDataParallel
> 

### 3. Data Set
  |구분|내용|
  |-|-|
  |영상출처|https://www.youtube.com/watch?v=_e_bzlDej5U&list=WL&index=15&t=209s&ab_channel=물축꾸리|
  |길이|원본 영상(18분 40초) 중 필요한 부분 2분 43초로 잘라서 진행|
  |해상도|1280*720 px|
  |프레임|원본 30fps에서 10fps로 추출하여 라벨링 및 학습 실행, 총 1627프레임 사용|

### 4. Folders

```
├── data
│   ├── Annotations
│   │   ├── COCODataset          # MMDetection에 최적화된 COCODataset 형태의 json 파일
│   │   └── OnetoOne             # 이미지와 일대일 대응되는 json 파일
│   │       ├── test_json
│   │       ├── train_json
│   │       └── val_json
│   └── Images
│       ├── MaskImages           # 라벨링을 토대로 만든 마스킹 이미지
│       │   ├── pixel_accuracy   # PA 계산을 진행한 3장의 이미지
│       │   │   ├── mask         # 라벨링을 토대로 만든 마스킹 이미지
│       │   │   ├── origin       # 모델이 예측한 이미지 (3차원)
│       │   │   └── output       # 모델이 예측한 이미지 (1차원)
│       │   ├── test
│       │   ├── train
│       │   │   └── mask_256     # (256, 256) 사이즈의 마스킹 이미지
│       │   └── validation
│       │       └── mask_256
│       ├── OriginalImages       # 모델 학습에 사용한 원본 이미지
│       └── Remainder            # 모델 학습에 사용하지 않은 나머지 이미지
├── ipynb                        # 모델을 구현할 때 사용한 ipynb 파일들
└── models                       # Mask RCNN 모델

```

## [3. Models](https://github.com/zb-dss-3rd/team-repo-1/tree/main/ipynb)

1. [Faster RCNN](https://github.com/zb-dss-3rd/team-repo-1/blob/main/ipynb/FasterRCNN.ipynb)
2. [Mask RCNN: 최종 사용 모델](https://github.com/zb-dss-3rd/team-repo-1/blob/main/ipynb/final_MaskRCNN.ipynb)
3. [UNet](https://github.com/zb-dss-3rd/team-repo-1/blob/main/ipynb/UNet.ipynb)

## 4. Performance
1. 결과 이미지
- train set inference image  
  <img width='663' src='https://user-images.githubusercontent.com/105214855/196437313-de2d7965-0405-4e6d-aa60-ebfc1202c836.png'>

- validation set inference image  
  <img width='663' src='https://user-images.githubusercontent.com/105214855/196437383-5ef76bcb-0cbf-47dc-a1a5-d4fad0c44b9a.png'>

- test set inference image  
  <img width='663' src='https://user-images.githubusercontent.com/105214855/196437489-d7a04a90-755c-4ce0-801a-4b44f2479d2d.png'>

2. AP(Average Precision)
- Mask RCNN 논문에 표기된 성능: AP = 35.7
  <img width="663" alt="Untitled (1)" src="https://user-images.githubusercontent.com/65541236/197061374-44dc501c-d1b8-4d7d-a398-33895a0e6e1f.png">  
- 프로젝트 최종 모델의 성능: AP = 36.6  
  <img width="663" alt="Untitled (2)" src="https://user-images.githubusercontent.com/65541236/197061477-b735e88b-41d8-4080-ab05-2a5a1a0d44bb.png">

3. PA(Pixel Accuracy)  
![그림1](https://user-images.githubusercontent.com/65541236/197062525-675c558c-03ed-4f29-b5a7-a2e9e5238d95.png)  
위에서부터 각각 92%, 97%, 76%를 보임  