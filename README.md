# Stagnant Water Detection
## 도로 청소 로봇 개발을 위한 물 웅덩이 감지 프로젝트
#### 제로베이스 데이터 사이언스 스쿨 3기 김영빈, 박재경

<br>

# 1. Discription
## ✔ 주제
  - 고여있는 물(형태가 일정하지 않은 물체) Segmentation
## ✔ 기획 의도
  - 도로 청소 로봇의 특성상 도로에 고여있는 물을 피해야하기 때문에 문제 발생
  - 청소를 해야하는 쓰레기와 물 웅덩이를 구분지어 물 웅덩이만을 피할 수 있도록 물 웅덩이를 감지하는 모델 개발
## ✔ Data Set
  |구분|내용|
  |-|-|
  |영상출처|https://www.youtube.com/watch?v=_e_bzlDej5U&list=WL&index=15&t=209s&ab_channel=물축꾸리|
  |길이|원본 영상(18분 40초) 중 필요한 부분 2분 43초로 잘라서 진행|
  |해상도|1280*720 px|
  |프레임|원본 30fps에서 10fps로 추출하여 라벨링 및 학습 실행, 총 1627프레임 사용|
  
## ✔ 환경
  - Colab Pro 사용
  - RAM : 고용량 RAM 설정(24.45GB)
  
## ✔ 프로젝트 일정
  - 09.06 ~ 10.19 총 43일간 진행
  
  |일정|내용|
  |-|-|
  |09.06 ~ 09.30|Data Labeling, COCO dataset Preprocessing|
  |10.01 ~ 10.11|모델링, 학습 및 평가 <br> - Faster RCNN <br> - Mask RCNN(epoch=12) <br> - UNet
  |10.13 ~ 10.17|모델 구현, 학습 및 평가 <br> - Mask RCNN(epoch=24)|
  |10.19|최종 발표|

<br>

# 2. Model Architecture
##### 최종 사용 모델은 Mask RCNN입니다.

<br>

## ✔ Mask RCNN
![image](https://user-images.githubusercontent.com/105214855/196432598-a1369ce9-8261-4c87-8465-43b46279076e.png)

Mask RCNN은 Object Detection에서 주로 사용되던 Faster RCNN에 Mask Branch를 추가한 모델입니다.

이 Mask RCNN은 Classification, Bounding Box Regression, Predicting Object Mask를 동시에 처리합니다.

아래는 Mask RCNN을 사용한 결과입니다.

![image](https://user-images.githubusercontent.com/105214855/196433402-ef55df8d-603f-440f-96ad-b381f3d8dd2f.png)

초록색 Bounding box와 확률, Masking이 된 결과물

## ✔ 사용한 모델의 기본 사항
##### 최종 모델은 epoch=24번이지만, 현재(10월 16일) 12번까지 완료되었기 때문에 epoch=12번 모델 기준으로 작성합니다.
  |구분|내용|
  |-|-|
  |Framework|MMDetection|
  |Back Bone|Resnet 101과 FPN 결합, pretained model 사용|
  |Batchsize|16|
  |Optimizer|SGD|
  |Learning rate|0.0025|
  |epoch|12, 24|
  |Image Size|1280 x 720 원본 그대로 사용, model의 input size는 1333 x 800으로 여백은 검은색으로 자동 패딩되어 진행됨|
  |학습 소요 시간|epoch 12 : 128min / epoch 24 : 254min|
  

##  ✔ Modeling

- Config 설정 코드
```
cfg.dataset_type = 'WATERDataset'
cfg.data_root = '/content/drive/MyDrive/water detection/WATER_Dataset_coco/'

cfg.data.train.type = 'WATERDataset'
cfg.data.train.data_root = '/content/drive/MyDrive/water detection/WATER_Dataset_coco/'
cfg.data.train.ann_file = 'train.json'
cfg.data.train.img_prefix = ''

cfg.data.val.type = 'WATERDataset'
cfg.data.val.data_root = '/content/drive/MyDrive/water detection/WATER_Dataset_coco/'
cfg.data.val.ann_file = 'val.json'
cfg.data.val.img_prefix = ''

cfg.data.test.type = 'WATERDataset'
cfg.data.test.data_root = '/content/drive/MyDrive/water detection/WATER_Dataset_coco/'
cfg.data.test.ann_file = 'test.json'
cfg.data.test.img_prefix = ''

# class의 갯수 설정
cfg.model.roi_head.bbox_head.num_classes = 2
cfg.model.roi_head.mask_head.num_classes = 2

# pretrained 모델
cfg.load_from = '/content/drive/MyDrive/mmdetection/checkpoints/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정
cfg.work_dir = '/content/drive/MyDrive/water detection/WATER_Dataset_coco/tutorial_exps'

# 학습율 변경 환경 파라미터 설정
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# CocoDataset의 경우 metric을 bbox로 설정(mAP아님. bbox로 설정하면 mAP를 iou threshold를 0.5 ~ 0.95까지 변경하면서 측정)
cfg.evaluation.metric = ['bbox', 'segm']
cfg.evaluation.interval = 12
cfg.checkpoint_config.interval = 12

# epoch 횟수 변경(기본값 = 12)
cfg.runner.max_epochs = 12

# 두번 config를 로드하면 lr_config의 policy가 사라지는 오류로 인하여 설정
cfg.lr_config.policy='step'
# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# ConfigDict' object has no attribute 'device 오류 발생시 반드시 설정 필요 https://github.com/open-mmlab/mmdetection/issues/7901
cfg.device='cuda'
```

- Model Learning
```
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# epochs는 config의 runner 파라미터로 지정, 기본 12회 
train_detector(model, datasets, cfg, distributed=False, validate=True)
```

<br>

# 3. 성과

|구분|내용|
|-|-|
|Batch Size|코랩 버그로 인하여 1로 수행|
|평가 소요 시간|이미지 1장 당 0.18s|

## ✔ Evaluation
```
# config 수정, 코랩 버그로 인해 test set에서 batch size 1로 변경
cfg.data.samples_per_gpu = 1

# test용 Dataset과 DataLoader 생성
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(dataset, samples_per_gpu=cfg.data.samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)

# checkpoint 저장된 model 파일을 이용하여 모델을 생성
checkpoint_file = '/content/drive/MyDrive/water detection/WATER_Dataset_coco/tutorial_exps/epoch_12.pth'
model_ckpt = init_detector(cfg, checkpoint_file, device='cuda:0')

model_ckpt = MMDataParallel(model_ckpt, device_ids=[0])
# single_gpu_test() 를 호출하여 test데이터 세트의 interence 수행
# 위에서 만든 show_test_output 디렉토리에 interence 결과가 시각화된 이미지 저장
outputs = single_gpu_test(model_ckpt, data_loader, True, '/content/drive/MyDrive/water detection/WATER_Dataset_coco/show_test_output', 0.3)
metric = dataset.evaluate(outputs, metric=['bbox', 'segm'])
```

## ✔ 결과 이미지

- train set inference image
![image](https://user-images.githubusercontent.com/105214855/196437313-de2d7965-0405-4e6d-aa60-ebfc1202c836.png)


- validation set inference image
![image](https://user-images.githubusercontent.com/105214855/196437383-5ef76bcb-0cbf-47dc-a1a5-d4fad0c44b9a.png)


- test set inference image
![image](https://user-images.githubusercontent.com/105214855/196437489-d7a04a90-755c-4ce0-801a-4b44f2479d2d.png)

## ✔ 성능 지표 확인

![image](https://user-images.githubusercontent.com/105214855/196437597-f805ac09-1146-40a7-8789-d6d8d1e177e5.png)

![image](https://user-images.githubusercontent.com/105214855/196437701-833cf049-a6b3-450e-ad1c-8953511fe30f.png)

Mask RCNN 논문에서 밝힌 mask AP는 35.7로, Mask RCNN이 나오기 이전에 대회에서 우승한 모델의 성능보다 높습니다. 

이와 비교했을 때, **저희가 학습한 모델의 mask AP는 34.9**로 논문에서의 성능과 비슷한 정도입니다. 추후 epoch를 24번으로 늘려 성능이 조금 더 개선될 수 있다고 예상하고 있습니다.
