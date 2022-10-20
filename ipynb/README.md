# Readme

# Description

## 1. Faster RCNN

MMDetection의 Mask RCNN을 구현하기 전, 학습 차원으로 구현한 모델입니다.

| 사용한 프레임워크 | MMDetection |
| --- | --- |
| back bone | Resnet 50과 FPN 결합, pretrain된 모델 사용 |
| Batch size | 16 |
| Optimizer | SGD |
| Learning rate | 0.0025 |
| epoch | 12 |
| 이미지 크기 | 1280 x 720 원본 그대로 사용, model의 input size는 1333 x 800으로 여백은 검은색으로 자동 패딩되어 진행됨 |
| 학습 소요 시간 | 약 42분 소요 |

결과 이미지

- train set inference image  
    <img width='663' src='https://user-images.githubusercontent.com/65541236/197068643-79c1b5c9-726f-4c6f-be47-ce7bf4fed889.png'>
- validation set inference image  
    <img width='663' src='https://user-images.githubusercontent.com/65541236/197068740-6e3f21ce-82f5-4b04-8dae-3eee6ae0a1ff.png'>
- test set inference image  
    <img width='663' src='https://user-images.githubusercontent.com/65541236/197068785-88f04ba1-89f1-4ef0-a9b1-10554dd80c7f.png'>

## 2. Mask RCNN (final_MaskRCNN.ipynb)

| 사용한 프레임워크 | MMDetection |
| --- | --- |
| back bone | Resnet 101과 FPN 결합, pretrain된 모델 사용 |
| Batch size | 16 |
| Optimizer | SGD |
| Learning rate | 0.0025 |
| epoch | 24 |
| 이미지 크기 | 1280 x 720 원본 그대로 사용, model의 input size는 1333 x 800으로 여백은 검은색으로 자동 패딩되어 진행됨 |
| 학습 소요 시간 | 약 128분 소요 |

결과 이미지

- train set inference image  
    <img width='663' src='https://user-images.githubusercontent.com/105214855/196437313-de2d7965-0405-4e6d-aa60-ebfc1202c836.png'>
- validation set inference image  
    <img width='663' src='https://user-images.githubusercontent.com/105214855/196437383-5ef76bcb-0cbf-47dc-a1a5-d4fad0c44b9a.png'>
- test set inference image  
    <img width='663' src='https://user-images.githubusercontent.com/105214855/196437489-d7a04a90-755c-4ce0-801a-4b44f2479d2d.png'>

## 3. UNet

| 사용한 프레임워크 | Pytorch |
| --- | --- |
| Batch size | 4 |
| Optimizer | Adam |
| Learning rate | 0.01 |
| epoch | 50 |
| 이미지 크기 | 256 x 256 사용 |
| 학습 소요 시간 | 약 100분 소요 |

결과 이미지

- train, validation set inference image  
    <img width="663" alt="unet" src="https://user-images.githubusercontent.com/65541236/197068951-816704d9-f6e6-4df0-99e3-14e9a6003d90.png">