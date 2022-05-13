# **XAI base Dog breed classification**   

## **💡프로젝트 요약**
반려견 품종을 구분하는 딥러닝 모델을 XAI를 이용하여 설명함으로써 분류의 근거를 얻는다. 이를 통해 펫샵과 같은 반려견 분양이 목적인 기관에서 품종에 대한 신뢰의 근거를 제시할 수 있게 된다.  



## **💡프로젝트 개요**  
 반려동물에 대한 관심이 날이 갈수록 뜨거워지고있다. 우리나라는 현재 소득 수준의 상승과 1인 가구의 증가로 반려동물을 기르는 인구가 급증하면서 반려동물이 하나의 산업이 된 **‘펫코노미’** 시대가 도래했다. 
 
 
사람들은 반려견을 입양할 때 원하는 외모, 성격에 따른 **품종**을 선택한다. 심각한 경우 품종을 탓하며 반려견을 유기하는 경우가 많이 발생하므로 개의 품종은 반려견 산업에서 매우 중요한 부분이다.  


반려견의 입양은 펫샵과 같은 분양이 목적인 기관에서 많이 이루어진다. 이러한 기관에서는 대부분 문자 혹은 구어로만 품종에 대한 정보를 전달하여 더 비싼 품종으로 속이는 경우가 많다고 한다. 반려견 시장이 커질수록 이 부분에 대해 문제도 많이 제기될 것이다. 따라서 본 프로젝트에서는 **반려견 분양 과정에서 신뢰성을 보장하는 방법을 제시**한다.  


우선 우리나라에서 많이 키우는 개의 품종을 기본 데이터 셋으로 선정하고 이와 구분이 어려운 품종을 추가하여 최종적으로 [데이터 셋](#데이터셋)을 선정하였다. 반려견의 품종을 구별하는 [딥러닝 모델](https://github.com/Classufy/xai-dog-breed-classification/blob/master/src/model.py)을 CNN을 모델 중 하나인 Xception을 전이학습 하여 구축하고 이를 XAI 기법 중 하나인 [LIME](https://github.com/marcotcr/lime)을 이용하여 분류 근거를 이미지로 보여준다. 이를 통해 최종적으로 반려견을 입양하는 사람은 자신이 입양하려는 반려견의 품종에 대한 확신을 얻고, 분양 기관도 품종의 근거를 말하며 판매에 대한 신뢰성을 높일 수 있게 하는 것이 목적이다.  

## **💡버전**
```
TensorFlow 2.8.0
Python 3.9.10
matplotlib 3.5.1
lime 0.2.0.1
```

## **💡사용법**
1. Clone this project
```
git clone https://github.com/Classufy/xai-dog-breed-classification
```
2. Run 
- [src/model.py](https://github.com/Classufy/xai-dog-breed-classification/blob/master/src/model.py) 학습 코드 실행 or [Download this file](https://drive.google.com/file/d/1vpug2xGuRcLEhWfOuW02AHp-GImZJUBW/view?usp=sharing)
```
python3 ./src/model.py
```
- [src/lime.ipynb](https://github.com/Classufy/xai-dog-breed-classification/blob/master/src/lime.ipynb)에서 img_path를 원하는 이미지 경로로 변경 후 explain_image(img_path) 실행
```
img_path = 'image_path' # 수정
explain_image(img_path)
```
- 실행 결과
<img width="350" alt="스크린샷 2022-05-13 오후 2 28 14" src="https://user-images.githubusercontent.com/66214527/168217081-a32d9dfc-4c32-42e2-9596-8f1f68b2f347.png">



## **데이터셋** 
- beagle 1009장
- cocker spaniel 1294장
- golden retriever 1335장
- maltese 770장
- pekinese 1274장
- pomeranian 936장
- poodle 800장
- samoyed 1325장
- shih-tzu 1155장
- white terrier 736장

_추가 예정_
