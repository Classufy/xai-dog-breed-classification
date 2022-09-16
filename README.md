# **XAI based Dog breed classification**   

## **💡프로젝트 요약**
반려견 품종을 구분하는 딥러닝 모델을 XAI를 이용하여 설명함으로써 분류의 근거를 얻는다. 이를 통해 펫샵과 같은 반려견 분양이 목적인 기관에서 품종에 대한 신뢰의 근거를 제시할 수 있게 된다.  



## **💡프로젝트 개요**  
 최근에, 펫샾에서 별다른 근거 없이 문자 혹은 말로만 품종에 대한 정보를 전달하여 더 비싼 품종으로 속이는 경우가 사회적 문제로 대두되고 있다. 이에 따라 Deep Learning를 이용하여 품종 구별에 대한 연구가 활발하게 진행되고 있다. 하지만 Deep Learning를 이용한 품종 구별에 대해 근거를 제시하지 않는데에 일반적인 사용자가 이해하기 어렵다는 한계가 있다. 
 이를 위해 반려견의 품종을 구별하는 CNN 기반의 딥러닝 모델을 구축한 후 이를 XAI 기법 중 하나인 LIME을 이용하여 분류 근거를 이미지로 보여준다. 
 
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
- [src/lime.ipynb](https://github.com/Classufy/xai-dog-breed-classification/blob/master/src/lime.ipynb)에서 img_path를 원하는 이미지 경로로 변경 후 explain_image(img_path) 실행
```
img_path = 'image_path' # 경로 수정
explain_image(img_path)
```
- 실행 결과
![output](https://user-images.githubusercontent.com/66214527/171214856-f0522b0e-e14a-496c-8495-f1aeb67935a3.png)

## **💡모델 성능** 
- Model : [/assets/best_model.h5](https://github.com/Classufy/xai-dog-breed-classification/blob/master/assets/best_model.h5)
- Test loss: 0.1255
- Test accuracy: 0.0.9583
- last update: 22/08/20

## **데이터셋** 
총 10종 선정
- beagle 762장
- cocker spaniel 756장
- golden retriever 756장
- maltese 770장
- pekinese 746장
- pomeranian 762장
- poodle 756장
- samoyed 756장
- shih-tzu 761장
- white terrier 736장

## Paper
[XAI 기반 반려견 품종 분류](http://www.riss.kr/search/detail/DetailView.do?p_mat_type=1a0202e37d52c72d&control_no=084eadfa92c7c31cc85d2949c297615a&keyword=%EB%B0%95%EB%AF%BC%EA%B7%9C%20xai)

## 개발기
[개발기](https://mangu.tistory.com/204)
