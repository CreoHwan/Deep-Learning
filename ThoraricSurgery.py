from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 

import numpy as np
import tensorflow as tf

Data_set = np.loadtxt("./dataset/ThoraricSurgery.csv", delimiter=",")

X = Data_set[:,0:17]
Y = Data_set[:,17]

# from sklearn.model_selection import train_test_split 제공되는 train_test_split( ) 기능 활용

# 학습 데이터 70 %, 테스트 데이터셋 30% 로 설정하기 
seed = 0
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)


#  딥러닝 모델 설계하기
# keras에서 제공되는 Sequential()의 add()활용하여 모델 설계
# 딥러닝 구조를 결정합니다(모델을 설정하는 부분)

model = Sequential() #keras에서 제공되는 기능 

#1번째 층 : 입력 x는 17개, 출력은 30개, 활성화함수 relu 
model.add(Dense(30, input_dim=17, activation='relu'))

#2번째 층 : 입력 x는 30개, 출력은 1개, 활성화함수 sigmoid
model.add(Dense(1, activation='sigmoid'))


#  딥러닝 모델 컴파일 및 실행하기
# keras의 Sequential()에서 제공하는 compile( ) 기능으로 loss, optimizer, metrics 설정
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#옵션 중에서 loss='mean_squared_error'는 선형회귀에서 사용
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#옵션 중에서 loss='binary_crossentropy'는 시그모이드를 이용한 2진 분류에서 사용

# keras의 Sequential()에서 제공하는 fit( ) 기능으로 x, y, 학습 반복횟수, 배치 사이즈 설정
model.fit(X_train, Y_train, epochs=500, batch_size=10) # 학습 데이터셋 329 개로 학습
#batch_size=1로 하면, 329개의 데이터를 1개씩 처리하므로 실행 결과에서 329/329 출력
#model.fit(X_train, Y_train, epochs=500, batch_size=1) # 학습 데이터셋 329 개로 학습
#batch_size=10 으로 하면, 329개의 데이터를 10개씩 모아서 처리하므로 실행 결과에서 33/33으로 출력 

#  테스트 데이터를 기반으로 딥러닝 평가하기
# keras의 Sequential()에서 제공하는 model.evaluate( ) 기능 사용
# 리턴 값 : [ loss 오차 , acc 정확도]
print(f'테스트 데이터 개수 : {len(X_test)}개')
ev = model.evaluate(X_test, Y_test, batch_size=1) #디폴트 출력 내용 loss, accuracy 있음 
print(f'[ loss 오차 , accuracy 정확도] = {ev}') 


# 새로운 데이터를 대상으로 딥러닝 모델을 사용하여 예측하기
# keras의 Sequential()에서 제공하는 predict() 기능 사용

# 새로운 데이터를 위해서 임의로 생성
Lee = np.array([[293,1,3.8,2.8,0,0,0,0,0,0,12,0,0,0,1,0,62]])
print(model.predict(Lee)) 

test_lee  = model.predict(Lee)*100 
print("Kim 폐암 수술 후 생존율 예측 : %.2f" %test_lee, "%")

# 딥러닝 모델 저장하기
# keras의 Sequential()에서 제공하는 save() 기능 사용
model.save('DL_RESULT/ThoraricSurgery.h5') 