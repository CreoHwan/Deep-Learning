import pandas as pd

df = pd.read_csv('./dataset/pima-indians-diabetes.csv', names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])


# < 데이터를 가공할 때의 주의점 >
# 우리가 무엇을 위해 작업을 하는지 그 최종 목적을 항상 생각해야 한다.
# 이 프로젝트의 목적 : 당뇨병 발병을 예측하
# 필요한 정보: 당뇨병 발병과 어떤 관계가 있는지

# 3-1임신횟수를 기반으로 당뇨병 발병 확률 분석하기
pregnant_analysis = df[['pregnant','class']].groupby(['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True)

# groupby() 함수를 사용해 pregnant 정보를 기준으로 하는 새 그룹 생성
# as_index=False는 pregnant 정보 좌측에 0, 1 인덱스(index) 생성
# sort_values() 함수를 써서 pregnant 컬럼을 ascending=True 으로 오름차순(ascending) 정렬하여 => 임신 횟수 당 당뇨병 발병 확률을 출력함


# 3-2혈당을 기반으로 당뇨병 발병 확률 분석하기

plasma_analysis = df[['plasma','class']].groupby(['plasma'], as_index=False).mean().sort_values(by='plasma', ascending=True)


# 3-3데이터의 상관관계를 그래프로 시각화하여 분석하기
import matplotlib.pyplot as plt
import seaborn as sns

colormap = plt.cm.gist_heat  
#matplotlib.pyplot에서 제공하는 색상 구성을 사용할 수 있도록 셋팅
plt.figure(figsize=(10,6))   
#그래프의 크기 셋팅


# heatmap 그래프 (1) : cmap=colormap 이용
sns.heatmap(df.corr(), cmap=colormap, linewidths=0.1,vmax=0.5,  linecolor='black', annot=True)
plt.show()
# seaborn에서 제공하는 heatmap : 두 항목씩 짝을 지은 뒤 각각 어떤 패턴으로 변화하는지를 관찰하는 함수
# 두 항목이 전혀 다른 패턴으로 변화하고 있으면 0을, 서로 비슷한 패턴으로 변할수록 1에 가까운 값을 출력함
# vmax의 값을 0.5로 지정할 경우, 0.5이상부터 흰색으로 표시됨 
# 그래프 위에 값이 출력되게 하려면 annot=True

# 그래프를 통해 plasma (공복 혈당 농도)는 0.47의 수치로 class 항목과 가장 상관관계가 높다는 것을 알 수 있음
# 이제 plasma와 class 항목만 따로 떼어 두 항목 간의 관계를 그래프로 다시 한번 확인해보자.


# 4 공복 혈당 농도를 기반으로 당뇨병 발병 확률 분석하기
# FacetGrid 그래프: 양한 범주형 값을 가지는 데이터를 시각화하는데 좋은 방법
grid = sns.FacetGrid(df, col='class') 
grid.map(plt.hist, 'plasma', bins=10) 
plt.show()
# # plasma -> x축, bins=10 -> 10개의 구간으로 분리하여 히스토그램 완성

# 당뇨병 환자의 경우 : class가 1에 해당하며 plasma 수치가 150이상인 경우가 많다는 것을 확인

# 5. 딥러닝
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

#  random( ) 함수
#  '랜덤 테이블’ 중 하나를 불러내 그 표의 순서대로 숫자를 보여 주는 것": seed 값이 무엇이냐 = 랜덤 테이블 => seed 값이 같으면 똑같은 랜덤 값을 출력함

seed=2
np.random.seed(seed) 
tf.random.set_seed(seed) 
# 넘파이 라이브러리를 사용하면서 텐서플로 기반으로 딥러닝을 구현할 때는 일정한 결과값을 얻기 위해서, 넘파이 seed 값과 텐서플로 seed 값을 모두 설정해야 함
# 최종 딥러닝 결과는 다양한 seed를 여러 번 실행하여 평균을 구하는 것이 가장 적절함

# 데이터
dataset = np.loadtxt("./dataset/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

# 데이터 학습용과 테스트용 생성
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

# 딥러닝 구조를 결정(모델 설정) ??
model = Sequential() #keras에서 제공되는 기능
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 모델 컴파일 및 실행하기
model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=200, batch_size=10)

# 테스트 데이터를 기반으로 딥러닝 평가
# keras의 Sequential()의 model.evaluate( ) 기능 사용 => 리턴 값 : [ loss 오차 , acc 정확도]
print("\n Test Accuracy : %.4f" % (model.evaluate(X_test, Y_test, batch_size=10)[1])) 

# 새로운 데이터를 대상으로 딥러닝 모델을 사용하여 예측
# keras의 Sequential()에서 제공하는 predict() 기능 사용
kim = np.array([[3, 78, 50, 32, 88, 31, 0.248, 26]])
park = np.array([[10, 115, 0, 0, 0, 35.3, 0.134, 29]])
choi = np.array([[2, 197, 70, 45, 543, 30.5, 0.158, 53]])

test_kim  = model.predict(kim)*100
test_park = model.predict(park)*100
test_choi = model.predict(choi)*100

print("Kim  당뇨병 가능성 예측 : %.2f" %test_kim, "%")
print("Park 당뇨병 가능성 예측 : %.2f" %test_park, "%")
print("Choi 당뇨병 가능성 예측 : %.2f" %test_choi, "%")

# 딥러닝 모델 저장하기
model.save('/.DL_RESULT/Diabetes.h5') 