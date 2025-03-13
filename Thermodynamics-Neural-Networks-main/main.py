'''
Created on 26 Jan 2021

@author: filippomasi & ioannisstefanou
'''

# 필요한 라이브러리 불러오기
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리
import numpy as np  # 수학적 계산 및 배열 처리를 위한 라이브러리
import tensorflow as tf
from classThermodynamicsANNs.ThermodynamicsANNs import TANNs  # 열역학 기반 인공신경망 클래스
from classThermodynamicsANNs.PreProcessingOperations import preProcessing  # 데이터 전처리 클래스

# LaTeX 설정 (그래프 제목 및 축 라벨에 LaTeX 스타일 사용 여부)
plt.rc('text', usetex=False)  

# 출력 제어 변수 설정
silent = False  # silent=True로 설정하면 메시지 출력 생략
silent_summary = True  # silent_summary=False로 설정하면 신경망 요약 정보 출력

# **데이터 전처리**
# 입력 데이터와 출력 데이터를 전처리하는 함수 호출
data = preProcessing(
    '/Users/kangminhyeok/Desktop/학부연구생/박막물성측정 연구/Thermodynamics/Thermodynamics-Neural-Networks-main/reference_data/input_data',
    '/Users/kangminhyeok/Desktop/학부연구생/박막물성측정 연구/Thermodynamics/Thermodynamics-Neural-Networks-main/reference_data/output_data',
    silent
)

# 전처리된 데이터를 반환받음 (학습, 검증, 테스트용 데이터로 나뉨)
uN_T, uN_V, uN_Tt, oN_T, oN_V, oN_Tt, nrm_inp, nrm_out = data.Out()

# **열역학 기반 인공신경망(TANNs) 초기화**
ThermoANN = TANNs(nrm_inp, nrm_out, silent)  # 신경망 객체 생성
inputs = (None, 4)  # 입력 데이터 형상 정의 (샘플 개수는 None, 특성은 4개)
ThermoANN.build(inputs)  # 신경망 빌드(구조 생성)
if not silent_summary: 
    print(ThermoANN.summary())  # 신경망 구조 요약 출력

# **신경망 학습 및 평가**
if not silent: 
    print("\n... Training")

# 학습 파라미터 설정: 학습률(learningRate), 에포크 수(nEpochs), 배치 크기(bSize)
learningRate = 1e-4  
nEpochs = 100  
bSize = 10  

# 신경망 학습 수행 및 학습 기록 저장
historyTraining = ThermoANN.setTraining(ThermoANN, uN_T, oN_T, uN_V, oN_V, learningRate, nEpochs, bSize)

# 학습 데이터셋에 대한 평가
ThermoANN.evaluate(uN_T, oN_T)  

# 검증 데이터셋에 대한 평가
ThermoANN.evaluate(uN_V, oN_V)  

# 테스트 데이터셋에 대한 평가
ThermoANN.evaluate(uN_Tt, oN_Tt)

# **히스토그램 그리기**
if not silent:
    print("\n... Plotting histograms")

hist_params = {"bins": 50, "alpha": 0.5, "density": False}  # 히스토그램 파라미터 설정

# fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # 그래프 레이아웃 설정 (3행 x 2열)

# 데이터 구조 확인 (데이터 크기와 모양 확인)
print("uN_T shape:", uN_T.shape)  
print("uN_V shape:", uN_V.shape)

# 리스트 형태의 출력 데이터(oN_T와 oN_V)의 구조 확인
print("oN_T type:", type(oN_T), " | length:", len(oN_T))
for i, array in enumerate(oN_T):
    print(f"oN_T[{i}] shape:", array.shape)

print("oN_V type:", type(oN_V), " | length:", len(oN_V))
for i, array in enumerate(oN_V):
    print(f"oN_V[{i}] shape:", array.shape)

# **데이터 매핑(변수 이름 변경)**  
epsilon1_train = uN_T[:, 0]  # 변형률 데이터 (첫 번째 열)
sigma1_train = oN_T[0]  # 응력 데이터 (첫 번째 리스트 요소)
zeta1_train = oN_T[1]  # 내부 변수 데이터 (두 번째 리스트 요소)
F_train = oN_T[2]  # 에너지 데이터 (세 번째 리스트 요소)
D_train = oN_T[3]  # 소산율 데이터 (네 번째 리스트 요소)

# 추가적인 계산 수행 (데이터 간 연산)
if F_train.shape == zeta1_train.shape:
    epsilon_p_minus_zeta_p = F_train - zeta1_train  
else:
    print("Warning: Shape mismatch in epsilon_p_minus_zeta_p calculation")
    epsilon_p_minus_zeta_p = np.zeros_like(F_train)  

zeta1_zeta2_zeta3 = np.power(zeta1_train, 3)  # 내부 변수의 세제곱 계산

# # **그래프 생성**
# fig, axes = plt.subplots(3, 2, figsize=(12, 12))  

# # (a) Random Loading Path: 시간에 따른 변형률 그래프
# increments = np.arange(len(epsilon1_train))  
# axes[0, 0].plot(increments, epsilon1_train, color='black')
# axes[0, 0].set_title("(a) Random Loading Path")
# axes[0, 0].set_xlabel("Increments (-)")
# axes[0, 0].set_ylabel(r"$\varepsilon_1$ (-)")

# # (b) Stress and Internal Variable
# # 응력과 내부 변수에 대한 그래프를 생성
# axes[1, 0].plot(epsilon1_train, sigma1_train, 'b-', label="model")  # 모델의 응력 데이터 선 그래프
# axes[1, 0].scatter(epsilon1_train, sigma1_train, color='blue', marker='x', label="TANN")  # TANN의 응력 데이터 산점도
# axes[1, 0].set_title("(b) Stress")  # 그래프 제목 설정
# axes[1, 0].set_xlabel(r"$\varepsilon_1$ (-)")  # x축 라벨: 변형률
# axes[1, 0].set_ylabel(r"$\sigma_1$ (MPa)")  # y축 라벨: 응력
# axes[1, 0].legend()  # 범례 추가

# axes[1, 1].plot(epsilon1_train, zeta1_train, 'b-', label="model")  # 모델의 내부 변수 데이터 선 그래프
# axes[1, 1].scatter(epsilon1_train, zeta1_train, color='blue', marker='x', label="TANN")  # TANN의 내부 변수 데이터 산점도
# axes[1, 1].set_title("(b) Internal Variable")  # 그래프 제목 설정
# axes[1, 1].set_xlabel(r"$\varepsilon_1$ (-)")  # x축 라벨: 변형률
# axes[1, 1].set_ylabel(r"$\zeta_1$ (-)")  # y축 라벨: 내부 변수
# axes[1, 1].legend()  # 범례 추가

# # (c) Energy and Dissipation Rate
# # 에너지와 소산율에 대한 그래프를 생성
# axes[2, 0].plot(epsilon_p_minus_zeta_p, F_train, 'gray', label="model")  # 모델의 에너지 데이터 선 그래프
# axes[2, 0].scatter(epsilon_p_minus_zeta_p, F_train, color='blue', marker='x', label="TANN")  # TANN의 에너지 데이터 산점도
# axes[2, 0].set_title("(c) Energy")  # 그래프 제목 설정
# axes[2, 0].set_xlabel(r"$\varepsilon_p - \zeta_p$ (-)")  # x축 라벨: 변형률과 내부 변수의 차이
# axes[2, 0].set_ylabel(r"$F$ (N-mm)")  # y축 라벨: 에너지
# axes[2, 0].legend()  # 범례 추가

# axes[2, 1].plot(zeta1_zeta2_zeta3, D_train, 'gray', label="model")  # 모델의 소산율 데이터 선 그래프
# axes[2, 1].scatter(zeta1_zeta2_zeta3, D_train, color='blue', marker='x', label="TANN")  # TANN의 소산율 데이터 산점도
# axes[2, 1].set_title("(c) Dissipation Rate")  # 그래프 제목 설정
# axes[2, 1].set_xlabel(r"$\zeta_1 \zeta_2 \zeta_3$ (-)")  # x축 라벨: 내부 변수의 세제곱 값
# axes[2, 1].set_ylabel(r"$D$ (N-mm/s)")  # y축 라벨: 소산율
# axes[2, 1].legend()  # 범례 추가

# plt.tight_layout()  # 그래프 간 간격 조정
# plt.show()  # 그래프 출력

# # 📌 가능한 데이터만 선택 (4개 feature만 존재하므로 수정)
# # 학습 및 검증 데이터를 각 특성별로 분리하여 저장 (p와 q는 각각 평균 응력과 편차 응력을 나타냄)
# p_train, p_t_dt_train = uN_T[:, 0], uN_T[:, 1]   # 학습 데이터에서 p와 p_t+Δt 추출
# p_val, p_t_dt_val = uN_V[:, 0], uN_V[:, 1]       # 검증 데이터에서 p와 p_t+Δt 추출

# q_train, q_t_dt_train = uN_T[:, 2], uN_T[:, 3]   # 학습 데이터에서 q와 q_t+Δt 추출
# q_val, q_t_dt_val = uN_V[:, 2], uN_V[:, 3]       # 검증 데이터에서 q와 q_t+Δt 추출

# # 📌 리스트에서 데이터 가져오기 (현재 사용 가능한 oN_T 요소 활용)
# # 에너지(F)와 소산율(D)을 학습 및 검증 데이터로 분리하여 저장
# F_t_dt_train = oN_T[0]   # 학습 데이터에서 에너지(F)
# F_t_dt_val = oN_V[0]     # 검증 데이터에서 에너지(F)

# D_t_dt_train = oN_T[1]   # 학습 데이터에서 소산율(D)
# D_t_dt_val = oN_V[1]     # 검증 데이터에서 소산율(D)

# # 📌 그래프 그리기
# # 2x2 레이아웃으로 히스토그램을 그리기 위한 Figure와 Axes 생성
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2행 x 2열의 그래프 레이아웃, 크기는 가로 12인치, 세로 8인치

# # **1️⃣ Mean Stress (p)의 히스토그램**
# # 학습 및 검증 데이터에 대한 평균 응력(p) 분포를 시각화
# axes[0, 0].hist(p_train, bins=50, alpha=0.5, label="p_train", color="blue")  # 학습 데이터 p의 분포 (파란색)
# axes[0, 0].hist(p_t_dt_train, bins=50, alpha=0.5, label="p_t+Δt_train", color="lightblue")  # 학습 데이터 p_t+Δt의 분포 (연한 파란색)
# axes[0, 0].hist(p_val, bins=50, alpha=0.5, label="p_val", color="red")  # 검증 데이터 p의 분포 (빨간색)
# axes[0, 0].hist(p_t_dt_val, bins=50, alpha=0.5, label="p_t+Δt_val", color="pink")  # 검증 데이터 p_t+Δt의 분포 (분홍색)
# axes[0, 0].set_title("Mean Stress (p)")  # 그래프 제목 설정
# axes[0, 0].set_xlabel("p (MPa)")  # x축 라벨: 평균 응력 (단위: MPa)
# axes[0, 0].set_ylabel("N samples")  # y축 라벨: 샘플 개수
# axes[0, 0].legend()  # 범례 추가

# # 2️⃣ Deviatoric Stress (q)
# # 학습 및 검증 데이터에 대한 편차 응력(q)의 분포를 시각화
# axes[0, 1].hist(q_train, bins=50, alpha=0.5, label="q_train", color="blue")  # 학습 데이터 q의 분포 (파란색)
# axes[0, 1].hist(q_t_dt_train, bins=50, alpha=0.5, label="q_t+Δt_train", color="lightblue")  # 학습 데이터 q_t+Δt의 분포 (연한 파란색)
# axes[0, 1].hist(q_val, bins=50, alpha=0.5, label="q_val", color="red")  # 검증 데이터 q의 분포 (빨간색)
# axes[0, 1].hist(q_t_dt_val, bins=50, alpha=0.5, label="q_t+Δt_val", color="pink")  # 검증 데이터 q_t+Δt의 분포 (분홍색)
# axes[0, 1].set_title("Deviatoric Stress (q)")  # 그래프 제목 설정
# axes[0, 1].set_xlabel("q (MPa)")  # x축 라벨: 편차 응력 (단위: MPa)
# axes[0, 1].set_ylabel("N samples")  # y축 라벨: 샘플 개수
# axes[0, 1].legend()  # 범례 추가

# # 3️⃣ Energy (F)
# # 학습 및 검증 데이터에 대한 에너지(F)의 분포를 시각화
# axes[1, 0].hist(F_t_dt_train, bins=50, alpha=0.5, label="F_train", color="blue")  # 학습 데이터 F의 분포 (파란색)
# axes[1, 0].hist(F_t_dt_val, bins=50, alpha=0.5, label="F_val", color="red")  # 검증 데이터 F의 분포 (빨간색)
# axes[1, 0].set_title("Energy (F)")  # 그래프 제목 설정
# axes[1, 0].set_xlabel("F (N-mm)")  # x축 라벨: 에너지 (단위: N-mm)
# axes[1, 0].set_ylabel("N samples")  # y축 라벨: 샘플 개수
# axes[1, 0].legend()  # 범례 추가

#  # 4️⃣ Dissipation Rate (D)
# # 학습 및 검증 데이터에 대한 소산율(D)의 분포를 시각화
# axes[1, 1].hist(D_t_dt_train, bins=50, alpha=0.5, label="D_train", color="blue")  # 학습 데이터 D의 분포 (파란색)
# axes[1, 1].hist(D_t_dt_val, bins=50, alpha=0.5, label="D_val", color="red")  # 검증 데이터 D의 분포 (빨간색)
# axes[1, 1].set_title("Dissipation Rate (D)")  # 그래프 제목 설정
# axes[1, 1].set_xlabel("D (N-mm/s)")  # x축 라벨: 소산율 (단위: N-mm/s)
# axes[1, 1].set_ylabel("N samples")  # y축 라벨: 샘플 개수
# axes[1, 1].legend()  # 범례 추가

# # 그래프 레이아웃 조정 및 출력
# plt.tight_layout()  # 그래프 간 간격을 자동으로 조정하여 보기 좋게 배치
# plt.show()  # 그래프 출력

# # **신경망 가중치 저장**
# if silent == False: 
#     print("\n... Saving weights")  # 가중치 저장 메시지 출력

# # 학습된 신경망의 가중치를 지정된 경로에 저장
# ThermoANN.save_weights('./output_data/ThermoTANN_weights', save_format='tf')  
# # 저장 형식은 TensorFlow 포맷(tf) 사용

# print("\n... Completed!")  # 완료 메시지 출력

# 입력 데이터를 TensorFlow Tensor로 변환
tf_uN_T = tf.constant(uN_T, dtype=tf.float32)

# 모델 예측값 얻기 (GradientTape로 수정)
with tf.GradientTape() as tape:
    tape.watch(tf_uN_T)
    predicted_outputs = ThermoANN(tf_uN_T)

predicted_outputs_unnorm = [ThermoANN.tf_u_stnd_nrml(pred, α, β)
                            for pred, α, β in zip(predicted_outputs, nrm_out[::2], nrm_out[1::2])]

# 데이터 매핑 (정확한 배열 사용)
epsilon1_train = uN_T[:, 0]
sigma1_true = oN_T[0]
sigma1_pred = predicted_outputs_unnorm[1].numpy()

zeta1_true = oN_T[1]
zeta1_pred = predicted_outputs_unnorm[0].numpy()

F_true = oN_T[2]
F_pred = predicted_outputs_unnorm[2].numpy()

D_true = oN_T[3]
D_pred = predicted_outputs_unnorm[3].numpy()

# 그래프 그리기
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# (a) Random Loading Path
increments = np.arange(len(epsilon1_train))  
axes[0, 0].plot(increments, epsilon1_train, color='black')
axes[0, 0].set_title("(a) Random Loading Path")
axes[0, 0].set_xlabel("Increments (-)")
axes[0, 0].set_ylabel(r"$\varepsilon_1$ (-)")

# Stress
axes[1, 0].plot(epsilon1_train, sigma1_true, 'gray', label="model")
axes[1, 0].scatter(epsilon1_train, sigma1_pred, c='blue', marker='x', label="TANN")
axes[1, 0].set_title("(b) Stress")
axes[1, 0].set_xlabel(r"$\varepsilon_1$ (-)")
axes[1, 0].set_ylabel(r"$\sigma_1$ (MPa)")
axes[1, 0].legend()

# Internal Variable
axes[1, 1].plot(epsilon1_train, zeta1_true, 'gray', label="model")
axes[1, 1].scatter(epsilon1_train, zeta1_pred, color='blue', marker='x', label="TANN")
axes[1, 1].set_title("(b) Internal Variable")
axes[1, 1].set_xlabel(r"$\varepsilon_1$ (-)")
axes[1, 1].set_ylabel(r"$\zeta_1$ (-)")
axes[1, 1].legend()

# Energy
axes[2, 0].plot(epsilon1_train, F_true, 'gray', label="model")
axes[2, 0].scatter(epsilon1_train, F_pred, color='blue', marker='x', label="TANN")
axes[2, 0].set_title("(c) Energy")
axes[2, 0].set_xlabel(r"$\varepsilon_1$ (-)")
axes[2, 0].set_ylabel(r"$F$ (N-mm)")
axes[2, 0].legend()

# Dissipation Rate
axes[2, 1].plot(epsilon1_train, D_true, 'gray', label="model")
axes[2, 1].scatter(epsilon1_train, D_pred, color='blue', marker='x', label="TANN")
axes[2, 1].set_title("(c) Dissipation Rate")
axes[2, 1].set_xlabel(r"$\varepsilon_1$ (-)")
axes[2, 1].set_ylabel(r"$D$ (N-mm/s)")
axes[2, 1].legend()

plt.tight_layout()
plt.show()

# 가중치 저장
if not silent: 
    print("\n... Saving weights")

ThermoANN.save_weights('./output_data/ThermoTANN_weights', save_format='tf')

print("\n... Completed!")