'''
Created on 26 Jan 2021

@author: filippomasi & ioannisstefanou
'''

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

#사용자 정의 활성화 함수 정의. x^2 적용 후 ELU 함수 사용
def act(x):
    return tf.keras.backend.elu(tf.keras.backend.pow(x, 2), alpha=1.0) 
tf.keras.utils.get_custom_objects().update({'act': tf.keras.layers.Activation(act)})

#TANNs 클래스 정의
class TANNs(tf.keras.Model):
    def __init__(self,nrm_inp,nrm_out,training_silent=True):
        self.training_silent=training_silent
        super(TANNs, self).__init__()

        #신경망 레이어 정의
        self.f0=tf.keras.layers.Dense(6, activation=tf.nn.leaky_relu) # 첫 번째 레이어, Leaky ReLU 활성화 사용
        self.f1=tf.keras.layers.Dense(1, use_bias=False) # 두 번째 레이어, 활성화 함수 없음
        self.f2=tf.keras.layers.Dense(9, activation='act') # 사용자 정의 활성화 함수 act 사용
        self.f3=tf.keras.layers.Dense(1, use_bias=False) # 마지막 레이어, 활성화 함수 없음

        #입력 및 출력의 정규화 및 역정규화를 위한 매개변수 저장
        self.ε_t_α,self.ε_t_β,self.ε_tdt_α,self.ε_tdt_β,self.Dε_t_α,self.Dε_t_β,self.σ_t_α,self.σ_t_β,self.α_t_α,self.α_t_β=nrm_inp
        self.α_tdt_α,self.α_tdt_β,self.Dα_tdt_α,self.Dα_tdt_β,self.F_tdt_α,self.F_tdt_β,self.σ_tdt_α,self.σ_tdt_β,self.Dσ_tdt_α,self.Dσ_tdt_β,self.D_tdt_α,self.D_tdt_β=nrm_out

    # 정규화 함수   
    def tf_out_stnd_nrml(self,u,α,β):
        ''' Standardize/normalize '''
        return tf.divide(tf.add(u,-β),α)
    
    # 역정규화 함수
    def tf_u_stnd_nrml(self,output,α,β):
        ''' Un-standardize/un-normalize '''
        return tf.add(tf.multiply(output,α),β)
    
    # 손실 함수의 기울기 계산
    def tf_get_gradients(self,f_tdt,u_ε_tdt,u_α_tdt,u_Dα_tdt,u_σ_t):
        ''' Compute normalized tf.gradients from normalized f @ t+dt and un-normalized ε and α'''
        # σ @ t+dt (non-normalized)
        # 다음 시점(t+dt)의 에너지(f_tdt)를 변형률(u_ε_tdt)로 미분하여, 다음 시점의 응력(σ_tdt)을 계산.
        f_σ_tdt=tf.gradients(f_tdt,u_ε_tdt)[0]  

        # Dσ @ t+dt (non-normalized)
        # 다음 시점의 응력(f_σ_tdt)에서 현재 시점의 응력(u_σ_t)을 뺀 값으로, 시점 간 응력의 변화를 나타낸다.
        f_Dσ_tdt=tf.math.subtract(f_σ_tdt,u_σ_t)  

        # σ @ t+dt (normalized)
        # 앞에서 계산된 응력 변화량을 다시 정규화하여 신경망에서 사용 가능한 형태로 변환합니다.
        nf_Dσ_tdt=self.tf_out_stnd_nrml(f_Dσ_tdt,self.Dσ_tdt_α,self.Dσ_tdt_β) 

        # 에너지(f_tdt)를 다음 시점 내부 상태변수(u_α_tdt)에 대해 미분하여 내부 상태변수 변화율을 계산합니다. 기호(-)는 역방향의 변화 방향을 나타낸다.
        f_chi_tdt=-tf.gradients(f_tdt,u_α_tdt)[0]

        # 위에서 구한 편미분 값(f_chi_tdt)과 내부 상태 변수의 변화량(u_Dα_tdt)을 곱하여 소산율을 계산
        f_d_tdt=tf.math.multiply(f_chi_tdt,u_Dα_tdt)

        # 계산된 소산율(f_d_tdt)을 정규화하여 신경망에서 예측 결과로 출력할 수 있도록 준비
        nf_d_tdt=self.tf_out_stnd_nrml(f_d_tdt,self.D_tdt_α,self.D_tdt_β)
        
        # 다음 시점의 정규화된 응력 변화량(nf_Dσ_tdt)과 소산율(nf_d_tdt)을 반환합니다.
        return nf_Dσ_tdt,nf_d_tdt
    

    def call(self,un):   
        # Slice ε # t+dt from inputs of ANN no.1
        un_ε_t_f = tf.slice(un,[0,0],[-1,1]) # 현재 시점t의 변형률
        un_Dε_t_f = tf.slice(un,[0,1],[-1,1]) # 변형률 변화량
        un_σ_t_f = tf.slice(un,[0,2],[-1,1]) # 현재 시점의 응력
        un_α_t_f = tf.slice(un,[0,3],[-1,1]) # 현재 시점의 물질 변수
        # Un-normalized ε_tdt, α_t, and α_tdt
        u_ε_t = self.tf_u_stnd_nrml(un_ε_t_f, self.ε_t_α, self.ε_t_β) # 복원된 현재 시점의 변형률
        u_Dε_t = self.tf_u_stnd_nrml(un_Dε_t_f, self.Dε_t_α, self.Dε_t_β)
        u_ε_tdt = tf.math.add(u_ε_t,u_Dε_t) # 다음 시점 t+dt 으 변형률 계산
        u_α_t = self.tf_u_stnd_nrml(un_α_t_f, self.α_t_α, self.α_t_β)
        u_σ_t = self.tf_u_stnd_nrml(un_σ_t_f, self.σ_t_α, self.σ_t_β)
        # Re-normalized ε_tdt, α_t, and α_tdt 다시 정규화
        nu_ε_tdt = self.tf_out_stnd_nrml(u_ε_tdt, self.ε_tdt_α, self.ε_tdt_β) # 정규화된 다음 시점 변형률
        nu_Dε_t = self.tf_out_stnd_nrml(u_Dε_t, self.Dε_t_α, self.Dε_t_β) # 정규화된 변형률 변화량
        nu_α_t = self.tf_out_stnd_nrml(u_α_t, self.α_t_α, self.α_t_β) # 정규화된 물질 변수
        nu_σ_t = self.tf_out_stnd_nrml(u_σ_t, self.σ_t_α, self.σ_t_β) #정규화된 응력
        un_concat_sub = tf.concat([nu_ε_tdt, nu_Dε_t, nu_α_t, nu_σ_t], 1, name = 'x1')
        # ANN no.1 첫번째 신경망 입력 준비
        nf0_sub = self.f0(un_concat_sub)
        nf_Dα_tdt = self.f1(nf0_sub) #α_tdt (normalized)
        u_Dα_tdt = self.tf_u_stnd_nrml(nf_Dα_tdt, self.Dα_tdt_α, self.Dα_tdt_β)
        u_α_tdt = tf.math.add(u_Dα_tdt, u_α_t)
        nu_α_tdt = self.tf_out_stnd_nrml(u_α_tdt, self.α_tdt_α, self.α_tdt_β)
        # Concatenate inputs (re-)normalized
        nu_concat = tf.concat([nu_ε_tdt, nu_α_tdt], 1, name = 'a2')
        # ANN no.2
        nf1 = self.f2(nu_concat)
        nf_tdt = self.f3(nf1) # energy @ t+dt (normalized)
        f_tdt = self.tf_u_stnd_nrml(nf_tdt, self.F_tdt_α, self.F_tdt_β) # energy @ t+dt (non-normalized)
        nf_Dσ_tdt, nf_d_tdt = self.tf_get_gradients(f_tdt, u_ε_tdt, u_α_tdt, u_Dα_tdt, u_σ_t)
        return nf_Dα_tdt, nf_tdt, nf_Dσ_tdt, nf_d_tdt
    
    def setTraining(self, TANN, input_training, output_training, input_validation, output_validation, learningRate, nEpochs, bSize):
        # training_silent이 False이면 verbose를 2로 설정 (훈련 중 자세한 로그 출력)
        # 그렇지 않으면 verbose를 0으로 설정 (훈련 중 로그 출력 없음)
        if self.training_silent == False: 
            silent_verbose = 2
        else: 
            silent_verbose = 0

        # Nadam 최적화 알고리즘을 사용하여 옵티마이저 생성
        optimizer = tf.keras.optimizers.Nadam(learningRate)

        # 손실 함수의 가중치 설정 (각 손실 항목에 대한 중요도를 조정)
        tol_e = 1.e+1  # 예를 들어, 에너지 손실 항목의 가중치
        tol_s = 1.e-1  # 예를 들어, 힘 손실 항목의 가중치
        wα = tol_e     # 첫 번째 손실 가중치
        wF = tol_s     # 두 번째 손실 가중치
        wσ = 2 * tol_s # 세 번째 손실 가중치
        wD = tol_e * tol_s # 네 번째 손실 가중치

        # 모델 컴파일: 여러 개의 손실 함수와 각각의 가중치를 설정
        TANN.compile(
            optimizer=optimizer,
            loss=['mae', 'mae', 'mae', 'mae'],  # 손실 함수로 MAE(Mean Absolute Error) 사용
            loss_weights=[wα, wF, wσ, wD]       # 각 손실 함수에 대한 가중치 적용
        )

        # EarlyStopping 콜백 설정: 검증 손실(val_loss)을 모니터링하여 학습 조기 종료
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',          # 검증 손실을 기준으로 모니터링
            min_delta=1.e-6,             # 개선 최소값 (변화가 이 값보다 작으면 종료)
            patience=2000,               # 개선되지 않는 상태가 2000 epochs 지속되면 종료
            verbose=0,                   # 로그 출력 여부 (0: 출력 안 함)
            mode='auto',                 # 자동으로 최적화 방향 결정 ('min' 또는 'max')
            baseline=None,               # 기준값 (None이면 무시)
            restore_best_weights=True    # 가장 좋은 가중치를 복원
        )

        # 모델 학습 시작 (fit 메서드 호출)
        history = TANN.fit(
            input_training,              # 훈련 입력 데이터
            output_training,             # 훈련 출력 데이터
            epochs=nEpochs,              # 총 학습 반복 횟수 (epochs)
            batch_size=bSize,            # 배치 크기
            verbose=silent_verbose,      # 로그 출력 수준 (위에서 설정한 값 사용)
            validation_data=(input_validation, output_validation),  # 검증 데이터 제공
            callbacks=[earlystop_callback]  # EarlyStopping 콜백 추가
        )

        # training_silent이 False인 경우 학습 완료 메시지 출력
        if self.training_silent == False: 
            print("\n Training completed in", nEpochs, " epochs")

        # 학습 기록(history) 반환 (훈련 및 검증 손실 기록 포함)
        return history
