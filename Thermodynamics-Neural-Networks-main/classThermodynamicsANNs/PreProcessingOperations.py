'''
Created on 26 Jan 2021

@author: filippomasi & ioannisstefanou
'''         
import numpy as np
import pickle

class preProcessing():
    def __init__(self, file_inp="", file_out="", silent=True):
        self.silent = silent  # silent=True이면 출력 메시지를 표시하지 않음
        
        if file_inp == "" or file_out == "":  # 입력 파일과 출력 파일이 제공되지 않은 경우
            if self.silent == False:
                print("No data provided for training")  # 학습 데이터를 제공받지 못했다는 메시지 출력
        else:
            if self.silent == False:
                print("... Loading data")  # 데이터 로드 중임을 알리는 메시지 출력
            
            # pickle 파일을 읽어서 데이터를 로드
            with open(file_inp, 'rb') as f_obj:
                inputs = pickle.load(f_obj)  # 입력 데이터 로드
            with open(file_out, 'rb') as f_obj:
                outputs = pickle.load(f_obj)  # 출력 데이터 로드
            
            # 입력 데이터 및 출력 데이터 변수를 분리
            e_t, e_tdt, De_t, ap_t, sg_t = inputs  # 입력 데이터 (시간 t에서의 값들)
            ap_tdt, Dap_t, F_tdt, sg_tdt, Dsg_t, D_tdt = outputs  # 출력 데이터 (시간 t+dt에서의 값들)
            
            # 데이터 샘플 개수 확인
            self.n_samples = e_tdt.shape[0]  # 총 샘플 개수
            
            # 학습, 검증, 테스트 데이터 비율 설정
            self.train_percentage = 0.5  # 학습 데이터 비율 50%
            self.n_train_samples = int(round(self.n_samples * self.train_percentage))  # 학습 데이터 개수
            self.n_val_samples = int(round(self.n_train_samples * (1. - self.train_percentage)))  # 검증 데이터 개수
            self.n_test_samples = self.n_val_samples  # 테스트 데이터 개수 (검증 데이터 개수와 동일하게 설정)
            
            if self.silent == False:
                print("... Slice data into training, validation, and test data-sets")  # 데이터 분할 메시지 출력
            
            # 데이터를 훈련, 검증, 테스트 세트로 분할하고 정규화 수행
            self.nrm_inp, self.nrm_out, inputs, outputs = self.slice_TVTt(
                e_t, e_tdt, De_t, ap_t, sg_t, ap_tdt, Dap_t, F_tdt, sg_tdt, Dsg_t, D_tdt
            )
            
            # 입력 및 출력 데이터를 학습, 검증, 테스트 세트로 구성
            self.uN_T, self.uN_V, self.uN_Tt, self.oN_T, self.oN_V, self.oN_Tt = self.preInandOut(inputs, outputs)


    def slice_TVTt(self,e_t,e_tdt,De_t,ap_t,sg_t,ap_tdt,Dap_t,F_tdt,sg_tdt,Dsg_t,D_tdt):
        '''Slice data into training, validation, and test data-sets '''
        ε_t_tv=e_t[:(self.n_train_samples + self.n_val_samples)] # ε @ t 변형률
        Dε_t_tv=De_t[:(self.n_train_samples + self.n_val_samples)] # Δε @ t 변형률 변화
        σ_t_tv=sg_t[:(self.n_train_samples + self.n_val_samples)] # σ @ t 응력
        α_t_tv=ap_t[:(self.n_train_samples + self.n_val_samples)] # α @ t 소성 변수?
        # Inputs (test) = ε_t, Δε_t, σ_t, α_t
        ε_t_test=e_t[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # ε @ t
        Dε_t_test=De_t[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # Δε @ t
        σ_t_test=sg_t[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # σ @ t
        α_t_test=ap_t[(self.n_train_samples+self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # α @ t
        # Outputs (training + validation = tv) = F_(t+dt), α_(t+dt), σ_(t+dt)
        F_tdt_tv=F_tdt[:(self.n_train_samples+self.n_val_samples)] # F_(t+dt) 힘
        Dα_tdt_tv=Dap_t[:(self.n_train_samples+self.n_val_samples)] # α_(t+dt) 응력 변화량
        Dσ_tdt_tv=Dsg_t[:(self.n_train_samples+self.n_val_samples)] # σ_(t+dt) 소성 변수 변화량
        α_tdt_tv=ap_tdt[:(self.n_train_samples+self.n_val_samples)] # α_(t+dt) 
        σ_tdt_tv=sg_tdt[:(self.n_train_samples+self.n_val_samples)] # σ_(t+dt)
        D_tdt_tv=D_tdt[:(self.n_train_samples+self.n_val_samples)] # D_(t+dt) 손상 변수
        # Outputs (test) = F_(t+dt), α_(t+dt), σ_(t+dt)
        F_tdt_test=F_tdt[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # F_(t+dt)
        Dα_tdt_test=Dap_t[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # α_(t+dt)
        Dσ_tdt_test=Dsg_t[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # σ_(t+dt)
        α_tdt_test=ap_tdt[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # α_(t+dt)
        σ_tdt_test=sg_tdt[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # σ_(t+dt)
        D_tdt_test=D_tdt[(self.n_train_samples + self.n_val_samples):(self.n_train_samples + self.n_val_samples + self.n_test_samples)] # F_(t+dt)
        
        if self.silent==False: print("... Normalizing data")
        norm_st = True
        ε_t_α, ε_t_β = self.get_α_β(ε_t_tv, norm_st) # 현재 시점(t) 의 정규화
        ε_tdt_α, ε_tdt_β = self.get_α_β(ε_t_tv+Dε_t_tv, norm_st) # 다음 시점(t+dt)의 변형률 정규화
        Dε_t_α, Dε_t_β = self.get_α_β(Dε_t_tv, norm_st) # 변형률 변화량(증가분) 정규화
        σ_t_α, σ_t_β = self.get_α_β(σ_t_tv, norm_st) # 현재 시점(t)의 응력 정규화
        α_t_α, α_t_β = self.get_α_β(α_t_tv, norm_st) # 현재 시점(t)의 물질 변수 정규화
        F_tdt_α, F_tdt_β = self.get_α_β(F_tdt_tv, norm_01=True) # 힘 정규화
        D_tdt_α, D_tdt_β = self.get_α_β(D_tdt_tv, norm_01=True) # 힘 변화량 정규화
        σ_tdt_α, σ_tdt_β = self.get_α_β(σ_tdt_tv, norm_st) # 다음 시점(t+dt)의 응력 정규화
        Dσ_tdt_α, Dσ_tdt_β = self.get_α_β(Dσ_tdt_tv, norm_st) # 응력 변화량(증가분) 정규화
        α_tdt_α, α_tdt_β = self.get_α_β(α_tdt_tv, norm_st) # 다음 시점(t+dt)의 물질 변수 정규화
        Dα_tdt_α, Dα_tdt_β = self.get_α_β(Dα_tdt_tv, norm_st) # 물질 변수 변화량 정규화
        
        n_ε_t_tv = self.out_stnd_nrml(ε_t_tv, ε_t_α, ε_t_β)
        n_Dε_t_tv = self.out_stnd_nrml(Dε_t_tv, Dε_t_α, Dε_t_β)
        n_σ_t_tv = self.out_stnd_nrml(σ_t_tv, σ_t_α, σ_t_β)
        n_α_t_tv = self.out_stnd_nrml(α_t_tv, α_t_α, α_t_β)
        n_ε_t_test = self.out_stnd_nrml(ε_t_test, ε_t_α, ε_t_β)
        n_Dε_t_test = self.out_stnd_nrml(Dε_t_test, Dε_t_α, Dε_t_β)
        n_σ_t_test = self.out_stnd_nrml(σ_t_test, σ_t_α, σ_t_β)
        n_α_t_test = self.out_stnd_nrml(α_t_test, α_t_α, α_t_β)
        n_F_tdt_tv = self.out_stnd_nrml(F_tdt_tv, F_tdt_α, F_tdt_β)
        n_D_tdt_tv = self.out_stnd_nrml(D_tdt_tv, D_tdt_α, D_tdt_β)
        n_Dσ_tdt_tv = self.out_stnd_nrml(Dσ_tdt_tv, Dσ_tdt_α, Dσ_tdt_β)
        n_Dα_tdt_tv = self.out_stnd_nrml(Dα_tdt_tv, Dα_tdt_α, Dα_tdt_β)
        n_F_tdt_test = self.out_stnd_nrml(F_tdt_test, F_tdt_α, F_tdt_β)
        n_D_tdt_test = self.out_stnd_nrml(D_tdt_test, D_tdt_α, D_tdt_β)
        n_Dσ_tdt_test = self.out_stnd_nrml(Dσ_tdt_test, Dσ_tdt_α, Dσ_tdt_β)
        n_Dα_tdt_test = self.out_stnd_nrml(Dα_tdt_test, Dα_tdt_α, Dα_tdt_β)
        
        nrm_inp=[ε_t_α, ε_t_β, ε_tdt_α, ε_tdt_β, Dε_t_α, Dε_t_β, σ_t_α, σ_t_β, α_t_α, α_t_β]
        nrm_out=[α_tdt_α, α_tdt_β, Dα_tdt_α, Dα_tdt_β, F_tdt_α, F_tdt_β, σ_tdt_α, σ_tdt_β, Dσ_tdt_α, Dσ_tdt_β, D_tdt_α, D_tdt_β]
        inputs=[n_ε_t_tv,n_Dε_t_tv,n_σ_t_tv,n_α_t_tv,n_ε_t_test,n_Dε_t_test,n_σ_t_test,n_α_t_test]
        outputs=[n_F_tdt_tv,n_D_tdt_tv,n_Dσ_tdt_tv,n_Dα_tdt_tv,n_F_tdt_test,n_D_tdt_test,n_Dσ_tdt_test,n_Dα_tdt_test]
        return nrm_inp,nrm_out,inputs,outputs
    
    def preInandOut(self,inputs,outputs):
        
        # 학습(training), 검증(validation), 테스트(test) 데이터를 조립하는 함수
        # 입력 데이터(inputs)와 출력 데이터(outputs)를 받아 각각 훈련, 검증, 테스트 세트로 분할함
        
        if self.silent==False: print("... Assembling data for training, validating, and testing")
        
        # 입력 데이터를 개별 변수로 분리
        n_ε_t_tv, n_Dε_t_tv, n_σ_t_tv, n_α_t_tv, n_ε_t_test, n_Dε_t_test, n_σ_t_test, n_α_t_test = inputs
        # 출력 데이터를 개별 변수로 분리
        n_F_tdt_tv, n_D_tdt_tv, n_Dσ_tdt_tv, n_Dα_tdt_tv, n_F_tdt_test, n_D_tdt_test, n_Dσ_tdt_test, n_Dα_tdt_test = outputs
        
        # 훈련 및 검증 입력 데이터를 하나의 배열로 결합 (입력 특징 4개: ε_t, Δε_t, σ_t, α_t)
        uN_TV = np.concatenate((np.expand_dims(n_ε_t_tv, axis=1),
                                np.expand_dims(n_Dε_t_tv, axis=1),
                                np.expand_dims(n_σ_t_tv, axis=1),
                                np.expand_dims(n_α_t_tv, axis=1)), axis=1)
        
        # 훈련 데이터 (Training Set) 분할
        uN_T = uN_TV[:self.n_train_samples,:]
        # 검증 데이터 (Validation Set) 분할
        uN_V = uN_TV[self.n_train_samples:(self.n_train_samples + self.n_val_samples),:]
        
        # 테스트 입력 데이터를 하나의 배열로 결합 (입력 특징 4개: ε_t, Δε_t, σ_t, α_t)
        uN_Tt = np.concatenate((np.expand_dims(n_ε_t_test, axis=1),
                                np.expand_dims(n_Dε_t_test, axis=1),
                                np.expand_dims(n_σ_t_test, axis=1),
                                np.expand_dims(n_α_t_test, axis=1)), axis=1)
        
        # 훈련 데이터 출력값 (Output Features) 분할
        oN_T = [n_Dα_tdt_tv[:self.n_train_samples],
                n_F_tdt_tv[:self.n_train_samples],
                n_Dσ_tdt_tv[:self.n_train_samples],
                n_D_tdt_tv[:self.n_train_samples]]
        
        # 검증 데이터 출력값 분할
        oN_V = [n_Dα_tdt_tv[self.n_train_samples:(self.n_train_samples + self.n_val_samples)],
                n_F_tdt_tv[self.n_train_samples:(self.n_train_samples + self.n_val_samples)],
                n_Dσ_tdt_tv[self.n_train_samples:(self.n_train_samples + self.n_val_samples)],
                n_D_tdt_tv[self.n_train_samples:(self.n_train_samples + self.n_val_samples)]]
        
        # 테스트 데이터 출력값 분할
        oN_Tt = [n_Dα_tdt_test, n_F_tdt_test, n_Dσ_tdt_test, n_D_tdt_test]
        
        return uN_T, uN_V, uN_Tt, oN_T, oN_V, oN_Tt
    
    def Out(self):
        '''
        훈련, 검증, 테스트 데이터 및 정규화(표준화) 정보를 반환하는 함수
        '''
        return self.uN_T, self.uN_V, self.uN_Tt, self.oN_T, self.oN_V, self.oN_Tt, self.nrm_inp, self.nrm_out
        
    def get_α_β(self,u,norm=True,norm_01=False,no=False):
        '''
        데이터 정규화(표준화)를 위한 α (스케일링 계수)와 β (평균 또는 최소값)를 계산하는 함수
        
        - norm=True: 표준 정규화 방식 적용
        - norm_01=True: 0~1 사이로 정규화 (min-max 정규화)
        - no=True: 정규화 적용하지 않음 (기본값으로 α=1, β=0 설정)
        '''
        if no == False:
            if norm == True:
                if norm_01 == False:
                    u_max = np.amax(u)  # 데이터 최댓값
                    u_min = np.amin(u)  # 데이터 최솟값
                    u_α = (u_max - u_min) / 2.  # 데이터 스케일링 계수 (α)
                    u_β = (u_max + u_min) / 2.  # 데이터 중심 값 (β)
                else:
                    u_α = np.amax(u)  # 최대값을 스케일링 계수로 사용
                    u_β = 0.  # 최소값 0으로 설정
    
            elif norm == False:
                u_α = np.std(u, axis=0)  # 표준편차를 스케일링 계수로 사용
                u_β = np.mean(u, axis=0)  # 평균을 중심값으로 사용
        
        else:
            u_α = 1.  # 정규화 적용 안 할 경우 α=1
            u_β = 0.  # 정규화 적용 안 할 경우 β=0
        
        return np.float32(u_α), np.float32(u_β)
    
    def out_stnd_nrml(self,u,α,β):
        '''
        입력 데이터를 표준화/정규화하는 함수
        u: 원본 데이터
        α: 스케일링 계수
        β: 중심 값 (평균 또는 최소값)
        
        (u - β) / α 를 계산하여 정규화된 데이터를 반환
        '''
        return (u - β) / α
    
    def u_stnd_nrml(self,output,α,β):
        '''
        정규화된 데이터를 원래 값으로 변환 (비정규화)하는 함수
        
        output: 정규화된 데이터
        α: 정규화에 사용된 스케일링 계수
        β: 정규화에 사용된 중심 값
        
        원래 데이터로 복원: output * α + β
        '''
        return output * α + β
