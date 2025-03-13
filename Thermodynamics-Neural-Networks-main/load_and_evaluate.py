from classThermodynamicsANNs.ThermodynamicsANNs import TANNs
from classThermodynamicsANNs.PreProcessingOperations import preProcessing

# 데이터 로드 및 전처리
silent = True
data = preProcessing('reference_data/input_data', 'reference_data/output_data', silent)
uN_T, uN_V, uN_Tt, oN_T, oN_V, oN_Tt, nrm_inp, nrm_out = data.Out()

# 모델 초기화 및 빌드
ThermoANN = TANNs(nrm_inp, nrm_out, silent)
ThermoANN.build((None, 4))  # 입력 형식 정의

# 저장된 가중치 로드
print("\n... Loading saved weights")
ThermoANN.load_weights('./output_data/ThermoTANN_weights')  # 저장된 경로 확인
print("Weights loaded successfully!")

# 가중치를 로드한 후, 평가나 예측 전에 모델을 컴파일
ThermoANN.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 평가
test_loss = ThermoANN.evaluate(uN_Tt, oN_Tt)
print(f"Test Loss: {test_loss}")

# 예측
predictions = ThermoANN.predict(uN_Tt)
print("Predictions on test data:", predictions)
