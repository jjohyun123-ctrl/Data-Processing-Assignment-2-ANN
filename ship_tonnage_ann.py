"""
선박 주요 치수(길이, 너비, 깊이)로 총톤수를 예측하는 인공신경망
자료: 울산항만공사_선박제원정보_20260527.csv (공공데이터포털)
데이터처리개론 2026년 2학기 4. 인공신경망 과제
Google Colab에서 셀 단위(# %%)로 나눠 실행하면 됩니다.
"""

# %% 1. 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow import keras

# %% 2. 설정
FILE_PATH = "/content/울산항만공사_선박제원정보_20260527.csv"   # 공공데이터포털에서 받은 원본 파일
X_COLS = ["선박길이", "선박너비", "선박깊이"]   # 입력 노드 3개 (L, B, D)
Y_COL = "총톤수"                              # 출력 노드 1개 (GT)

HIDDEN_NODES = 6          # 은닉층 노드(o) 개수
EPOCHS = 60               # 전체 데이터를 몇 번 반복 학습할지
BATCH_SIZE = 256          # 가중치를 한 번 고칠 때 사용하는 데이터 개수
LEARNING_RATE = 0.01      # 경사하강법의 이동 폭

# %% 3. 원본 데이터 불러오기
try:
    raw = pd.read_csv(FILE_PATH, encoding="cp949")    # 공공데이터는 대부분 cp949(EUC-KR)
except UnicodeDecodeError:
    raw = pd.read_csv(FILE_PATH, encoding="utf-8-sig")
print("원본 행 수:", len(raw))
raw.head()

# %% 4. 데이터 정리 (이상치 제거)
df = raw[raw["사용여부"] == "Y"]
print("사용여부 Y:", len(df))

df = df[X_COLS + [Y_COL]]
df = df[(df > 0).all(axis=1)]
print("0 값 제거:", len(df))

df = df[(df["총톤수"] < 999999) & (df["선박길이"] < 999) & (df["선박너비"] < 999) & (df["선박깊이"] < 99.9)]
print("자리채움 값(999999 등) 제거:", len(df))

L, B, D = df["선박길이"], df["선박너비"], df["선박깊이"]
df = df[(L / B).between(2.5, 10) & (B / D).between(0.8, 4) & L.between(10, 400)]
print("배 모양으로 불가능한 치수 제거:", len(df))

C = df["총톤수"] / (df["선박길이"] * df["선박너비"] * df["선박깊이"])   # 부피 대비 톤수 비율
df = df[C.between(0.15, 0.8)]
print("톤수와 치수가 맞지 않는 기록 제거:", len(df))

df = df.drop_duplicates()
print("중복 제거 후 최종:", len(df))
df.describe()

# %% 5. 전처리 (0~1 정규화 + 학습/테스트 분리)
X = df[X_COLS].values
Y = df[[Y_COL]].values

x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
X_s = x_scaler.fit_transform(X)
Y_s = y_scaler.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_s, Y_s, test_size=0.2, random_state=42)
print("학습용:", len(X_train), " 테스트용:", len(X_test))

# %% 6. 인공신경망 만들기 (layer → Dense, o → node)
model = keras.Sequential([
    keras.Input(shape=(3,)),                                  # 입력층: 길이, 너비, 깊이
    keras.layers.Dense(HIDDEN_NODES, activation="sigmoid"),   # 은닉층: Σ(입력×w)+b → 시그모이드
    keras.layers.Dense(1)                                     # 출력층: 총톤수
])
model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")
model.summary()

# %% 7. 학습 + 오차 감소 그래프
history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_test, Y_test), verbose=0)

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="test loss")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("MSE (scaled)"); plt.legend()
plt.title("Loss during training"); plt.show()

# %% 8. 선형회귀와 비교
y_true = y_scaler.inverse_transform(Y_test)
pred_nn = y_scaler.inverse_transform(model.predict(X_test, verbose=0))
lr = LinearRegression().fit(X_train, Y_train)
pred_lr = y_scaler.inverse_transform(lr.predict(X_test))

for name, p in [("인공신경망", pred_nn), ("선형회귀", pred_lr)]:
    print(f"{name}: R² = {r2_score(y_true, p):.3f}, 평균절대오차 = {mean_absolute_error(y_true, p):,.0f} 톤, "
          f"음수 예측 비율 = {(p < 0).mean() * 100:.1f}%")

plt.figure(figsize=(6, 6))
plt.scatter(y_true, pred_lr, s=4, alpha=0.4, label="Linear regression")
plt.scatter(y_true, pred_nn, s=4, alpha=0.4, label="Neural network")
lim = [0, y_true.max()]
plt.plot(lim, lim, "k--", label="perfect")
plt.xlabel("actual GT"); plt.ylabel("predicted GT"); plt.legend(); plt.title("Actual vs Predicted (test)")
plt.show()

# %% 9. 학습된 가중치 확인
for i, layer in enumerate(model.layers):
    w, b = layer.get_weights()
    print(f"[층 {i+1}] 가중치 행렬 {w.shape} → {w.size}개, 바이어스 {b.size}개")
print("은닉층 가중치 (행: 길이/너비/깊이, 열: 은닉 노드)")
print(np.round(model.layers[0].get_weights()[0], 2))

# %% 10. 선박 치수를 넣어 총톤수 예측
L_in = float(input("선박길이 L (m): "))
B_in = float(input("선박너비 B (m): "))
D_in = float(input("선박깊이 D (m): "))

x_new = x_scaler.transform([[L_in, B_in, D_in]])
gt_nn = y_scaler.inverse_transform(model.predict(x_new, verbose=0))[0, 0]
gt_lr = y_scaler.inverse_transform(lr.predict(x_new))[0, 0]
print(f"인공신경망 예측 총톤수: {gt_nn:,.0f} GT")
print(f"선형회귀   예측 총톤수: {gt_lr:,.0f} GT")
