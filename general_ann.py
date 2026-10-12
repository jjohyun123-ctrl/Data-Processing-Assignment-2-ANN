"""
어떤 CSV/엑셀 데이터든 인공신경망으로 예측하는 범용 코드
2번 설정 셀의 FILE_PATH, X_COLS, Y_COL만 바꾸면 됩니다.
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
from sklearn.metrics import r2_score
from tensorflow import keras

# %% 2. 설정 — 다른 데이터를 쓸 때는 이 셀만 고치면 됨
FILE_PATH = "/content/울산항_선박제원_정제.csv"   # csv, xlsx 모두 가능
X_COLS = ["선박길이", "선박너비", "선박깊이"]      # 입력으로 쓸 열 이름들 (개수 = 입력 노드 개수)
Y_COL = "총톤수"                                 # 예측할 열 이름 (출력 노드 1개)

HIDDEN_NODES = 8          # 은닉층 노드(o) 개수
EPOCHS = 100              # 반복 학습 횟수
BATCH_SIZE = 64
LEARNING_RATE = 0.01

# %% 3. 데이터 불러오기 (인코딩 자동 처리)
def load_data(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ["utf-8-sig", "cp949"]:          # 공공데이터는 cp949인 경우가 많음
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("파일 인코딩을 읽을 수 없습니다.")

data = load_data(FILE_PATH)
for col in X_COLS + [Y_COL]:                      # "1,234" 같은 쉼표 숫자도 숫자로 변환
    data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", ""), errors="coerce")
data = data[X_COLS + [Y_COL]].dropna()
print("사용할 데이터 개수:", len(data))
data.describe()

# %% 4. 전처리
X = data[X_COLS].values
Y = data[[Y_COL]].values
x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
X_s = x_scaler.fit_transform(X)
Y_s = y_scaler.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_s, Y_s, test_size=0.2, random_state=42)

# %% 5. 인공신경망 만들기
model = keras.Sequential([
    keras.Input(shape=(len(X_COLS),)),
    keras.layers.Dense(HIDDEN_NODES, activation="sigmoid"),
    keras.layers.Dense(1)
])
model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")
history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_test, Y_test), verbose=0)
plt.plot(history.history["loss"], label="train"); plt.plot(history.history["val_loss"], label="test")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.show()

# %% 6. 결과 비교
y_true = y_scaler.inverse_transform(Y_test)
pred_nn = y_scaler.inverse_transform(model.predict(X_test, verbose=0))
lr = LinearRegression().fit(X_train, Y_train)
pred_lr = y_scaler.inverse_transform(lr.predict(X_test))
print(f"인공신경망 R² = {r2_score(y_true, pred_nn):.3f}")
print(f"선형회귀   R² = {r2_score(y_true, pred_lr):.3f}")

if len(X_COLS) == 1:     # 입력 1개: 데이터 위에 예측선
    order = np.argsort(X_s[:, 0]); xp = x_scaler.inverse_transform(X_s[order])
    plt.scatter(X, Y, s=6, color="gray")
    plt.plot(xp, y_scaler.inverse_transform(model.predict(X_s[order], verbose=0)), "r", label="Neural network")
    plt.plot(xp, y_scaler.inverse_transform(lr.predict(X_s[order])), "b", label="Linear regression")
else:                    # 입력 여러 개: 실제값 vs 예측값
    plt.scatter(y_true, pred_lr, s=5, alpha=0.5, label="Linear regression")
    plt.scatter(y_true, pred_nn, s=5, alpha=0.5, label="Neural network")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--")
    plt.xlabel("actual"); plt.ylabel("predicted")
plt.legend(); plt.show()

# %% 7. 원하는 값을 넣어 예측하기
values = [float(input(f"{col} 값을 입력하세요: ")) for col in X_COLS]
x_new = x_scaler.transform([values])
print(f"인공신경망 예측 {Y_COL}: {y_scaler.inverse_transform(model.predict(x_new, verbose=0))[0, 0]:,.2f}")
print(f"선형회귀   예측 {Y_COL}: {y_scaler.inverse_transform(lr.predict(x_new))[0, 0]:,.2f}")
