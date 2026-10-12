/*
 * ann.js — 인공신경망과 선형회귀 계산 코드
 * 데이터처리개론 4. 인공신경망 기초 과제
 *
 * 구조: 입력층(d개) → 은닉층 Dense(H, sigmoid) → 출력층 Dense(1)
 * 학습: 평균제곱오차(MSE)를 오차 역전파로 미분하고 Adam(경사하강법 개선판)으로 가중치를 고침
 * Colab 코드의 keras.Sequential([Input(d), Dense(H, "sigmoid"), Dense(1)])과 같은 구조를 직접 구현한 것
 */
(function (global) {
  'use strict';

  // 같은 결과가 다시 나오도록 시드를 고정한 난수
  function makeRandom(seed) {
    let s = seed;
    const rand = () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
    const gauss = () => {                       // 정규분포 난수 (Box-Muller)
      let u = 0, v = 0; while (!u) u = rand(); while (!v) v = rand();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    return { rand, gauss };
  }

  // 활성화 함수 σ(z) = 1 / (1 + e^-z),  미분 σ'(z) = σ(z)(1 − σ(z))
  const sigmoid = z => 1 / (1 + Math.exp(-z));

  class NeuralNet {
    /**
     * @param {number} d  입력 노드 개수
     * @param {number} H  은닉 노드 개수
     * @param {object} o  초기값 설정 {seed, wScale, bShift, vScale, c0}
     */
    constructor(d, H, o = {}) {
      const { seed = 11, wScale = 3, bShift = 0.5, vScale = 0.3, c0 = 0.1 } = o;
      const R = makeRandom(seed);
      this.d = d; this.H = H;
      this.W = [];   // 입력 → 은닉 가중치 (H × d)
      this.b = [];   // 은닉 바이어스 (H)
      this.v = [];   // 은닉 → 출력 가중치 (H)
      for (let j = 0; j < H; j++) {
        const w = Array.from({ length: d }, () => R.gauss() * wScale);
        this.W.push(w);
        this.b.push(-w.reduce((a, x) => a + x * R.rand() * bShift, 0));  // 시그모이드가 데이터 범위 안에서 꺾이도록
        this.v.push(R.gauss() * vScale);
      }
      this.c = c0;   // 출력 바이어스
      const P = H * d + 2 * H + 1;              // 학습되는 값의 총 개수
      this.m = new Float64Array(P); this.s = new Float64Array(P); this.t = 0;   // Adam 상태
    }

    // 은닉 노드 j의 입력 합: z = Σ(w·x) + b
    z(j, x) { let z = this.b[j]; const w = this.W[j]; for (let k = 0; k < this.d; k++) z += w[k] * x[k]; return z; }

    // 순전파: ŷ = Σ v_j·σ(z_j) + c
    predict(x) { let o = this.c; for (let j = 0; j < this.H; j++) o += this.v[j] * sigmoid(this.z(j, x)); return o; }

    /**
     * 학습 1 epoch: 전체 학습 데이터로 기울기를 구하고 가중치를 한 번 고침
     * @returns 이번 epoch의 평균제곱오차
     */
    trainStep(X, Y, idx, lr) {
      const { d, H, W, b, v } = this, n = idx.length;
      const gW = W.map(() => new Float64Array(d)), gb = new Float64Array(H), gv = new Float64Array(H);
      const a = new Float64Array(H); let gc = 0, loss = 0;
      for (const i of idx) {
        const x = X[i]; let o = this.c;
        for (let j = 0; j < H; j++) { a[j] = sigmoid(this.z(j, x)); o += v[j] * a[j]; }   // 순전파
        const r = Y[i] - o; loss += r * r;
        const e = -2 * r / n;                                // ∂E/∂ŷ
        gc += e;
        for (let j = 0; j < H; j++) {                        // 역전파 (연쇄법칙)
          gv[j] += e * a[j];                                 // ∂E/∂v_j
          const dz = e * v[j] * a[j] * (1 - a[j]);           // ∂E/∂z_j = ∂E/∂ŷ · v_j · σ'(z_j)
          gb[j] += dz;
          for (let k = 0; k < d; k++) gW[j][k] += dz * x[k]; // ∂E/∂w_jk
        }
      }
      // Adam: 기울기의 이동평균(m)과 크기(s)로 가중치마다 이동 폭을 조절
      this.t++; const b1 = 0.9, b2 = 0.999, m = this.m, s = this.s, t = this.t;
      const upd = (q, g) => {
        m[q] = b1 * m[q] + (1 - b1) * g; s[q] = b2 * s[q] + (1 - b2) * g * g;
        return lr * (m[q] / (1 - b1 ** t)) / (Math.sqrt(s[q] / (1 - b2 ** t)) + 1e-8);
      };
      let q = 0;
      for (let j = 0; j < H; j++) for (let k = 0; k < d; k++) W[j][k] -= upd(q++, gW[j][k]);
      for (let j = 0; j < H; j++) { b[j] -= upd(q++, gb[j]); v[j] -= upd(q++, gv[j]); }
      this.c -= upd(q, gc);
      return loss / n;
    }
  }

  // 선형회귀: 정규방정식 (XᵀX)β = XᵀY 을 가우스 소거법으로 풀어 최소제곱 해를 구함
  function linearRegression(X, Y, idx) {
    const d = X[idx[0]].length, p = d + 1;
    const M = Array.from({ length: p }, () => new Array(p + 1).fill(0));
    for (const i of idx) {
      const x = [1, ...X[i]];
      for (let a = 0; a < p; a++) { for (let c = 0; c < p; c++) M[a][c] += x[a] * x[c]; M[a][p] += x[a] * Y[i]; }
    }
    for (let a = 1; a < p; a++) M[a][a] += 1e-9;
    for (let c = 0; c < p; c++) {
      let piv = c; for (let r = c + 1; r < p; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = 0; r < p; r++) { if (r === c || !M[c][c]) continue; const f = M[r][c] / M[c][c]; for (let k = c; k <= p; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map((r, i) => r[i] ? r[p] / r[i] : 0);   // [상수, a1, a2, ...]
  }
  const predictLinear = (beta, x) => beta[0] + x.reduce((s, v, k) => s + beta[k + 1] * v, 0);

  global.ANN = { makeRandom, sigmoid, NeuralNet, linearRegression, predictLinear };
})(window);
