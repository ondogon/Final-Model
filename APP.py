# streamlit_app.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

# ---------------------- 경로 ----------------------
MODEL_PATH  = Path("arimax_lstm_model.keras")
SCALER_PATH = Path("scaler1.pkl")
CAL_PATH    = Path("calibrator.json")

# ---------------------- 캐시 로딩 ----------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(CAL_PATH, "r", encoding="utf-8") as f:
        cal = json.load(f)
    return model, scaler, cal

model, scaler, CAL = load_artifacts()
W_MAX = float(CAL.get("blend_w_max", 0.30))

# ---------------------- 모델 유틸 ----------------------
def _to_scaled_init(sod0, cat0):
    s_sod = scaler.transform(pd.DataFrame([[0, sod0, 0]],
        columns=["UV_time","Pred_SOD","Pred_CAT"]))[0][1]
    s_cat = scaler.transform(pd.DataFrame([[0, 0, cat0]],
        columns=["UV_time","Pred_SOD","Pred_CAT"]))[0][2]
    return s_sod, s_cat

def _scale_uv(uv_min):
    return scaler.transform(pd.DataFrame([[uv_min,0,0]],
        columns=["UV_time","Pred_SOD","Pred_CAT"]))[0][0]

def rollout_raw(uv_min, sod0, cat0, alpha=0.2, horizon=120, step=2):
    s_sod0, s_cat0 = _to_scaled_init(sod0, cat0)
    uv_s = _scale_uv(uv_min)
    seq = [[1, uv_s, s_sod0, s_cat0]] * 5
    s_hist, c_hist = [], []
    steps = horizon // step
    for _ in range(steps):
        x = np.array(seq[-5:], dtype=float).reshape(1,5,4)
        pred = model.predict(x, verbose=0)[0]
        last_s, last_c = seq[-1][2], seq[-1][3]
        ns = (1-alpha)*last_s + alpha*pred[0]
        nc = (1-alpha)*last_c + alpha*pred[1]
        s_hist.append(ns); c_hist.append(nc)
        seq.append([1, uv_s, ns, nc])
    sod = np.array([scaler.inverse_transform([[0,s,0]])[0][1] for s in s_hist])
    cat = np.array([scaler.inverse_transform([[0,0,c]])[0][2] for c in c_hist])
    t   = np.arange(0, steps*step, step)
    return t, sod, cat

def _smoothstep(u):  # 끝점 미분 0
    return u*u*(3-2*u)

def piecewise_smooth(times, t_kn, y_kn):
    times = np.asarray(times, float)
    t_kn  = np.asarray(t_kn,  float)
    y_kn  = np.asarray(y_kn,  float)
    order = np.argsort(t_kn)
    t_kn, y_kn = t_kn[order], y_kn[order]
    uniq = np.concatenate([[True], t_kn[1:] != t_kn[:-1]])
    t_kn, y_kn = t_kn[uniq], y_kn[uniq]
    y = np.empty_like(times, float)
    # 매듭점 값 고정
    for tk, yk in zip(t_kn, y_kn):
        y[np.isclose(times, tk)] = yk
    y[times <  t_kn[0]]  = y_kn[0]
    y[times >  t_kn[-1]] = y_kn[-1]
    for i in range(len(t_kn)-1):
        t0, t1 = t_kn[i], t_kn[i+1]
        y0, y1 = y_kn[i], y_kn[i+1]
        m = (times > t0) & (times < t1)
        if not np.any(m): continue
        u = (times[m]-t0)/(t1-t0)
        s = _smoothstep(u)
        y[m] = y0 + (y1-y0)*s
    return np.clip(y, 0.0, 1.0)

def bell_weight(times, t_kn, w_max):
    times = np.asarray(times, float)
    t_kn  = np.asarray(t_kn,  float)
    w = np.zeros_like(times, float)
    for i in range(len(t_kn)-1):
        t0, t1 = t_kn[i], t_kn[i+1]
        m = (times >= t0) & (times <= t1)
        if not np.any(m): continue
        u = (times[m]-t0)/(t1-t0)
        s = _smoothstep(u)
        w[m] = w_max * (s*(1-s)) * 4.0
    return w

@st.cache_data(show_spinner=False)
def predict_curve_final(uv_time_min, horizon_min=120, step_min=2):
    uv_key = str(int(uv_time_min))
    info = CAL.get("uv", {}).get(uv_key, None)
    if info is None:
        raise ValueError(f"UV {uv_time_min} min targets not found in calibrator.json")

    t_all = np.array(info["t"], dtype=float)
    # SOD
    maskS = ~pd.isna(info["SOD"])
    tS = t_all[maskS]
    yS = np.array([v for v in info["SOD"] if not pd.isna(v)], dtype=float)
    # CAT
    maskC = ~pd.isna(info["CAT"])
    tC = t_all[maskC]
    yC = np.clip(np.array([v for v in info["CAT"] if not pd.isna(v)], dtype=float), 0.0, 1.0)

    # 초기값: 0분 타깃 있으면 그 값, 없으면 0
    S0 = float(yS[tS==0][0]) if len(tS)>0 and (tS==0).any() else 0.0
    C0 = float(yC[tC==0][0]) if len(tC)>0 and (tC==0).any() else 0.0

    # LSTM 원시 곡선
    t, S_raw, C_raw = rollout_raw(int(uv_time_min), S0, C0, alpha=0.2,
                                  horizon=horizon_min, step=step_min)

    # 실험 타깃 보간(실험점 정확 통과)
    S_tar = piecewise_smooth(t, tS, yS) if len(tS)>=2 else np.interp(t, tS, yS)
    C_tar = piecewise_smooth(t, tC, yC) if len(tC)>=2 else np.interp(t, tC, yC)

    # 소량 블렌드(구간 중앙에서만 가설형태 가미)
    wS = bell_weight(t, tS, W_MAX) if len(tS)>=2 else 0.0
    wC = bell_weight(t, tC, W_MAX) if len(tC)>=2 else 0.0
    S = np.clip(S_tar + wS*(S_raw - S_tar), 0.0, 1.0)
    C = np.clip(C_tar + wC*(C_raw - C_tar), 0.0, 1.0)

    df = pd.DataFrame({"t_min": t, "SOD_pred_act": S, "CAT_pred_act": C})
    # 적분(0~120) — 총 활성량
    area_sod = float(np.trapz(df["SOD_pred_act"], df["t_min"]))
    area_cat = float(np.trapz(df["CAT_pred_act"], df["t_min"]))
    return df, area_sod, area_cat

# ---------------------- UI ----------------------
st.set_page_config(page_title="식물 스트레스 예측(최종 모델)", layout="wide")
st.title("🌱 식물 스트레스 예측 — 최종 모델")

with st.sidebar:
    uv_pick = st.radio("자외선 조사 시간(분) 선택", [15, 30], index=0, horizontal=True)
    show_markers = st.checkbox("마커 표시", value=True)
    st.caption("시간축은 0분~120분")
    st.caption("마우스를 그래프 위에 올려 활성도 확인")
# 예측 곡선 생성
curve, area_sod, area_cat = predict_curve_final(uv_pick, 120, 2)

# 메트릭(적분값)
c1, c2, _ = st.columns([1,1,3])
c1.metric(f"SOD 면적(0–120, UV {uv_pick}분)", f"{area_sod:.1f}")
c2.metric(f"CAT 면적(0–120, UV {uv_pick}분)", f"{area_cat:.1f}")

# Plotly
mode = "lines+markers" if show_markers else "lines"
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=curve["t_min"], y=curve["SOD_pred_act"], mode=mode, name="SOD",
    hovertemplate="시간 %{x:.0f}분<br>SOD %{y:.3f}", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(
    x=curve["t_min"], y=curve["CAT_pred_act"], mode=mode, name="CAT",
    hovertemplate="시간 %{x:.0f}분<br>CAT %{y:.3f}", line=dict(color="#ff7f0e")))

fig.update_layout(
    title=f"UV {uv_pick}분 — 예측 곡선",
    xaxis_title="경과 시간 (분)", yaxis_title="상대 활성(↑=more)",
    hovermode="x unified", template="plotly_white", legend_title_text=""
)
st.plotly_chart(fig, use_container_width=True)

# 데이터 다운로드
csv = curve.to_csv(index=False).encode("utf-8")
st.download_button(
    "CSV 다운로드 (t,SOD,CAT)", data=csv, file_name=f"curve_uv{uv_pick}.csv",
    mime="text/csv"
)

st.caption("생기부 끝!!")
st.caption("개발 기간 : 2025.04.17 - 2025.08.17, 약 123일)
