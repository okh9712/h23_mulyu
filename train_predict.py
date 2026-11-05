# -*- coding: utf-8 -*-
"""이력 Excel 학습 -> 후보 ML 모델 시계열 교차검증 -> 상위 3개 선정 -> 27~28년 예측 -> JSON/Excel 산출"""
import calendar, json, warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
from common import LINES, PRODUCTS, FLOWS, FLOW_COLS, SEGMENTS

warnings.filterwarnings("ignore")
HIST = "data/H23_이력데이터_2026.01-10.xlsx"
FUT = "data/H23_중장기생산계획_2027.01-2028.12.xlsx"
FEATS = [f"{l}|{p}" for l in LINES for p in PRODUCTS]

# ---------- 데이터 적재 ----------
hp = pd.read_excel(HIST, sheet_name="일별_생산계획_WIP")
hf = pd.read_excel(HIST, sheet_name="일별_물동량")
hp["일자"] = pd.to_datetime(hp["일자"]); hf["일자"] = pd.to_datetime(hf["일자"])
X = hp.pivot_table(index="일자", columns=["라인", "제품"], values="생산계획(Lot)", aggfunc="sum")
X.columns = [f"{a}|{b}" for a, b in X.columns]
X = X[FEATS]
Y = hf.set_index("일자")[FLOW_COLS].loc[X.index]
month = X.index.month.values

# ---------- 후보 모델 ----------
def tt(est):  # 입력/출력 표준화 래퍼
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), est), transformer=StandardScaler())

class Normalized(BaseEstimator, RegressorMixin):
    """물량 정규화 래퍼: 총 생산계획으로 나눈 비율(믹스·단위이동량)을 학습 -> 물량이 이력 범위를 벗어나도 비례 확장
    (트리/커널 모델의 외삽 한계 보완)"""
    def __init__(self, est=None):
        self.est = est
    def fit(self, X, y):
        tot = X.sum(1, keepdims=True)
        self.m_ = clone(self.est).fit(X / tot, y / tot)
        return self
    def predict(self, X):
        tot = X.sum(1, keepdims=True)
        return self.m_.predict(X / tot) * tot


CANDS = {
    "Ridge 회귀": lambda: tt(Ridge(alpha=1.0)),
    "ElasticNet": lambda: tt(ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=20000)),
    "SVR (RBF)": lambda: Normalized(tt(MultiOutputRegressor(SVR(kernel="rbf", C=10, epsilon=0.02, gamma="scale")))),
    "Random Forest": lambda: Normalized(RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1)),
    "Extra Trees": lambda: Normalized(ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1)),
    "Gradient Boosting": lambda: Normalized(MultiOutputRegressor(GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.06, subsample=0.8, random_state=0), n_jobs=-1)),
}


def month_acc(pred, act, m):
    """월 합계 기준 정확도(%) = 100 - MAPE (흐름별 월 합계 비교)"""
    accs = []
    for mm in np.unique(m):
        idx = m == mm
        a, p = act[idx].sum(0), pred[idx].sum(0)
        accs.append(100 - np.abs(p - a) / a * 100)
    return np.array(accs)  # (months, targets)


# ---------- 확장 윈도우 시계열 CV (1~k월 학습 -> k+1월 검증) ----------
FOLDS = [5, 6, 7, 8, 9]
cv_pred = {n: [] for n in CANDS}
fold_rows = []
for n, mk in CANDS.items():
    accs, d_r2, d_mape = [], [], []
    for k in FOLDS:
        tr, te = month <= k, month == k + 1
        mdl = mk().fit(X[tr].values, Y[tr].values)
        pr = mdl.predict(X[te].values)
        a = month_acc(pr, Y[te].values, month[te])[0]
        accs.append(a)
        d_r2.append(r2_score(Y[te].values, pr))
        d_mape.append(np.mean(np.abs(pr - Y[te].values) / Y[te].values) * 100)
        fold_rows.append({"model": n, "test_month": k + 1, "acc": float(a.mean())})
    cv_pred[n] = dict(acc=np.array(accs), r2=np.mean(d_r2), dmape=np.mean(d_mape))

rank = []
for n, r in cv_pred.items():
    rank.append({"model": n, "month_acc": float(r["acc"].mean()), "min_acc": float(r["acc"].mean(1).min()),
                 "daily_mape": float(r["dmape"]), "daily_r2": float(r["r2"])})
rank.sort(key=lambda d: -d["month_acc"])
for i, r in enumerate(rank, 1):
    r["rank"] = i
    print(f'{i}. {r["model"]:18s} 월합계정확도 {r["month_acc"]:.2f}%  일MAPE {r["daily_mape"]:.2f}%  R2 {r["daily_r2"]:.3f}')
top = [r["model"] for r in rank[:3]]
w = np.array([r["month_acc"] for r in rank[:3]])
w = (100 - w) ** -1                      # 오차 역수 가중
w = w / w.sum()
weights = dict(zip(top, w))
print("선정:", top, np.round(w, 3))

# 앙상블 CV 정확도 (선정 모델 가중 평균)
ens_accs = []
for k in FOLDS:
    tr, te = month <= k, month == k + 1
    pr = sum(weights[n] * CANDS[n]().fit(X[tr].values, Y[tr].values).predict(X[te].values) for n in top)
    ens_accs.append(month_acc(pr, Y[te].values, month[te])[0])
ens_accs = np.array(ens_accs)
seg_acc = {}
for s in SEGMENTS:  # 구간(합계) 단위 월 정확도
    ix = [i for i, f in enumerate(FLOWS) if f["seg"] == s["id"]]
    vals = []
    for k in FOLDS:
        tr, te = month <= k, month == k + 1
        pr = sum(weights[n] * CANDS[n]().fit(X[tr].values, Y[tr].values).predict(X[te].values) for n in top)[:, ix].sum(1).sum()
        vals.append(100 - abs(pr - Y[te].values[:, ix].sum()) / Y[te].values[:, ix].sum() * 100)
    seg_acc[s["id"]] = float(np.mean(vals))
print("앙상블 월합계 정확도: %.2f%%" % ens_accs.mean(), seg_acc)

# ---------- 최종 학습 (26.1~10 전체) ----------
final = {n: CANDS[n]().fit(X.values, Y.values) for n in top}
ridge_raw = Ridge(alpha=1.0).fit(X.values, Y.values)           # 브라우저 재예측용(선형)
fit_acc = {n: float(month_acc(m.predict(X.values), Y.values, month).mean()) for n, m in final.items()}

# ---------- 27~28 예측 ----------
fp = pd.read_excel(FUT, sheet_name="월별_생산계획")
fut_months = sorted(fp["년월"].unique())
Fm = fp.pivot_table(index="년월", columns=["라인", "제품"], values="생산계획(Lot)", aggfunc="sum")
Fm.columns = [f"{a}|{b}" for a, b in Fm.columns]
Fm = Fm[FEATS]
days = np.array([calendar.monthrange(int(m[:4]), int(m[5:]))[1] for m in fut_months])
Xd = Fm.loc[fut_months].values / days[:, None]
pred_by_model = {n: final[n].predict(Xd) * days[:, None] for n in top}
pred_ens = sum(weights[n] * pred_by_model[n] for n in top)
pred_ens = np.clip(pred_ens, 0, None)

# ---------- JSON 구성 ----------
hist_months = [f"2026-{m:02d}" for m in range(1, 11)]
monthly_act = hf.assign(ym=hf["일자"].dt.strftime("%Y-%m")).groupby("ym")[FLOW_COLS].sum()
flows_rows = []
for ym in hist_months:
    for i, f in enumerate(FLOWS):
        flows_rows.append([ym, f["col"], int(monthly_act.loc[ym, f["col"]]), "actual"])
for j, ym in enumerate(fut_months):
    for i, f in enumerate(FLOWS):
        flows_rows.append([ym, f["col"], int(round(pred_ens[j, i])), "forecast"])
model_flow = {n: {ym: [int(round(v)) for v in pred_by_model[n][j]] for j, ym in enumerate(fut_months)} for n in top}

hp["ym"] = hp["일자"].dt.strftime("%Y-%m")
pm = hp.groupby(["ym", "라인", "제품"]).agg(plan=("생산계획(Lot)", "sum"), wip=("재공(WIP,Lot)", "mean")).reset_index()
plan_rows = [[r.ym, r.라인, r.제품, int(r.plan), int(round(r.wip)), "actual"] for r in pm.itertuples()]
# WIP 비율(재공/일평균 계획) -> 예측기간 재공 추정
hp_days = hp.groupby("ym")["일자"].nunique()
pm["daily"] = pm.apply(lambda r: r.plan / hp_days[r.ym], axis=1)
ratio = (pm.groupby(["라인", "제품"]).apply(lambda g: g.wip.sum() / g.daily.sum())).to_dict()
for j, ym in enumerate(fut_months):
    for l in LINES:
        for p in PRODUCTS:
            q = int(Fm.loc[ym, f"{l}|{p}"])
            plan_rows.append([ym, l, p, q, int(round(q / days[j] * ratio[(l, p)])), "forecast"])

coef = {"intercept": ridge_raw.intercept_.tolist(), "coef": ridge_raw.coef_.tolist(), "feats": FEATS, "targets": FLOW_COLS}
ridge_cv = cv_pred["Ridge 회귀"]["acc"].mean()

DESC = {
    "Ridge 회귀": "L2 규제 선형회귀. 생산계획(Lot) 대비 이동량이 선형적으로 비례하는 구조를 안정적으로 포착, 계획이 이력 범위를 넘어도 외삽 가능",
    "ElasticNet": "L1+L2 규제 선형모델. 불필요한 제품·라인 변수의 영향을 줄여 과적합 억제",
    "SVR (RBF)": "커널 기반 비선형 회귀(물량 정규화 적용). 제품 믹스 변화에 따른 완만한 비선형 효과 반영",
    "Random Forest": "다수 결정트리 앙상블(물량 정규화 적용). 비선형·상호작용에 강함",
    "Extra Trees": "무작위 분할 트리 앙상블(물량 정규화 적용). 노이즈에 강건",
    "Gradient Boosting": "순차적 트리 부스팅(물량 정규화 적용: 총계획 대비 비율을 학습해 물량 증가 시에도 비례 확장). 제품 믹스에 따른 비선형 패턴 학습",
}
out = {
    "hist_months": hist_months, "fut_months": fut_months,
    "flows": [{k: f[k] for k in ("col", "seg", "type", "segname", "dir", "floor", "from", "to")} for f in FLOWS],
    "segments": SEGMENTS, "lines": LINES, "products": PRODUCTS,
    "flow_rows": flows_rows, "plan_rows": plan_rows,
    "ridge": coef, "wip_ratio": {f"{l}|{p}": v for (l, p), v in ratio.items()},
    "model_flow": model_flow,
    "models": {
        "ranking": rank, "selected": top, "weights": {k: float(v) for k, v in weights.items()},
        "desc": DESC, "ens_month_acc": float(ens_accs.mean()), "ens_fold_acc": [float(a.mean()) for a in ens_accs],
        "seg_acc": seg_acc, "fit_acc": fit_acc, "folds": [k + 1 for k in FOLDS],
        "n_train_days": int(len(X)), "n_features": len(FEATS), "n_targets": len(FLOW_COLS),
    },
}
json.dump(out, open("data/result.json", "w", encoding="utf-8"), ensure_ascii=False)

# ---------- 예측결과 Excel ----------
rows = []
for ym in fut_months:
    nd = calendar.monthrange(int(ym[:4]), int(ym[5:]))[1]
    j = fut_months.index(ym)
    for i, f in enumerate(FLOWS):
        v = pred_ens[j, i]
        rows.append({"년월": ym, "구간유형": f["type"], "구간": f["segname"], "방향": f["dir"], "층": f["floor"],
                     "FROM": f["from"], "TO": f["to"], "예측 물동량(Lot/월)": int(round(v)),
                     "예측 물동량(Lot/Hr)": round(v / nd / 24, 1)})
pr_df = pd.DataFrame(rows)
seg_rows = []
for s in SEGMENTS:
    for ym in hist_months + fut_months:
        nd = calendar.monthrange(int(ym[:4]), int(ym[5:]))[1]
        v = sum(r[2] for r in flows_rows if r[0] == ym and next(f for f in FLOWS if f["col"] == r[1])["seg"] == s["id"])
        u = v / nd / 24
        seg_rows.append({"년월": ym, "구분": "이력" if ym in hist_months else "예측", "구간": s["name"],
                         "물동량(Lot/월)": int(v), "물동량(Lot/Hr)": round(u, 1), "MAX CAPA(Lot/Hr)": s["cap"],
                         "부하율(%)": round(u / s["cap"] * 100, 1),
                         "상태": "초과" if u > s["cap"] else ("주의(90%↑)" if u > .9 * s["cap"] else "정상")})
seg_df = pd.DataFrame(seg_rows)
mdl_df = pd.DataFrame([{"순위": r["rank"], "모델": r["model"], "선정": "O" if r["model"] in top else "",
                        "앙상블 가중치": round(weights.get(r["model"], 0), 3), "월합계 정확도(%) CV": round(r["month_acc"], 2),
                        "최저 월 정확도(%)": round(r["min_acc"], 2), "일별 MAPE(%)": round(r["daily_mape"], 2),
                        "일별 R2": round(r["daily_r2"], 3)} for r in rank])
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
with pd.ExcelWriter("data/H23_물동량_예측결과_2027.01-2028.12.xlsx", engine="openpyxl") as w:
    seg_df.to_excel(w, sheet_name="구간별_부하율", index=False)
    pr_df.to_excel(w, sheet_name="월별_방향층별_예측", index=False)
    mdl_df.to_excel(w, sheet_name="모델_성능", index=False)
    for ws in w.sheets.values():
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="1F3A5F")
            c.alignment = Alignment(horizontal="center", wrap_text=True)
        ws.freeze_panes = "A2"
        for i, col in enumerate(ws.columns, 1):
            ws.column_dimensions[get_column_letter(i)].width = min(max(len(str(c.value or "")) for c in col) + 4, 26)

# 초과 요약
for s in SEGMENTS:
    sub = seg_df[(seg_df["구간"] == s["name"]) & (seg_df["구분"] == "예측")]
    ov = sub[sub["상태"] == "초과"]
    print(s["name"], "최대부하율", sub["부하율(%)"].max(), "첫 초과:", ov["년월"].iloc[0] if len(ov) else "-", "초과월수", len(ov))
