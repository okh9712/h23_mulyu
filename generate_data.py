# -*- coding: utf-8 -*-
"""더미 데이터 생성: 26.1~26.10 일별 이력(생산계획/WIP/물동량) + 27.1~28.12 월별 중장기 생산계획 -> Excel"""
import calendar
import numpy as np
import pandas as pd
from common import LINES, PRODUCTS, FLOWS, FLOW_COLS, SEGMENTS

rng = np.random.default_rng(20261105)
P = len(PRODUCTS)

# ---------------- 생산계획(월) 생성 ----------------
BASE = {"15L": 30000, "16L": 27000, "17L": 19000}          # 26.1월 라인별 월 생산계획(Lot)
GROWTH = {"15L": 0.0080, "16L": 0.0055, "17L": 0.0105}     # 월 성장률
MIX0 = {"15L": [.34, .26, .21, .13, .06], "16L": [.36, .25, .20, .13, .06], "17L": [.32, .26, .21, .14, .07]}
MIX1 = {"15L": [.16, .20, .24, .20, .20], "16L": [.18, .21, .25, .20, .16], "17L": [.14, .19, .23, .21, .23]}
N_MONTHS = 36  # 2026-01 ~ 2028-12


def month_plan(t, line):
    s0, s1 = np.array(MIX0[line]), np.array(MIX1[line])
    w = t / (N_MONTHS - 1)
    share = (1 - w) * s0 + w * s1
    share = share / share.sum()
    total = BASE[line] * (1 + GROWTH[line]) ** t
    nd = calendar.monthrange(2026 + t // 12, t % 12 + 1)[1]
    total *= nd / 31 * (1 + rng.normal(0, 0.012))   # 월 일수 반영
    return total * share * (1 + rng.normal(0, 0.01, P))


plan_m = {}  # (t,line)->array(P)
for t in range(N_MONTHS):
    for ln in LINES:
        plan_m[(t, ln)] = month_plan(t, ln)

# ---------------- 물동량 발생 모델(정답 생성 규칙, 모델은 이를 모른다) ----------------
LIFT_UP = np.array([16, 18, 20, 22, 26.0])
LINE_MOD = {"15L": 1.00, "16L": 1.05, "17L": 0.95}
BR_X = np.array([30, 34, 38, 42, 52.0])      # 15-16 Bridge 제품별 이동 횟수/Lot
BR_Z = np.array([8, 10, 11, 13, 16.0])        # 15-17 Bridge
UP_SHARE = np.array([.62, .58, .55, .50, .45])  # 상부층 비중
REF_TOTAL = {ln: BASE[ln] / 31 for ln in LINES}
# 최초(26.1월) 목표 부하율 -> 스케일 보정
TARGET_UTIL0 = {"B1516": .70, "B1517": .74, "L15": .80, "L16": .74, "L17": .55}
CAP = {s["id"]: s["cap"] for s in SEGMENTS}
SCALE = {k: 1.0 for k in CAP}


def true_flows(plan, scale=None):
    """plan: dict line-> array(P) (일 단위 Lot) -> dict col -> 일 물동량(Lot)"""
    sc = scale or SCALE
    out = {}
    cong = {ln: 1 + 0.5 * (plan[ln].sum() / REF_TOTAL[ln] - 1) ** 2 for ln in LINES}
    for ln in LINES:
        up = (plan[ln] * LIFT_UP * LINE_MOD[ln]).sum() * cong[ln] * sc["L" + ln[:2]]
        out[f"Lifter {ln} 상승"] = up
        out[f"Lifter {ln} 하강"] = up * 0.95
    for seg, a, b, coef, cross in (("B1516", "15L", "16L", BR_X, .30), ("B1517", "15L", "17L", BR_Z, .25)):
        s = sc[seg]
        na, nb = a[:2], b[:2]
        fab = (plan[a] + cross * plan[b]) * coef * cong[a] * s
        fba = (plan[b] + cross * plan[a]) * coef * 0.95 * cong[b] * s
        sname = f"Bridge {na}-{nb}"
        out[f"{sname} {na}→{nb} 상부"] = (fab * UP_SHARE).sum()
        out[f"{sname} {na}→{nb} 하부"] = (fab * (1 - UP_SHARE)).sum()
        out[f"{sname} {nb}→{na} 상부"] = (fba * (UP_SHARE - 0.02)).sum()
        out[f"{sname} {nb}→{na} 하부"] = (fba * (1 - UP_SHARE + 0.02)).sum()
    return out


def seg_hourly(flows_day, seg):
    cols = [f["col"] for f in FLOWS if f["seg"] == seg]
    return sum(flows_day[c] for c in cols) / 24.0


# 스케일 보정 (26.1월 기준)
d0 = {ln: plan_m[(0, ln)] / 31 for ln in LINES}
f0 = true_flows(d0)
for seg in CAP:
    SCALE[seg] = TARGET_UTIL0[seg] * CAP[seg] / seg_hourly(f0, seg)

# 월 단위 숨은 변동(설비 상태 등) : 흐름별 σ=1.5%
month_eff = {(t, c): rng.normal(0, 0.015) for t in range(N_MONTHS) for c in FLOW_COLS}

# ---------------- 26.1~26.10 일별 이력 ----------------
dates = pd.date_range("2026-01-01", "2026-10-31", freq="D")
CT = {"15L": 38, "16L": 36, "17L": 34}  # 사이클타임(일) -> WIP 산정
plan_rows, flow_rows = [], []
daily_plan = {ln: [] for ln in LINES}
for d in dates:
    t = d.month - 1
    nd = calendar.monthrange(d.year, d.month)[1]
    dp = {}
    for ln in LINES:
        dp[ln] = plan_m[(t, ln)] / nd * np.exp(rng.normal(0, 0.07, P))
        daily_plan[ln].append(dp[ln])
    tf = true_flows(dp)
    row = {"일자": d.date()}
    for c in FLOW_COLS:
        row[c] = round(tf[c] * (1 + month_eff[(t, c)]) * np.exp(rng.normal(0, 0.04)))
    flow_rows.append(row)

wip_rows = []
for ln in LINES:
    arr = np.array(daily_plan[ln])                    # (days,P)
    ew = pd.DataFrame(arr).ewm(span=22, adjust=False).mean().values
    for i, d in enumerate(dates):
        for p in range(P):
            wip = ew[i, p] * CT[ln] * np.exp(rng.normal(0, 0.025))
            plan_rows.append({"일자": d.date(), "라인": ln, "제품": PRODUCTS[p],
                              "생산계획(Lot)": round(arr[i, p]), "재공(WIP,Lot)": round(wip)})
hist_plan = pd.DataFrame(plan_rows)
hist_flow = pd.DataFrame(flow_rows)

# ---------------- 27.1~28.12 중장기 생산계획(월) ----------------
fut_rows = []
for t in range(12, N_MONTHS):
    y, m = 2026 + t // 12, t % 12 + 1
    for ln in LINES:
        for p in range(P):
            fut_rows.append({"년월": f"{y}-{m:02d}", "라인": ln, "제품": PRODUCTS[p],
                             "생산계획(Lot)": int(round(plan_m[(t, ln)][p]))})
fut_plan = pd.DataFrame(fut_rows)

# ---------------- Excel 저장 ----------------
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


def style(ws, widths=None):
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F3A5F")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    for i, col in enumerate(ws.columns, 1):
        w = (widths or {}).get(i) or min(max(len(str(c.value or "")) for c in col) + 4, 34)
        ws.column_dimensions[get_column_letter(i)].width = max(w, 10)


readme_hist = pd.DataFrame({"항목": ["목적", "기간", "대상", "시트1", "시트2", "주의"],
    "내용": ["H23 15/16/17L 층/동간 물동량 예측 PoC - 학습용 이력 데이터",
             "2026-01-01 ~ 2026-10-31 (일별)",
             "라인: 15L/16L/17L, 제품 A~E, Bridge(15-16, 15-17)/Lifter(라인별)",
             "일별_생산계획_WIP : 일자/라인/제품별 생산계획·재공 (long 형식)",
             "일별_물동량 : 일자별 구간·방향·층별 이동 Lot 수 (wide 형식, 14개 컬럼)",
             "※ 본 데이터는 PoC용 DUMMY 데이터이며 실제 값이 아닙니다."]})
with pd.ExcelWriter("data/H23_이력데이터_2026.01-10.xlsx", engine="openpyxl") as w:
    readme_hist.to_excel(w, sheet_name="설명", index=False)
    hist_plan.to_excel(w, sheet_name="일별_생산계획_WIP", index=False)
    hist_flow.to_excel(w, sheet_name="일별_물동량", index=False)
    for n in ("설명", "일별_생산계획_WIP", "일별_물동량"):
        style(w.sheets[n], {1: 14, 2: 80} if n == "설명" else None)

readme_fut = pd.DataFrame({"항목": ["목적", "기간", "형식", "주의"],
    "내용": ["H23 15/16/17L 중장기 월별 제품별 생산계획 (예측 입력값)",
             "2027-01 ~ 2028-12 (월별)",
             "년월 / 라인 / 제품 / 생산계획(Lot) - 계획 변경 시 이 시트 값만 수정 후 대시보드에 업로드",
             "※ 본 데이터는 PoC용 DUMMY 데이터이며 실제 값이 아닙니다."]})
with pd.ExcelWriter("data/H23_중장기생산계획_2027.01-2028.12.xlsx", engine="openpyxl") as w:
    readme_fut.to_excel(w, sheet_name="설명", index=False)
    fut_plan.to_excel(w, sheet_name="월별_생산계획", index=False)
    style(w.sheets["설명"], {1: 14, 2: 90})
    style(w.sheets["월별_생산계획"])

# 이력 요약 출력
hf = hist_flow.copy()
hf["m"] = pd.to_datetime(hf["일자"]).dt.month
mon = hf.groupby("m")[FLOW_COLS].sum()
for s in SEGMENTS:
    cols = [f["col"] for f in FLOWS if f["seg"] == s["id"]]
    util = mon[cols].sum(axis=1) / (pd.Series([calendar.monthrange(2026, m)[1] for m in mon.index], index=mon.index) * 24) / s["cap"]
    print(s["name"], "26년 부하율(%):", (util * 100).round(0).astype(int).tolist())
print("rows:", len(hist_plan), len(hist_flow), len(fut_plan))
