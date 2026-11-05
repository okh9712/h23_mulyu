# -*- coding: utf-8 -*-
"""H23 물동량 예측 PoC - 공통 정의 (라인/제품/구간/CAPA)"""
LINES = ["15L", "16L", "17L"]
PRODUCTS = ["제품A", "제품B", "제품C", "제품D", "제품E"]
FLOORS = ["상부", "하부"]

# MAX CAPA (Lot/Hr) - 기획서 기준. Bridge/Lifter 모두 양방향 합산 기준으로 가정
SEGMENTS = [
    {"id": "B1516", "type": "Bridge", "name": "Bridge 15-16", "cap": 4400},
    {"id": "B1517", "type": "Bridge", "name": "Bridge 15-17", "cap": 1750},
    {"id": "L15", "type": "Lifter", "name": "Lifter 15L", "cap": 2000},
    {"id": "L16", "type": "Lifter", "name": "Lifter 16L", "cap": 2000},
    {"id": "L17", "type": "Lifter", "name": "Lifter 17L", "cap": 2000},
]


def build_flows():
    """14개 예측 대상(물동량 흐름) 정의"""
    flows = []
    for seg, a, b in (("B1516", "15", "16"), ("B1517", "15", "17")):
        sname = f"Bridge {a}-{b}"
        for d_from, d_to in ((a, b), (b, a)):
            for fl in FLOORS:
                flows.append({
                    "col": f"{sname} {d_from}→{d_to} {fl}",
                    "seg": seg, "type": "Bridge", "segname": sname,
                    "dir": f"{d_from}→{d_to}", "floor": fl,
                    "from": f"{d_from}L {fl}", "to": f"{d_to}L {fl}",
                })
    for ln in ("15", "16", "17"):
        sname = f"Lifter {ln}L"
        flows.append({"col": f"{sname} 상승", "seg": f"L{ln}", "type": "Lifter", "segname": sname,
                      "dir": "상승", "floor": "층간", "from": f"{ln}L 하부", "to": f"{ln}L 상부"})
        flows.append({"col": f"{sname} 하강", "seg": f"L{ln}", "type": "Lifter", "segname": sname,
                      "dir": "하강", "floor": "층간", "from": f"{ln}L 상부", "to": f"{ln}L 하부"})
    return flows


FLOWS = build_flows()
FLOW_COLS = [f["col"] for f in FLOWS]
