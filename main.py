from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np, cv2, traceback, base64, math
from pipeline import (analyze_image, find_cross_scale, detect_cracks,
                      draw_cracks, calculate_properties, find_nearest_scol,
                      polyline_length_px, compute_scol_stats, _clamp_d)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

_store = {}   # {orig_b64, results, scale, loadP, manual_d, manual_diam}


@app.get("/")
async def index():
    return HTMLResponse(open("index.html", encoding="utf-8").read())


# ── Analyze ─────────────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    crossSize: float = Form(25),
    loadP:     float = Form(0.98),
    minArea:   float = Form(200),
):
    try:
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "image decode failed"}
        _, buf = cv2.imencode(".jpg", img)
        _store["orig_b64"] = base64.b64encode(buf).decode()
        # preserve any existing manual overrides across re-analyze
        manual_d    = _store.get("manual_d",    {})
        manual_diam = _store.get("manual_diam", {})

        result = analyze_image(img, crossSize, loadP, int(minArea),
                               manual_d, manual_diam)

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = find_cross_scale(gray, crossSize) or float(crossSize) / 200.0
        det, _, _ = detect_cracks(gray, min_area=int(minArea))
        _store.update({"results": det, "scale": scale, "loadP": loadP,
                        "manual_d": manual_d, "manual_diam": manual_diam})
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Redraw with current kept-ids / manual cracks / overrides ────────────────
@app.post("/api/redraw")
async def redraw(data: dict):
    try:
        kept_ids      = set(data.get("kept_crack_ids", []))
        manual_cracks = data.get("manual_cracks", [])
        load_P        = float(data.get("load_P", _store.get("loadP", 0.98)))

        orig_b64 = _store.get("orig_b64", "")
        if not orig_b64:
            return JSONResponse(status_code=400,
                                content={"error": "No original image. Re-upload."})
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(orig_b64), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "image decode failed"}

        results     = _store.get("results", [])
        scale       = _store.get("scale", 0.125)
        manual_d    = _store.get("manual_d",    {})
        manual_diam = _store.get("manual_diam", {})

        manual_by_scol = {}
        for mc in manual_cracks:
            sid = mc.get("scol_id")
            manual_by_scol.setdefault(sid, []).append(mc)

        scol_stats, global_stats = compute_scol_stats(
            results, kept_ids, scale, manual_by_scol, load_P,
            manual_d, manual_diam)

        out = draw_cracks(img, results, kept_ids, scale,
                          manual_cracks=manual_cracks,
                          manual_d_overrides=manual_d,
                          manual_diam_overrides=manual_diam)
        _, buf = cv2.imencode(".jpg", out)
        return {
            "image":         base64.b64encode(buf).decode(),
            "crack_lengths": global_stats.get("lengths", []),
            "count":         global_stats.get("count", 0),
            "mean_length":   global_stats.get("mean_l", 0.0),
            "std_length":    global_stats.get("std_l",  0.0),
            "sem":           global_stats.get("sem_l",  0.0),
            "properties":    global_stats.get("props",  {}),
            "scol_stats":    scol_stats,
            "global_stats":  global_stats,
            "mc":            global_stats.get("mc", {}),
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Manual crack (polyline) ──────────────────────────────────────────────────
@app.post("/api/finish_manual_crack")
async def finish_manual_crack(data: dict):
    try:
        points  = data.get("points", [])
        scale   = _store.get("scale", 0.125)
        results = _store.get("results", [])
        if len(points) < 2:
            return {"length_um": 0, "scol_id": None}
        length_um = round(polyline_length_px(points) * scale, 3)
        mid = points[len(points) // 2]
        scol_id = find_nearest_scol(mid[0], mid[1], results) if results else None
        center  = next(([r["center_x_px"], r["center_y_px"]] for r in results
                        if r["scol_id"] == scol_id), None)
        return {"length_um": length_um, "scol_id": scol_id, "nearest_center": center}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Set manual diagonal d for a scol ────────────────────────────────────────
@app.post("/api/set_manual_d")
async def set_manual_d(data: dict):
    """
    data: {scol_id: int, pts: [[x1,y1],[x2,y2]]}
    Calculates d_um from pts, stores override, returns d_um.
    Pass scol_id=null to clear all.
    """
    try:
        scol_id = data.get("scol_id")
        pts     = data.get("pts")
        scale   = _store.get("scale", 0.125)
        manual_d = _store.setdefault("manual_d", {})

        if scol_id is None:
            manual_d.clear()
            return {"ok": True, "cleared": True}

        if pts is None:
            manual_d.pop(int(scol_id), None)
            return {"ok": True, "removed": True}

        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        d_um = round(_clamp_d(math.hypot(dx, dy) * scale), 3)
        manual_d[int(scol_id)] = {"d_um": d_um, "pts": pts}
        return {"ok": True, "d_um": d_um, "scol_id": scol_id}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Set manual full scol diameter ────────────────────────────────────────────
@app.post("/api/set_manual_diam")
async def set_manual_diam(data: dict):
    """
    data: {scol_id: int, pts: [[x1,y1],[x2,y2]]}
    """
    try:
        scol_id  = data.get("scol_id")
        pts      = data.get("pts")
        scale    = _store.get("scale", 0.125)
        manual_diam = _store.setdefault("manual_diam", {})

        if scol_id is None:
            manual_diam.clear()
            return {"ok": True, "cleared": True}

        if pts is None:
            manual_diam.pop(int(scol_id), None)
            return {"ok": True, "removed": True}

        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        diam_um = round(_clamp_d(math.hypot(dx, dy) * scale), 3)
        manual_diam[int(scol_id)] = {"diam_um": diam_um, "pts": pts}
        return {"ok": True, "diam_um": diam_um, "scol_id": scol_id}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
