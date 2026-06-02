import math, cv2
import numpy as np
import base64
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

# ── Detection defaults ──────────────────────────────────────────────────────
MIN_AREA       = 100    # was 300 — lower = more regions detected
BLUR_KSIZE     = 5
ADAPTIVE_BLOCK = 40     # was 51 — smaller block = more local contrast
ADAPTIVE_C     = 1      # was 2
SAMPLE_ANGLES  = 360

# ── Physical constants ───────────────────────────────────────────────────────
E_DEFAULT  = 10e9   # Pa
V_DEFAULT  = 0.3
D_MIN_UM   = 1.0
D_MAX_UM   = 600.0


def find_cross_scale(gray, cross_real_um=50.0):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is None or len(lines) < 2:
        return None
    horizontal, vertical = [], []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        if angle < 10 or angle > 170:
            horizontal.append(rho)
        elif 80 < angle < 100:
            vertical.append(rho)
    if not horizontal or not vertical:
        return None
    cross_px = max(abs(max(horizontal) - min(horizontal)),
                   abs(max(vertical) - min(vertical)))
    return float(cross_real_um / cross_px) if cross_px >= 20 else None


def detect_cracks(gray, min_area=MIN_AREA, blur_ksize=BLUR_KSIZE,
                  adaptive_block=ADAPTIVE_BLOCK, adaptive_C=ADAPTIVE_C,
                  sample_angles=SAMPLE_ANGLES):
    h, w = gray.shape
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    gray_blur = cv2.GaussianBlur(gray, (k, k), 0)
    block = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
    thr = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, block, adaptive_C)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN,  kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = thr.astype(bool)
    mask = remove_small_objects(mask, min_size=min_area)
    lab  = label(mask)
    regions = regionprops(lab)
    if len(regions) == 0:
        return [], mask, regions

    def ray_radius(mask_bool, cx, cy, theta, rmax):
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        last_inside = 0
        for r in range(0, rmax):
            x = int(round(cx + r * cos_t))
            y = int(round(cy + r * sin_t))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if mask_bool[y, x]:
                last_inside = r
            elif r > 0:
                return last_inside
        return last_inside

    rmax   = int(math.hypot(w, h)) // 2
    thetas = np.linspace(0, 2 * np.pi, sample_angles, endpoint=False)
    all_results = []

    for i, reg in enumerate(regions, start=1):
        area = reg.area
        cy_c, cx_c = reg.centroid
        cx_c, cy_c = float(cx_c), float(cy_c)

        radii = np.array([ray_radius(mask, cx_c, cy_c, t, rmax) for t in thetas],
                         dtype=np.float32)

        # ── Segment detection: slightly relaxed threshold ──────────────────
        r_threshold = float(max(4.0, np.mean(radii) * 0.18))
        segments, start_idx = [], None
        for j, rv in enumerate(radii):
            if rv > r_threshold:
                if start_idx is None:
                    start_idx = j
            else:
                if start_idx is not None:
                    if j - start_idx >= 4:   # was 5
                        segments.append((start_idx, j - 1))
                    start_idx = None
        if start_idx is not None and len(radii) - start_idx >= 4:
            segments.append((start_idx, len(radii) - 1))

        rays_lengths = [float(np.max(radii[s:e + 1])) for s, e in segments]

        # ── Robust indent half-diagonal estimate ───────────────────────────
        # The indent body = compact bright core at the center.
        # Its radius ≈ lower quartile of ray lengths (cracks go far, core stays short).
        nonzero = radii[radii > 2]
        if len(nonzero) > 10:
            d_half_px = float(np.percentile(nonzero, 15))
        else:
            d_half_px = float(np.median(radii[radii > 0])) if np.any(radii > 0) else 5.0
        d_half_px = max(d_half_px, 2.0)

        all_results.append({
            "scol_id":          i,
            "area_px":          float(area),
            "center_x_px":      cx_c,
            "center_y_px":      cy_c,
            "rays_count":       len(segments),
            "rays_lengths_px":  rays_lengths,
            "d_half_px":        d_half_px,      # half-diagonal of indent
            "d_full_px":        d_half_px * 2,  # full diagonal
            "radius_min_px":    float(np.min(radii)),
            "radius_median_px": float(np.median(radii)),
            "radius_max_px":    float(np.max(radii)),
            "radius_mean_px":   float(np.mean(radii)),
            "segments":         segments,
            "radii":            radii,
            "thetas":           thetas,
        })

    return all_results, mask, regions


def _clamp_d(d_um):
    return max(D_MIN_UM, min(D_MAX_UM, float(d_um)))


def calculate_properties(L_mean_um, d_um, P=0.98, E=E_DEFAULT, v=V_DEFAULT):
    L_m = float(L_mean_um) * 1e-6
    d_m = float(d_um)      * 1e-6
    if L_m <= 0 or d_m <= 0:
        return {"H_GPa": 0.0, "K_MN": 0.0, "G_J": 0.0, "Y_J": 0.0, "h_um": 0.0}
    H  = (1.854 * float(P)) / (d_m ** 2)
    h  = d_m / 7.0
    K  = 0.016 * math.sqrt(E / H) * (float(P) / (L_m ** 1.5))
    dn = 1.0 + v + 2.0 * (1.0 - v) * H * L_m ** 2 / float(P)
    G  = (0.627 * H**2 * h * (1.0 - v)) / (E * dn**2)
    Y  = K**2 / (2.0 * E)
    return {
        "H_GPa": round(H / 1e9, 3),
        "K_MN":  round(K / 1e6, 3),
        "G_J":   round(float(G), 9),
        "Y_J":   round(float(Y), 9),
        "h_um":  round(float(h * 1e6), 3)
    }


def monte_carlo_uncertainty(L_mean_um, L_std_um, d_um, d_std_um,
                            P=0.98, E=E_DEFAULT, v=V_DEFAULT, N=10000):
    rng = np.random.default_rng(42)
    L_s = rng.normal(L_mean_um, max(float(L_std_um), 0.5), N)
    d_s = rng.normal(d_um,      max(float(d_std_um), 0.1), N)
    H_a, K_a, G_a, Y_a = [], [], [], []
    for Lv, dv in zip(L_s, d_s):
        Lv, dv = abs(Lv), _clamp_d(abs(dv))
        if Lv < 1.0 or dv < D_MIN_UM:
            continue
        p = calculate_properties(Lv, dv, P, E, v)
        H_a.append(p["H_GPa"]); K_a.append(p["K_MN"])
        G_a.append(p["G_J"]);   Y_a.append(p["Y_J"])

    def ci(arr):
        if len(arr) < 10:
            return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
        a = np.array(arr)
        return {"mean":  round(float(np.mean(a)), 3),
                "ci_lo": round(float(np.percentile(a, 2.5)), 3),
                "ci_hi": round(float(np.percentile(a, 97.5)), 3)}

    return {"H": ci(H_a), "K": ci(K_a), "G": ci(G_a), "Y": ci(Y_a),
            "n_valid": len(H_a)}


def find_nearest_scol(cx, cy, results):
    best_id, best_dist = None, float("inf")
    for res in results:
        d = math.hypot(cx - res["center_x_px"], cy - res["center_y_px"])
        if d < best_dist:
            best_dist, best_id = d, res["scol_id"]
    return best_id


def polyline_length_px(points):
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i-1][0],
                            points[i][1] - points[i-1][1])
    return total


def draw_cracks(original_image, results, kept_crack_ids, scale,
                min_length_um=15, manual_cracks=None,
                manual_d_overrides=None, manual_diam_overrides=None):
    """
    manual_d_overrides:    {scol_id: d_um}   — user-measured indent diagonal
    manual_diam_overrides: {scol_id: diam_um} — user-measured full scol circle
    """
    out = original_image.copy()
    COLORS_BGR = [
        (255, 166, 88), (114, 123, 255), (80,  185,  63), (65,  179, 227),
        (203,  99, 168), (88, 214, 188), (60,  166, 251), (163, 224,  57),
    ]

    for res in results:
        sid = res['scol_id']
        cx, cy = res['center_x_px'], res['center_y_px']
        color  = COLORS_BGR[(sid - 1) % len(COLORS_BGR)]

        for seg_idx, (seg_start, seg_end) in enumerate(res['segments']):
            crack_key = f"{sid}_{seg_idx}"
            if crack_key not in kept_crack_ids:
                continue
            seg_radii = res['radii'][seg_start:seg_end + 1]
            max_loc   = int(np.argmax(seg_radii))
            max_theta = float(res['thetas'][seg_start + max_loc])
            max_r     = float(seg_radii[max_loc])
            length_um = round(max_r * scale, 1)
            if length_um < min_length_um:
                continue
            x2 = int(cx + max_r * math.cos(max_theta))
            y2 = int(cy + max_r * math.sin(max_theta))
            cv2.line(out, (int(cx), int(cy)), (x2, y2), color, 2)
            cv2.putText(out, f"{length_um:.1f}",
                        (x2 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.circle(out, (int(cx), int(cy)), 4, color, -1)
        cv2.putText(out, f"S{sid}", (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw manual d measurement line if present
        if manual_d_overrides and sid in manual_d_overrides:
            md = manual_d_overrides[sid]
            if "pts" in md:
                pts = md["pts"]
                cv2.line(out, tuple(map(int, pts[0])), tuple(map(int, pts[1])),
                         (0, 255, 255), 2)
                cv2.putText(out, f"d={md['d_um']:.1f}µm",
                            (int(pts[0][0]) + 4, int(pts[0][1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

        # Draw manual diameter measurement line if present
        if manual_diam_overrides and sid in manual_diam_overrides:
            md2 = manual_diam_overrides[sid]
            if "pts" in md2:
                pts = md2["pts"]
                cv2.line(out, tuple(map(int, pts[0])), tuple(map(int, pts[1])),
                         (255, 128, 0), 2)
                cv2.putText(out, f"D={md2['diam_um']:.1f}µm",
                            (int(pts[0][0]) + 4, int(pts[0][1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 128, 0), 1)

    # Manual cracks (polylines)
    if manual_cracks:
        for mc in manual_cracks:
            pts = mc.get("points", [])
            if len(pts) < 2:
                continue
            sid   = mc.get("scol_id")
            color = COLORS_BGR[(sid - 1) % len(COLORS_BGR)] if sid else (0, 180, 255)
            np_pts = np.array(pts, dtype=np.int32)
            cv2.polylines(out, [np_pts], False, color, 2)
            mid = np_pts[len(np_pts) // 2]
            cv2.putText(out, f"{mc['length_um']:.1f}*",
                        (int(mid[0]) + 4, int(mid[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def compute_scol_stats(results, kept_crack_ids, scale,
                       manual_cracks_by_scol, P,
                       manual_d_overrides=None,
                       manual_diam_overrides=None):
    """
    manual_d_overrides:    {scol_id: {"d_um": float, "pts": [[x1,y1],[x2,y2]]}}
    manual_diam_overrides: {scol_id: {"diam_um": float, "pts": ...}}
    """
    E, v = E_DEFAULT, V_DEFAULT
    manual_d_overrides    = manual_d_overrides    or {}
    manual_diam_overrides = manual_diam_overrides or {}
    scol_stats, all_lengths = [], []

    for res in results:
        sid     = res['scol_id']
        lengths = []

        for seg_idx, (seg_start, seg_end) in enumerate(res['segments']):
            crack_key = f"{sid}_{seg_idx}"
            if crack_key not in kept_crack_ids:
                continue
            seg_radii = res['radii'][seg_start:seg_end + 1]
            max_r = float(np.max(seg_radii))
            l_um  = round(max_r * scale, 2)
            if l_um >= 15:
                lengths.append(l_um)

        for mc in manual_cracks_by_scol.get(sid, []):
            lengths.append(float(mc['length_um']))

        if not lengths:
            continue

        mean_l = float(np.mean(lengths))
        std_l  = float(np.std(lengths))  if len(lengths) > 1 else 0.0
        sem_l  = std_l / math.sqrt(len(lengths)) if len(lengths) > 1 else 0.0

        # ── Determine d (indent diagonal) ──────────────────────────────────
        if sid in manual_d_overrides:
            d_um   = _clamp_d(manual_d_overrides[sid]["d_um"])
            d_std  = max(2.0 * scale, 0.1)
            d_src  = "manual"
        else:
            auto_d = res['d_half_px'] * 2.0 * scale
            d_um   = _clamp_d(auto_d)
            d_std  = max(2.0 * scale, 0.1)
            d_src  = "auto"

        # ── Determine diam (full scol circle) ──────────────────────────────
        if sid in manual_diam_overrides:
            diam_um = _clamp_d(manual_diam_overrides[sid]["diam_um"])
            diam_src = "manual"
        else:
            diam_um  = res['radius_max_px'] * 2.0 * scale
            diam_src = "auto"

        props     = calculate_properties(mean_l, d_um, P, E, v)
        mc_result = monte_carlo_uncertainty(mean_l, std_l, d_um, d_std, P, E, v)

        all_lengths.extend(lengths)
        scol_stats.append({
            "scol_id": sid,
            "count":   len(lengths),
            "lengths": lengths,
            "mean_l":  round(mean_l, 3),
            "std_l":   round(std_l,  3),
            "sem_l":   round(sem_l,  3),
            "d_um":    round(d_um,   3),
            "d_src":   d_src,
            "diam_um": round(diam_um, 3),
            "diam_src":diam_src,
            "props":   props,
            "mc":      mc_result,
        })

    global_stats = {}
    if all_lengths:
        gm  = float(np.mean(all_lengths))
        gs  = float(np.std(all_lengths)) if len(all_lengths) > 1 else 0.0
        gsem= gs / math.sqrt(len(all_lengths)) if len(all_lengths) > 1 else 0.0
        d_um_g  = float(np.mean([s['d_um'] for s in scol_stats])) if scol_stats else 27.2
        d_std_g = max(2.0 * scale, 0.1)
        global_stats = {
            "count":   len(all_lengths),
            "lengths": all_lengths,
            "mean_l":  round(gm,    3),
            "std_l":   round(gs,    3),
            "sem_l":   round(gsem,  3),
            "d_um":    round(d_um_g, 3),
            "props":   calculate_properties(gm, d_um_g, P, E, v),
            "mc":      monte_carlo_uncertainty(gm, gs, d_um_g, d_std_g, P, E, v),
        }

    return scol_stats, global_stats


def analyze_image(image, cross_size_um, load_P, min_area=MIN_AREA,
                  manual_d_overrides=None, manual_diam_overrides=None):
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cross_um  = float(cross_size_um) if cross_size_um else 25.0
    P         = float(load_P) if load_P else 0.98
    scale     = find_cross_scale(gray, cross_um) or float(cross_um) / 200.0
    results, mask, regions = detect_cracks(gray, min_area=int(min_area))

    empty = {"crack_lengths":[],"cracks_by_scol":[],"count":0,"mean_length":0.0,
             "std_length":0.0,"damage_diameter":0.0,"inner_diameter":0.0,"scale":scale,
             "image":"","properties":{},"scol_stats":[],"global_stats":{},"mc":{}}
    if not results:
        return empty

    cracks_by_scol, all_crack_ids, all_lengths_um = [], set(), []
    for res in results:
        scol_cracks = []
        for seg_idx, (seg_start, seg_end) in enumerate(res['segments']):
            seg_radii = res['radii'][seg_start:seg_end + 1]
            max_r     = float(np.max(seg_radii))
            l_um      = round(max_r * scale, 3)
            if l_um < 15:
                continue
            crack_key = f"{res['scol_id']}_{seg_idx}"
            scol_cracks.append({"id": crack_key, "length_um": l_um, "seg_idx": seg_idx})
            all_crack_ids.add(crack_key)
            all_lengths_um.append(l_um)
        cracks_by_scol.append({
            "scol_id":  res['scol_id'],
            "center_x": res['center_x_px'],
            "center_y": res['center_y_px'],
            "d_um_auto": round(res['d_half_px'] * 2.0 * scale, 3),
            "diam_um_auto": round(res['radius_max_px'] * 2.0 * scale, 3),
            "cracks":   scol_cracks
        })

    mean_length = round(float(np.mean(all_lengths_um)), 3) if all_lengths_um else 0.0
    std_length  = round(float(np.std(all_lengths_um)),  3) if len(all_lengths_um) > 1 else 0.0

    scol_stats, global_stats = compute_scol_stats(
        results, all_crack_ids, scale, {}, P,
        manual_d_overrides, manual_diam_overrides)

    d_um_first = (manual_d_overrides or {}).get(results[0]['scol_id'], {}).get("d_um") \
                 if results else None
    if d_um_first is None:
        d_um_first = _clamp_d(results[0]['d_half_px'] * 2.0 * scale) if results else 27.2

    out = draw_cracks(image, results, all_crack_ids, scale,
                      manual_d_overrides=manual_d_overrides,
                      manual_diam_overrides=manual_diam_overrides)
    _, buf = cv2.imencode(".jpg", out)

    return {
        "crack_lengths":   all_lengths_um,
        "cracks_by_scol":  cracks_by_scol,
        "count":           len(all_lengths_um),
        "mean_length":     mean_length,
        "std_length":      std_length,
        "damage_diameter": round(d_um_first * 3, 3),
        "inner_diameter":  round(d_um_first,     3),
        "scale":           scale,
        "image":           base64.b64encode(buf).decode(),
        "properties":      global_stats.get("props", {}),
        "scol_stats":      scol_stats,
        "global_stats":    global_stats,
        "mc":              global_stats.get("mc", {}),
    }
