#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Lokal Model Test Betigi

Egitilmis modelleri test videolari uzerinde calistirarak
her video icin ayri bir results.json dosyasi uretir.

Kullanim:
  python tools/test_models.py --video test_videos/video1.mp4
  python tools/test_models.py --video_dir test_videos/
  python tools/test_models.py --video test_videos/video1.mp4 --visualize

Ciktilar:
  output/<video_adi>/results.json    — Yarisma formatinda JSON cikti
  output/<video_adi>/preview.mp4     — Gorsel annotasyonlu video (--visualize ile)
"""

import os
import sys
import json
import time
import argparse
import logging
import gc
from pathlib import Path

# Proje kokunu sys.path'e ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test")

# ==============================================================================
# Sabitler
# ==============================================================================
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Sinif ID -> Isim esleme (class_config.yaml ile uyumlu)
VEHICLE_CLASS_IDS = {
    0: "sedan", 1: "suv", 2: "hatchback", 3: "pickup",
    4: "minibus", 5: "panelvan", 6: "kamyon",
}
PLATE_CLASS_ID = 20

# Gecerli renk isimleri (kilavuzdan)
VALID_COLORS = [
    "beyaz", "siyah", "gri", "kirmizi", "mavi",
    "sari", "yesil", "turuncu", "kahverengi",
]


def clean_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(1, box1[2] - box1[0]) * max(1, box1[3] - box1[1])
    area2 = max(1, box2[2] - box2[0]) * max(1, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def process_video(video_path: str, visualize: bool = False):
    """Tek bir video uzerinde tam kaskat cikarim yapar."""
    from ultralytics import YOLO
    import torch

    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video bulunamadi: {video_path}")
        return

    video_name = video_path.stem
    out_dir = OUTPUT_DIR / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"VIDEO ANALIZI: {video_path.name}")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Cihaz: {device}")

    # ----- Model Yukleme -----
    det_model_path = MODELS_DIR / "global_yolo.pt"
    cls_model_path = MODELS_DIR / "color_yolo.pt"

    if not det_model_path.exists():
        logger.error(f"Detection model bulunamadi: {det_model_path}")
        return

    logger.info(f"Detection model: {det_model_path}")
    det_model = YOLO(str(det_model_path))

    color_model = None
    if cls_model_path.exists():
        logger.info(f"Renk modeli: {cls_model_path}")
        color_model = YOLO(str(cls_model_path))
    else:
        logger.warning("Renk modeli bulunamadi, HSV fallback kullanilacak.")

    cabin_model = None
    cab_model_path = MODELS_DIR / "cabin_yolo.pt"
    if cab_model_path.exists():
        logger.info(f"Kabin modeli: {cab_model_path}")
        cabin_model = YOLO(str(cab_model_path))

    # ----- Modül 4: Kinematik Yörünge Takipçisi -----
    from src.module4_trajectory.tracker import TrajectoryTracker
    traj_tracker = TrajectoryTracker(window_seconds=5.0)
    person_model = YOLO("yolo11n.pt") # Yolcu ROI taranması için

    # ----- Fast-Plate-OCR / PlateReader -----
    from src.module3_ocr.plate_reader import PlateReader
    plate_reader = PlateReader(ocr_backend="fast_plate_ocr")

    # ----- Video Ac -----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Video acilamadi: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    logger.info(f"  Cozunurluk: {width}x{height} @ {fps:.1f} FPS")
    logger.info(f"  Sure: {duration:.1f} sn ({total_frames} kare)")

    # ----- Visualize icin VideoWriter -----
    writer = None
    if visualize:
        preview_path = out_dir / "preview.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (width, height))
        logger.info(f"  Gorsel cikti: {preview_path}")

    # ----- Cikarim Dongusu -----
    frame_skip = 2  # Her 2. kare islenir (hizli araclari kacirmamak icin)
    frame_count = 0
    processed = 0
    start_time = time.time()

    tracks = []          # MOT Takip havuzu
    all_detections = []  # Yol guvenligi tespitleri

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            if writer:
                writer.write(frame)
            continue

        try:
            # Dusuk esik (conf=0.08) ile hizli/blurlu sedan araclari dahi yakala
            results = det_model.predict(
                frame, conf=0.08, iou=0.45, imgsz=640, verbose=False
            )

            if results and len(results) > 0:
                pred = results[0]
                if pred.boxes is not None:
                    # 1. Adim: Karedeki kutulari ayristir
                    v_boxes = [] # araclar
                    p_boxes = [] # plakalar
                    
                    for box in pred.boxes:
                        cid = int(box.cls[0])
                        cf = float(box.conf[0])
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        
                        if cid in VEHICLE_CLASS_IDS:
                            v_boxes.append((cid, cf, coords))
                        elif cid == PLATE_CLASS_ID:
                            p_boxes.append((cf, coords))

                    # 2. Adim: Her arac kutusu icin MOT esleme ve ozellik toplama
                    for cid, cf, coords in v_boxes:
                        x1, y1, x2, y2 = coords
                        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
                        vtype = VEHICLE_CLASS_IDS[cid]

                        # Aktif track bul
                        best_track = None
                        max_iou = 0.0
                        for t in tracks:
                            if frame_count - t["last_seen"] > 45:
                                continue
                            iou = compute_iou(coords, t["bbox"])
                            if iou > max_iou:
                                max_iou = iou
                                best_track = t

                        if best_track is not None and max_iou > 0.25:
                            track = best_track
                            track["bbox"] = coords
                            track["last_seen"] = frame_count
                        else:
                            track = {
                                "id": len(tracks) + 1,
                                "bbox": coords,
                                "last_seen": frame_count,
                                "type_votes": {},
                                "color_votes": {},
                                "plate_texts": [],
                                "aspect_ratios": [],
                            }
                            tracks.append(track)

                        track["type_votes"].setdefault(vtype, []).append(cf)
                        track["aspect_ratios"].append(box_w / box_h)

                        # --- RENK ---
                        vx1, vy1 = max(0, x1), max(0, y1)
                        vx2, vy2 = min(width, x2), min(height, y2)
                        vehicle_crop = frame[vy1:vy2, vx1:vx2]

                        if vehicle_crop.size > 0:
                            detected_c = None
                            c_conf = 0.5
                            if color_model is not None:
                                cls_res = color_model.predict(vehicle_crop, imgsz=224, verbose=False)
                                if cls_res and len(cls_res) > 0 and cls_res[0].probs is not None:
                                    tidx = int(cls_res[0].probs.top1)
                                    tcf = float(cls_res[0].probs.top1conf)
                                    raw_c = cls_res[0].names[tidx].lower()
                                    en_tr = {
                                        "black": "siyah", "white": "beyaz", "grey": "gri", "gray": "gri", 
                                        "silver": "gri", "red": "kirmizi", "blue": "mavi", "yellow": "sari", 
                                        "gold": "sari", "green": "yesil", "orange": "turuncu", "brown": "kahverengi", 
                                        "tan": "kahverengi", "beige": "kahverengi"
                                    }
                                    mapped_c = en_tr.get(raw_c, raw_c)
                                    if mapped_c in VALID_COLORS:
                                        detected_c = mapped_c
                                        c_conf = tcf

                            # HSV Fallback (Model null dönerse veya algılayamazsa)
                            if detected_c is None:
                                hsv_c = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
                                sm, vm = np.mean(hsv_c[:, :, 1]), np.mean(hsv_c[:, :, 2])
                                hm = np.mean(hsv_c[:, :, 0])
                                if sm < 40:
                                    detected_c = "beyaz" if vm > 190 else ("siyah" if vm < 70 else "gri")
                                else:
                                    detected_c = "kirmizi" if (hm < 10 or hm > 170) else ("sari" if 10 <= hm < 35 else ("yesil" if 35 <= hm < 85 else ("mavi" if 85 <= hm < 135 else "turuncu")))

                            if detected_c in VALID_COLORS:
                                hsv_crop = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
                                s_mean = float(np.mean(hsv_crop[:, :, 1])) / 255.0
                                v_mean = float(np.mean(hsv_crop[:, :, 2])) / 255.0
                                lw = max(0.15, s_mean * v_mean) if detected_c != "beyaz" else 1.0
                                track["color_votes"].setdefault(detected_c, []).append(c_conf * lw)

                        # --- KABİN / SÜRÜCÜ EYLEMİ ANALİZİ ---
                        if cabin_model is not None:
                            vh = max(1, y2 - y1)
                            wy1, wy2 = y1 + int(vh * 0.15), y1 + int(vh * 0.65)
                            w_crop = frame[max(0, wy1):min(height, wy2), max(0, x1):min(width, x2)]
                            if w_crop.size > 0:
                                cab_res = cabin_model.predict(w_crop, conf=0.25, verbose=False)
                                if cab_res and len(cab_res) > 0 and cab_res[0].boxes:
                                    act_names = {
                                        7: "arkaya_bakma", 8: "esneme", 9: "sigara_icme",
                                        10: "su_icme", 11: "telefonla_konusma", 13: "etrafa_bakinma",
                                        14: "emniyet_kemeri_ihlali",
                                    }
                                    for cbox in cab_res[0].boxes:
                                        acid = int(cbox.cls[0])
                                        aconf = float(cbox.conf[0])
                                        if acid in act_names:
                                            aname = act_names[acid]
                                            all_detections.append({
                                                "zaman_saniye": round(frame_count / fps, 2),
                                                "kategori": "sofor_eylemi",
                                                "etiket": aname,
                                                "confidence_score": round(aconf, 4),
                                            })

                                # Geometrik Yolcu Sabit ROI Tarama (FTR §Tablo2)
                                p_res = person_model.predict(w_crop, conf=0.35, verbose=False)
                                if p_res and len(p_res) > 0 and p_res[0].boxes:
                                    ch, cw = w_crop.shape[:2]
                                    for pbox in p_res[0].boxes:
                                        if int(pbox.cls[0]) == 0: # person
                                            bx1, by1, bx2, by2 = map(int, pbox.xyxy[0])
                                            cx, cy = (bx1+bx2)/(2.0*cw), (by1+by2)/(2.0*ch)
                                            s_lbl = "on_koltuk" if cx > 0.48 else ("arka_koltuk_1" if cy < 0.6 else "arka_koltuk_2")
                                            all_detections.append({
                                                "zaman_saniye": round(frame_count / fps, 2),
                                                "kategori": "yolcular",
                                                "etiket": s_lbl,
                                                "confidence_score": round(float(pbox.conf[0]), 4),
                                            })

                        # --- MODÜL 4: KİNEMATİK SLALOM VE YÖRÜNGE ANALİZİ ---
                        cur_t = frame_count / fps
                        traj_tracker.update(vehicle_bbox=(x1, y1, x2, y2), frame_time=cur_t)
                        if writer:
                            traj_tracker.draw_trajectory(frame, cur_t)

                        # Gorsel Annotasyon
                        if writer:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            top_plate = track["plate_texts"][-1][0] if track["plate_texts"] else ""
                            lbl = f"#{track['id']} {vtype} {top_plate}"
                            cv2.putText(frame, lbl, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 3. Adim: Plaka Okuma (Eskisi gibi bağımsız kesim + En yakın araca eşleme)
                    for pcf, pcoords in p_boxes:
                        px1, py1, px2, py2 = pcoords
                        plate_crop = frame[max(0, py1):min(height, py2), max(0, px1):min(width, px2)]
                        if plate_crop.size > 0:
                            ptext = plate_reader.read(plate_crop)
                            if ptext and len(ptext) >= 2:
                                # En yakin aktif track bul
                                pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                                best_t = None
                                min_d = float('inf')
                                for t in tracks:
                                    tx1, ty1, tx2, ty2 = t["bbox"]
                                    tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
                                    d = (pcx - tcx)**2 + (pcy - tcy)**2
                                    if d < min_d:
                                        min_d = d
                                        best_t = t
                                
                                if best_t is not None:
                                    best_t["plate_texts"].append((ptext, pcf))
                                else:
                                    # Track yoksa varsayilan ana track olustur
                                    dummy_t = {
                                        "id": 1, "bbox": [0, 0, width, height],
                                        "last_seen": frame_count, "type_votes": {},
                                        "color_votes": {}, "plate_texts": [(ptext, pcf)],
                                        "aspect_ratios": []
                                    }
                                    tracks.append(dummy_t)

                                if writer:
                                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                                    cv2.putText(frame, ptext, (px1, py2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if writer:
                writer.write(frame)

            processed += 1

            if processed % 50 == 0:
                pct = frame_count / total_frames * 100 if total_frames > 0 else 0
                logger.info(f"  Ilerleme: {pct:.0f}% ({frame_count}/{total_frames})")

        except Exception as e:
            logger.warning(f"  Kare {frame_count} hatasi: {e}")
            if writer:
                writer.write(frame)
            continue

    cap.release()
    if writer:
        writer.release()

    elapsed = time.time() - start_time
    logger.info(f"  Cikarim tamamlandi: {processed} kare, {elapsed:.1f} sn")

    # ========== SONUCLARI TOPLA (Video Geneli Oylama Havuzu) ==========
    from src.postprocessor import PostProcessor
    postprocessor = PostProcessor()

    global_type_votes = {}
    global_color_votes = {}
    global_plate_texts = []
    all_ars = []

    for t in tracks:
        for k, v in t["type_votes"].items():
            global_type_votes.setdefault(k, []).extend(v)
        for k, v in t["color_votes"].items():
            global_color_votes.setdefault(k, []).extend(v)
        global_plate_texts.extend(t["plate_texts"])
        all_ars.extend(t["aspect_ratios"])

    avg_ar = float(np.mean(all_ars)) if all_ars else 1.5
    main_vehicle_info = postprocessor.aggregate_vehicle_info(
        global_type_votes, global_color_votes, global_plate_texts, bbox_aspect_ratio=avg_ar
    )

    # Kılavuza 1-1 uyum için ek keyleri temizle
    main_vehicle_info.pop("track_id", None)

    best_type = main_vehicle_info.get("tip", "sedan")
    best_color = main_vehicle_info.get("renk", "")
    best_plate = main_vehicle_info.get("plaka", "tespit edilemedi")
    best_conf = main_vehicle_info.get("confidence_score", 0.0)

    # Slalom ihlallerini ekle
    all_detections.extend(traj_tracker.get_slalom_events())

    # ========== JSON CIKTI (Kılavuza birebir) ==========
    output = {
        "video_id": video_path.name,
        "arac_bilgisi": main_vehicle_info,
        "tespitler": all_detections,
    }

    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ========== SONUC RAPORU ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("SONUCLAR")
    logger.info("=" * 60)
    logger.info(f"  Arac Tipi  : {best_type} (conf: {best_conf:.4f})")
    logger.info(f"  Arac Rengi : {best_color}")
    logger.info(f"  Plaka      : {best_plate}")
    logger.info(f"  JSON       : {json_path}")

    logger.info("=" * 60)
    logger.info("")

    clean_memory()
    return output


# ==============================================================================
# ANA GIRIS
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TEKNOFEST 2026 ANKAAI — Lokal Model Test Betigi"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Tek video dosyasi yolu")
    group.add_argument("--video_dir", type=str, help="Birden fazla video iceren dizin")
    parser.add_argument("--visualize", action="store_true",
                        help="Gorsel annotasyonlu preview.mp4 uret")
    args = parser.parse_args()

    # Modelleri kontrol et
    if not (MODELS_DIR / "global_yolo.pt").exists():
        logger.error(f"Detection model bulunamadi: {MODELS_DIR / 'global_yolo.pt'}")
        sys.exit(1)

    if args.video:
        process_video(args.video, visualize=args.visualize)
    elif args.video_dir:
        video_dir = Path(args.video_dir)
        videos = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")))
        if not videos:
            logger.error(f"Dizinde video bulunamadi: {video_dir}")
            sys.exit(1)

        logger.info(f"{len(videos)} video bulundu. Sirasi ile isleniyor...")
        for vpath in videos:
            process_video(str(vpath), visualize=args.visualize)

    logger.info("Tum islemler tamamlandi!")
