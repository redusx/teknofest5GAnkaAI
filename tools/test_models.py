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
    frame_skip = 3  # Her 3. kare islenir
    frame_count = 0
    processed = 0
    start_time = time.time()

    # Birikimli veriler
    type_votes = {}      # {"sedan": [0.9, 0.85, ...], ...}
    color_votes = {}     # {"beyaz": [weighted_score, ...], ...}
    plate_texts = []     # [("34ABC123", 0.88), ...]
    aspect_ratios = []   # Bbox w/h oranlari (SUV vs Sedan evristigi icin)
    all_detections = []  # Tespitler listesi

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            if writer:
                writer.write(frame)
            continue

        frame_time = round(frame_count / fps, 2)

        try:
            # ========== MODUL 1: DETECTION ==========
            results = det_model.predict(
                frame, conf=0.25, iou=0.45, imgsz=640, verbose=False
            )

            if results and len(results) > 0:
                pred = results[0]
                if pred.boxes is not None:
                    for box in pred.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)

                        # --- Arac Tipi ---
                        if cls_id in VEHICLE_CLASS_IDS:
                            vtype = VEHICLE_CLASS_IDS[cls_id]
                            type_votes.setdefault(vtype, []).append(conf)
                            aspect_ratios.append(box_w / box_h)

                            # --- RENK TESPITI (Parlaklik Agirlikli) ---
                            vx1, vy1 = max(0, x1), max(0, y1)
                            vx2, vy2 = min(width, x2), min(height, y2)
                            vehicle_crop = frame[vy1:vy2, vx1:vx2]

                            if vehicle_crop.size > 0 and color_model is not None:
                                cls_results = color_model.predict(
                                    vehicle_crop, imgsz=224, verbose=False
                                )
                                if cls_results and len(cls_results) > 0:
                                    cls_pred = cls_results[0]
                                    if cls_pred.probs is not None:
                                        top1_idx = int(cls_pred.probs.top1)
                                        top1_conf = float(cls_pred.probs.top1conf)
                                        color_name = cls_pred.names[top1_idx]
                                        if color_name in VALID_COLORS:
                                            # Luminance weighting (V ve S katsayisi)
                                            hsv_crop = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
                                            s_mean = float(np.mean(hsv_crop[:, :, 1])) / 255.0
                                            v_mean = float(np.mean(hsv_crop[:, :, 2])) / 255.0
                                            lum_weight = max(0.15, s_mean * v_mean)
                                            color_votes.setdefault(color_name, []).append(top1_conf * lum_weight)

                            # Gorsel annotasyon
                            if writer:
                                label = f"{vtype} {conf:.2f}"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # --- Plaka ---
                        elif cls_id == PLATE_CLASS_ID:
                            if writer:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, f"plaka {conf:.2f}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            # Fast-Plate-OCR / PlateReader cagirimi
                            px1, py1 = max(0, x1), max(0, y1)
                            px2, py2 = min(width, x2), min(height, y2)
                            plate_crop = frame[py1:py2, px1:px2]

                            if plate_crop.size > 0:
                                plate_text = plate_reader.read(plate_crop)
                                if plate_text and len(plate_text) >= 2:
                                    plate_texts.append((plate_text, conf))
                                    if writer:
                                        cv2.putText(frame, plate_text, (x1, y2 + 20),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if writer:
                writer.write(frame)

            processed += 1

            # Ilerleme
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

    # ========== SONUCLARI TOPLA (PostProcessor ile) ==========
    from src.postprocessor import PostProcessor
    postprocessor = PostProcessor()
    avg_aspect_ratio = float(np.mean(aspect_ratios)) if aspect_ratios else 1.5

    vehicle_info = postprocessor.aggregate_vehicle_info(
        type_votes, color_votes, plate_texts, bbox_aspect_ratio=avg_aspect_ratio
    )

    best_type = vehicle_info.get("tip", "sedan")
    best_color = vehicle_info.get("renk", "")
    best_plate = vehicle_info.get("plaka", "tespit edilemedi")
    best_conf = vehicle_info.get("confidence_score", 0.0)

    # ========== JSON CIKTI ==========
    output = {
        "video_id": video_path.name,
        "arac_bilgisi": vehicle_info,
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
    logger.info(f"  Arac Tipi  : {best_type} (conf: {best_conf:.4f}, {len(type_votes.get(best_type, []))} oy)")
    logger.info(f"  Arac Rengi : {best_color} ({len(color_votes.get(best_color, []))} oy)")
    logger.info(f"  Plaka      : {best_plate}")
    logger.info(f"  JSON       : {json_path}")

    if type_votes:
        logger.info(f"  Tip Dagilimi: {', '.join(f'{k}({len(v)})' for k, v in sorted(type_votes.items(), key=lambda x: -len(x[1])))}")
    if color_votes:
        logger.info(f"  Renk Dagilimi: {', '.join(f'{k}({len(v)})' for k, v in sorted(color_votes.items(), key=lambda x: -len(x[1])))}")
    if plate_texts:
        unique_plates = list(set(t for t, _ in plate_texts))
        logger.info(f"  Plaka Okumalari ({len(plate_texts)}x): {', '.join(unique_plates[:5])}")

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
