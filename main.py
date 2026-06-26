#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - 5G & Yapay Zeka ile Akilli Yol Guvenligi Yarismasi
Takim: ANKAAI

Ana Giris Noktasi (Entrypoint)

Bu dosya, Docker konteyneri ayaga kalktiginda otomatik olarak calisir.
4 Kademeli Kaskat Mikro-Model mimarisi ile video analizi yapar
ve sonuclari yarisma standartlarina uygun JSON formatinda kaydeder.

Mimari:
  Modul 1: Kuresel Tespit (YOLOv12s)   — arac, renk, nesneler, ROI
  Modul 2: Kabin Analizi (YOLOv12n)    — sofor eylemleri, yolcular
  Modul 3: Plaka OCR (CCT-S-V2)        — plaka metni
  Modul 4: Yorunge Takibi (ByteTrack)  — slalom tespiti

Girdi:  /app/data/input/video.mp4
Cikti:  /app/data/output/results.json
Model:  /app/models/

Kullanim:
    python3 main.py
"""

import os
import sys
import json
import time
import logging

# ==============================================================================
# Logger Konfigurasyonu
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# src klasorundeki modullere erisim saglamak icin import yollarini ekliyoruz
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Ana cikarim hattini ice aktar
from src.pipeline import InferencePipeline

# ==============================================================================
# Sabitler (FTR Model Kilavuz §6 ve §8)
# ==============================================================================
INPUT_PATH = "/app/data/input/video.mp4"
OUTPUT_PATH = "/app/data/output/results.json"
MODELS_DIR = "/app/models/"


def main():
    """
    Ana giris noktasi.

    1. Girdi video dosyasini kontrol eder
    2. Model dizinini kontrol eder
    3. 4 kademeli kaskat cikarim hattini baslatir
    4. Sonuclari JSON formatinda diske yazar (ensure_ascii=False)

    Tum islemler try-except bloklari ile korunmustur (FTR Model Kilavuz §7).
    """
    logger.info("=" * 60)
    logger.info("TEKNOFEST 2026 - ANKAAI")
    logger.info("5G & Yapay Zeka ile Akilli Yol Guvenligi")
    logger.info("4 Kademeli Kaskat Cikarim Hatti Baslatiliyor...")
    logger.info("=" * 60)

    start_time = time.time()

    # --- Girdi Kontrolu ---
    logger.info(f"Girdi videosu : {INPUT_PATH}")
    logger.info(f"Model dizini  : {MODELS_DIR}")
    logger.info(f"Cikti yolu    : {OUTPUT_PATH}")

    if not os.path.exists(INPUT_PATH):
        logger.error(f"Hata: Girdi videosu bulunamadi -> {INPUT_PATH}")
        sys.exit(1)

    # Model dizini kontrolu
    models_dir = MODELS_DIR
    if not os.path.exists(models_dir):
        logger.warning(
            f"Model dizini bulunamadi: {models_dir}, alternatif araniyor..."
        )
        # Alternatif yollar dene
        for alt_dir in ["/app/weights/", "./models/", "./weights/"]:
            if os.path.exists(alt_dir):
                models_dir = alt_dir
                logger.info(f"Alternatif model dizini bulundu: {models_dir}")
                break
        else:
            logger.error("Hicbir model dizini bulunamadi.")
            sys.exit(1)

    # --- Cikarim Islemi ---
    try:
        logger.info("")
        logger.info("Kaskat cikarim hatti tetikleniyor...")

        # Modül 2 bağımsız kaskat modeli var mı kontrol et
        has_cabin_model = any(
            os.path.exists(os.path.join(models_dir, f"cabin_yolo{ext}"))
            for ext in [".pt", ".onnx", ".engine"]
        )
        shared_mode = not has_cabin_model
        if not shared_mode:
            logger.info("Bağımsız Modül 2 kabin modeli bulundu, kaskat mod aktif edildi.")

        # Pipeline olustur
        pipeline = InferencePipeline(
            models_dir=models_dir,
            device="auto",
            use_tracker=True,
            shared_model=shared_mode,
        )

        # Cikarim yap
        output_data = pipeline.run(video_path=INPUT_PATH)

        # --- Cikti Dizinini Olustur ---
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        # --- Sonuclari JSON Formatinda Yaz ---
        # ensure_ascii=False: Turkce karakterlerin duzgun yazilmasi (ZORUNLU)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("Islem basariyla tamamlandi!")
        logger.info(f"Cikti kaydedildi: {OUTPUT_PATH}")
        logger.info(f"Toplam sure: {elapsed:.2f} saniye")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"Dosya bulunamadi hatasi: {str(e)}")
        sys.exit(1)

    except RuntimeError as e:
        logger.error(f"Calisma zamani hatasi: {str(e)}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Model calistirilirken hata: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
