#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
YOLO Model Egitim Betigi (OOM-Safe / GPU Optimize)

Kullanim:
  python tools/train_models.py --task detection
  python tools/train_models.py --task classification

Guvenlik Onlemleri:
  - workers=2 (RAM sizintisi onlenir — gece oturan OOM bugun tekrarlanmaz)
  - batch=16 detection / batch=32 classification (VRAM guvenli sinir)
  - Egitim oncesi ve sonrasi acik gc.collect() + torch.cuda.empty_cache()
  - save_period=5 (her 5 epoch'ta checkpoint — cokse bile kayip en fazla 5 epoch)
  - Hata yakalamayla best.pt kopyalama garanti edilir
"""

import os
import sys
import gc
import argparse
import shutil
from pathlib import Path

# Ortam degiskenlerini ayarla
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("[HATA] ultralytics veya torch yuklu degil.")
    print("  Cozum: python -m pip install ultralytics")
    sys.exit(1)

# ==============================================================================
# Sabitler
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def clean_memory():
    """GPU ve RAM belleklerini temizle."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def print_gpu_status():
    """GPU bellek durumunu yazdir."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Bellek: {allocated:.1f} GB kullanilan / {reserved:.1f} GB ayrilmis / {total:.1f} GB toplam")


def safe_copy_weights(src: Path, dst: Path):
    """Model agirliklarini guvenli sekilde kopyala."""
    try:
        if src.exists():
            shutil.copy2(src, dst)
            size_mb = dst.stat().st_size / 1024**2
            print(f"✅ Model kaydedildi: {dst} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"⚠️ Kaynak dosya bulunamadi: {src}")
            return False
    except Exception as e:
        print(f"❌ Kopyalama hatasi: {e}")
        return False


# ==============================================================================
# DETECTION EGITIMI (Modul 1)
# ==============================================================================
def train_detection():
    """
    Modul 1 (Kuresel Tespit): Arac tipi + Plaka bbox detection
    OOM-safe parametreler:
      - batch=16 (32'den dusuruldu — VRAM guvenli)
      - workers=2 (8'den dusuruldu — RAM sizintisi onlendi)
      - save_period=5 (cokmeye karsi periyodik kayit)
    """
    print("=" * 60)
    print("MODUL 1: KURESEL TESPIT (DETECTION) EGITIMI")
    print("=" * 60)

    data_yaml = DATASETS_DIR / "merged_detection" / "data.yaml"
    if not data_yaml.exists():
        print(f"[HATA] Veri seti bulunamadi: {data_yaml}")
        return

    clean_memory()
    print_gpu_status()

    # Model yukle
    model_name = "yolo11s.pt"
    try:
        model = YOLO(model_name)
    except Exception:
        print(f"  {model_name} bulunamadi, yolov8s.pt deneniyor...")
        model = YOLO("yolov8s.pt")

    # --- OOM-SAFE PARAMETRELER ---
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,             # 32'den 16'ya dusuruldu (VRAM guvenli)
        device=0,
        workers=2,            # 8'den 2'ye dusuruldu (RAM sizintisi onlemi)
        project=str(BASE_DIR / "runs" / "train"),
        name="global_detection",
        exist_ok=True,
        amp=True,             # Mixed precision — VRAM tasarrufu
        patience=15,
        save_period=5,        # Her 5 epoch'ta checkpoint kaydet
        cache=False,          # RAM'de onbellekleme KAPALI (16GB RAM icin zorunlu)
        resume=False,
    )

    clean_memory()

    # En iyi agirligi kopyala
    best_pt = BASE_DIR / "runs" / "train" / "global_detection" / "weights" / "best.pt"
    safe_copy_weights(best_pt, MODELS_DIR / "global_yolo.pt")


# ==============================================================================
# CLASSIFICATION EGITIMI (Arac Rengi)
# ==============================================================================
def train_classification():
    """
    Arac Rengi Siniflandirma
    OOM-safe parametreler:
      - batch=32 (64'den dusuruldu)
      - workers=2 (8'den dusuruldu)
      - imgsz=224 (siniflandirma icin standart)
    """
    print("=" * 60)
    print("ARAC RENGI SINIFLANDIRMA (CLASSIFICATION) EGITIMI")
    print("=" * 60)

    data_dir = DATASETS_DIR / "color_classification"
    if not data_dir.exists():
        print(f"[HATA] Veri seti bulunamadi: {data_dir}")
        return

    clean_memory()
    print_gpu_status()

    # Model yukle
    model_name = "yolo11n-cls.pt"
    try:
        model = YOLO(model_name)
    except Exception:
        print(f"  {model_name} bulunamadi, yolov8n-cls.pt deneniyor...")
        model = YOLO("yolov8n-cls.pt")

    # --- OOM-SAFE PARAMETRELER ---
    results = model.train(
        data=str(data_dir),
        epochs=30,
        imgsz=224,
        batch=32,             # 64'den 32'ye dusuruldu
        device=0,
        workers=2,            # 8'den 2'ye dusuruldu
        project=str(BASE_DIR / "runs" / "train"),
        name="color_classification",
        exist_ok=True,
        amp=True,
        patience=10,
        save_period=5,        # Her 5 epoch'ta checkpoint
        cache=False,
        resume=False,
    )

    clean_memory()

    # En iyi agirligi kopyala
    best_pt = BASE_DIR / "runs" / "train" / "color_classification" / "weights" / "best.pt"
    safe_copy_weights(best_pt, MODELS_DIR / "color_yolo.pt")


# ==============================================================================
# CABIN EGITIMI (Modul 2)
# ==============================================================================
def train_cabin():
    """
    Modul 2 (Kabin Analizi): Sofor eylemleri + Emniyet kemeri detection
    OOM-safe parametreler:
      - batch=16
      - workers=2
      - save_period=5
      - model: yolo11n.pt (Nano model - Modul 2 hiz ve hafiflik kistina uygun)
    """
    print("=" * 60)
    print("MODUL 2: KABIN ICI ANALIZ (CABIN DETECTION) EGITIMI")
    print("=" * 60)

    data_yaml = DATASETS_DIR / "merged_cabin_monitoring" / "data.yaml"
    if not data_yaml.exists():
        print(f"[HATA] Veri seti bulunamadi: {data_yaml}")
        print("  Öncelikle çalıştırın: python tools/prepare_cabin_datasets.py")
        return

    clean_memory()
    print_gpu_status()

    # Model yukle
    model_name = "yolo11n.pt"
    try:
        model = YOLO(model_name)
    except Exception:
        print(f"  {model_name} bulunamadi, yolov8n.pt deneniyor...")
        model = YOLO("yolov8n.pt")

    # --- OOM-SAFE PARAMETRELER ---
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        project=str(BASE_DIR / "runs" / "train"),
        name="cabin_detection",
        exist_ok=True,
        amp=True,
        patience=15,
        save_period=5,
        cache=False,
        resume=False,
    )

    clean_memory()

    # En iyi agirligi kopyala
    best_pt = BASE_DIR / "runs" / "train" / "cabin_detection" / "weights" / "best.pt"
    safe_copy_weights(best_pt, MODELS_DIR / "cabin_yolo.pt")


# ==============================================================================
# ANA GIRIS
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TEKNOFEST 2026 ANKAAI — OOM-Safe YOLO Model Egitim Betigi"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["detection", "classification", "cabin"],
        required=True,
        help="Hangi model egitilecek? (detection / classification / cabin)",
    )
    args = parser.parse_args()

    # Sistem bilgisi
    print("=" * 60)
    print("SISTEM BILGISI")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda if torch.cuda.is_available() else 'YOK'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
        print_gpu_status()
    else:
        print("  ⚠️ GPU bulunamadi, CPU ile devam ediliyor (cok yavas olacak).")
    print()

    # Gorevi calistir
    try:
        if args.task == "detection":
            train_detection()
        elif args.task == "classification":
            train_classification()
        elif args.task == "cabin":
            train_cabin()
    except Exception as e:
        print(f"\n❌ EGITIM SIRASINDA HATA: {e}")
        clean_memory()
        sys.exit(1)
    finally:
        clean_memory()
        print("\n🏁 Betik tamamlandi. Bellek temizlendi.")

