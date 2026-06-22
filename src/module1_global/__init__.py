# -*- coding: utf-8 -*-
"""
Modul 1: Kuresel Baglam ve Makro Tespit Agi

YOLOv12s (TensorRT FP16) ile tam kare uzerinde:
  - Arac tipi ve renk siniflandirmasi
  - Teknocan ve bilgisayar nesnelerinin tespiti
  - Plaka bolgesi (Plate ROI) cikarimi
  - On cam bolgesi (Windshield ROI) cikarimi
"""

from src.module1_global.detector import GlobalDetector

__all__ = ["GlobalDetector"]
