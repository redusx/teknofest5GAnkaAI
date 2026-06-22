# -*- coding: utf-8 -*-
"""
Modul 3: Plaka Okuma (OCR) Agi

Fast-Plate-OCR (CCT-S-V2) ile plaka ROI kirpintisi uzerinde:
  - Perspektif duzeltme (perspective warp)
  - Optik karakter tanima
  - Turkiye plaka standardi regex dogrulamasi
"""

from src.module3_ocr.plate_reader import PlateReader

__all__ = ["PlateReader"]
