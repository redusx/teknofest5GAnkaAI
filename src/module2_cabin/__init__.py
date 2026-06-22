# -*- coding: utf-8 -*-
"""
Modul 2: Kabin Ici Davranis ve Yolcu Analizi

YOLOv12n (TensorRT FP16) ile on cam kirpintisi uzerinde:
  - Sofor eylemleri (telefon, su, sigara, esneme, kemer ihlali vb.)
  - Yolcu konumlandirmasi (on_koltuk, arka_koltuk_1, arka_koltuk_2)
"""

from src.module2_cabin.analyzer import CabinAnalyzer

__all__ = ["CabinAnalyzer"]
