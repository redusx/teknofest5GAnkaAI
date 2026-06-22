# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
src paketi — 4 Kademeli Kaskat Mikro-Model Mimarisi

Moduller:
  module1_global    : Kuresel Baglam ve Makro Tespit (YOLOv12s)
  module2_cabin     : Kabin Ici Davranis ve Yolcu Analizi (YOLOv12n)
  module3_ocr       : Plaka Okuma / OCR (Fast-Plate-OCR / CCT-S-V2)
  module4_trajectory: Kinematik Yorunge ve Slalom Tespiti (ByteTrack + EKF)
  pipeline          : Ana orkestrator (cikarim hatti)
  postprocessor     : Son islem motoru (zamansal NMS, JSON cikti)
  utils             : Yardimci fonksiyonlar
"""
