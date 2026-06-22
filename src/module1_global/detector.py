#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Modul 1: Kuresel Baglam ve Makro Tespit Agi (Global Detector)

YOLOv12s (TensorRT FP16) ile tam kare uzerinde:
  - Arac tipi siniflandirmasi (sedan, suv, hatchback, pickup, minibus, panelvan, kamyon)
  - Arac renk tespiti (HSV tabanli)
  - Teknocan ve bilgisayar nesnelerinin tespiti
  - Plaka bolgesi (Plate ROI) cikarimi
  - On cam bolgesi (Windshield ROI) cikarimi

Bu modul, diger modullere ROI kirpintilari saglar:
  - Plaka ROI -> Modul 3 (OCR)
  - On Cam ROI -> Modul 2 (Kabin Analizi)
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.utils import (
    sanitize_label,
    clamp_confidence,
    VALID_VEHICLE_TYPES,
    VALID_COLORS,
)

logger = logging.getLogger("module1_global")


# ==============================================================================
# Veri Yapilari
# ==============================================================================
@dataclass
class GlobalDetectionResult:
    """Modul 1 cikarim sonucu."""

    # Arac tipi ve confidence
    vehicle_type: Optional[str] = None
    vehicle_type_conf: float = 0.0
    vehicle_bbox: Optional[Tuple[int, int, int, int]] = None

    # Arac rengi
    vehicle_color: Optional[str] = None
    vehicle_color_conf: float = 0.0

    # Plaka ROI (kirpma koordinatlari)
    plate_roi: Optional[Tuple[int, int, int, int]] = None
    plate_conf: float = 0.0

    # On cam ROI (kirpma koordinatlari)
    windshield_roi: Optional[Tuple[int, int, int, int]] = None
    windshield_conf: float = 0.0

    # Nesne tespitleri (teknocan, bilgisayar)
    object_detections: List[Dict[str, Any]] = field(default_factory=list)


# ==============================================================================
# Sinif ID Esleme Tablolari (class_config.yaml ile uyumlu)
# ==============================================================================
DEFAULT_VEHICLE_CLASS_IDS = {
    0: "sedan",
    1: "suv",
    2: "hatchback",
    3: "pickup",
    4: "minibus",
    5: "panelvan",
    6: "kamyon",
}

DEFAULT_OBJECT_CLASS_IDS = {
    15: "teknocan",
    16: "bilgisayar",
}

DEFAULT_PLATE_CLASS_ID = 20
DEFAULT_WINDSHIELD_CLASS_ID = 21  # Eger modelde varsa


class GlobalDetector:
    """
    Kuresel baglam ve makro tespit modulu.

    Video karesinin tamamini analiz ederek arac bilgilerini ve ROI
    bolgelerini cikarir. Diger modullere girdi saglayan ana moduldur.

    Attributes:
        model: YOLO modeli.
        vehicle_class_ids: Arac tipi sinif ID esleme tablosu.
        object_class_ids: Nesne sinif ID esleme tablosu.
        plate_class_id: Plaka sinif ID'si.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        vehicle_class_ids: Optional[Dict[int, str]] = None,
        object_class_ids: Optional[Dict[int, str]] = None,
        plate_class_id: int = DEFAULT_PLATE_CLASS_ID,
        windshield_class_id: int = DEFAULT_WINDSHIELD_CLASS_ID,
    ):
        """
        Args:
            model_path: YOLOv12s model dosyasi yolu (.pt veya .engine).
            device: Cihaz secimi ("auto", "cuda", "cpu").
            conf_threshold: Minimum guven esigi.
            iou_threshold: NMS IOU esigi.
            img_size: YOLO girdi boyutu.
            vehicle_class_ids: Arac tipi sinif ID esleme tablosu.
            object_class_ids: Nesne sinif ID esleme tablosu.
            plate_class_id: Plaka sinif ID'si.
            windshield_class_id: On cam sinif ID'si.
        """
        self.model = None
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        self.vehicle_class_ids = vehicle_class_ids or DEFAULT_VEHICLE_CLASS_IDS
        self.object_class_ids = object_class_ids or DEFAULT_OBJECT_CLASS_IDS
        self.plate_class_id = plate_class_id
        self.windshield_class_id = windshield_class_id

        # Video boyunca birikimli veriler
        self._type_votes: Dict[str, List[float]] = {}
        self._color_votes: Dict[str, List[float]] = {}
        self._plate_detections: List[Tuple[str, float]] = []

        if model_path:
            self.load_model(model_path, device)

    def load_model(self, model_path: str, device: str = "auto"):
        """
        YOLO modelini yukler.

        Args:
            model_path: Model dosyasi yolu (.pt veya .engine).
            device: Cihaz secimi.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[Modul 1] Model dosyasi bulunamadi: {model_path}"
            )

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)

            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model.to(device)
            self.device = device
            logger.info(f"[Modul 1] Model yuklendi: {model_path} ({device})")

        except Exception as e:
            raise RuntimeError(f"[Modul 1] Model yuklenemedi: {e}")

    def reset(self):
        """Yeni bir video icin tum birikimleri sifirlar."""
        self._type_votes.clear()
        self._color_votes.clear()
        self._plate_detections.clear()

    # ==================================================================
    # Ana Tespit Fonksiyonu
    # ==================================================================
    def detect(
        self,
        frame: np.ndarray,
        frame_time: float = 0.0,
        use_tracker: bool = True,
    ) -> GlobalDetectionResult:
        """
        Tam kare uzerinde kuresel tespit yapar.

        Args:
            frame: OpenCV goruntusu (BGR, numpy array).
            frame_time: Karenin video icerisindeki zamani (saniye).
            use_tracker: ByteTrack takip algoritmasi kullanimi.

        Returns:
            GlobalDetectionResult: Tespit sonuclari ve ROI koordinatlari.
        """
        result = GlobalDetectionResult()

        if self.model is None:
            logger.warning("[Modul 1] Model yuklenmemis, tespit atlanıyor.")
            return result

        try:
            # YOLO cikarim
            if use_tracker:
                predictions = self.model.track(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            else:
                predictions = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    verbose=False,
                )

            if not predictions or len(predictions) == 0:
                return result

            pred = predictions[0]
            if pred.boxes is None:
                return result

            # Her kutuyu isle
            for box in pred.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # --- Arac Tipi Tespiti ---
                if class_id in self.vehicle_class_ids:
                    vtype = self.vehicle_class_ids[class_id]
                    result.vehicle_type = vtype
                    result.vehicle_type_conf = confidence
                    result.vehicle_bbox = (x1, y1, x2, y2)

                    # Birikimli oy
                    self._type_votes.setdefault(vtype, []).append(confidence)

                    # Renk tespiti (arac bbox icinden)
                    color = self._detect_color(frame, x1, y1, x2, y2)
                    if color:
                        result.vehicle_color = color
                        result.vehicle_color_conf = confidence
                        self._color_votes.setdefault(color, []).append(
                            confidence
                        )

                # --- Plaka ROI ---
                elif class_id == self.plate_class_id:
                    result.plate_roi = (x1, y1, x2, y2)
                    result.plate_conf = confidence

                # --- On Cam ROI ---
                elif class_id == self.windshield_class_id:
                    result.windshield_roi = (x1, y1, x2, y2)
                    result.windshield_conf = confidence

                # --- Nesne Tespiti (teknocan, bilgisayar) ---
                elif class_id in self.object_class_ids:
                    label = self.object_class_ids[class_id]
                    result.object_detections.append(
                        {
                            "zaman_saniye": round(frame_time, 2),
                            "kategori": "nesneler",
                            "etiket": sanitize_label(label),
                            "confidence_score": clamp_confidence(confidence),
                        }
                    )

        except Exception as e:
            logger.warning(f"[Modul 1] Kare islenirken hata: {e}")

        return result

    # ==================================================================
    # Renk Tespiti
    # ==================================================================
    def _detect_color(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> Optional[str]:
        """
        Arac bbox'i icerisinden baskin rengi belirler.
        HSV renk uzayinda ortalama ton (hue) degerine gore siniflandirma yapar.

        Args:
            frame: Orijinal goruntu.
            x1, y1, x2, y2: Bounding box koordinatlari.

        Returns:
            Renk etiketi veya None.
        """
        try:
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                return None

            roi = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            v_mean = np.mean(hsv[:, :, 2])

            # Dusuk doygunluk -> beyaz, siyah veya gri
            if s_mean < 40:
                if v_mean > 200:
                    return "beyaz"
                elif v_mean < 60:
                    return "siyah"
                else:
                    return "gri"

            # Ton bazli renk siniflandirma
            if h_mean < 10 or h_mean > 170:
                return "kirmizi"
            elif 10 <= h_mean < 25:
                return "turuncu"
            elif 25 <= h_mean < 35:
                return "sari"
            elif 35 <= h_mean < 85:
                return "yesil"
            elif 85 <= h_mean < 130:
                return "mavi"
            elif 130 <= h_mean < 170:
                return "kirmizi"

            return None

        except Exception as e:
            logger.debug(f"Renk tespiti hatasi: {e}")
            return None

    # ==================================================================
    # Birikimli Sonuc Erisicileri
    # ==================================================================
    def get_type_votes(self) -> Dict[str, List[float]]:
        """Video boyunca biriken arac tipi oylarini dondurur."""
        return self._type_votes

    def get_color_votes(self) -> Dict[str, List[float]]:
        """Video boyunca biriken renk oylarini dondurur."""
        return self._color_votes

    def get_plate_detections(self) -> List[Tuple[str, float]]:
        """Video boyunca biriken plaka tespitlerini dondurur."""
        return self._plate_detections

    def add_plate_detection(self, plate_text: str, confidence: float):
        """Modul 3'ten gelen plaka metnini birikime ekler."""
        self._plate_detections.append((plate_text, confidence))
