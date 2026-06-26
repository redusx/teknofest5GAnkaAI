#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Modul 2: Kabin Ici Davranis ve Yolcu Analizi (Cabin Analyzer)

YOLOv12n (TensorRT FP16) ile on cam kirpintisi uzerinde:
  - Sofor eylemleri tespiti
  - Yolcu konumlandirmasi (heuristik kural seti)
  - Emniyet kemeri ihlali tespiti

Desteklenen etiketler (FTR Model Kilavuz §Tablo2):
  sofor_eylemi: arkaya_bakma, esneme, sigara_icme, su_icme,
                telefonla_konusma, etrafa_bakinma, emniyet_kemeri_ihlali
  yolcular:     arka_koltuk_1, arka_koltuk_2, on_koltuk
"""

import os
import logging
from typing import Dict, List, Optional, Any

import numpy as np

from src.utils import (
    sanitize_label,
    clamp_confidence,
    get_category_for_class,
    VALID_DRIVER_ACTIONS,
    VALID_PASSENGERS,
)

logger = logging.getLogger("module2_cabin")


# ==============================================================================
# Sinif ID Esleme Tablolari
# ==============================================================================
DEFAULT_ACTION_CLASS_IDS = {
    7: "arkaya_bakma",
    8: "esneme",
    9: "sigara_icme",
    10: "su_icme",
    11: "telefonla_konusma",
    13: "etrafa_bakinma",
    14: "emniyet_kemeri_ihlali",
}

DEFAULT_PASSENGER_CLASS_IDS = {
    17: "arka_koltuk_1",
    18: "arka_koltuk_2",
    19: "on_koltuk",
}


class CabinAnalyzer:
    """
    Kabin ici davranis ve yolcu analiz modulu.

    Modul 1'in cikardigi on cam kirpintisini alir ve:
      1. Sofor eylemlerini tespit eder (telefon, su, sigara, esneme vb.)
      2. Emniyet kemeri ihlali tespit eder
      3. Yolcuları konum bazli siniflandirir (heuristik kural seti)

    NOT: Eger on cam kirpintisi gelmezse (Modul 1 on cam bulamadiysa)
         bu modul calistirilmaz (kosullu dallanma).

    Attributes:
        model: YOLOv12n modeli.
        action_class_ids: Sofor eylemi sinif ID esleme tablosu.
        passenger_class_ids: Yolcu sinif ID esleme tablosu.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        action_class_ids: Optional[Dict[int, str]] = None,
        passenger_class_ids: Optional[Dict[int, str]] = None,
        min_confidence: float = 0.25,
    ):
        """
        Args:
            model_path: YOLOv12n model dosyasi yolu (.pt veya .engine).
            device: Cihaz secimi.
            conf_threshold: Minimum guven esigi.
            iou_threshold: NMS IOU esigi.
            img_size: YOLO girdi boyutu.
            action_class_ids: Sofor eylemi sinif ID esleme tablosu.
            passenger_class_ids: Yolcu sinif ID esleme tablosu.
            min_confidence: Minimum raporlama guven esigi.
        """
        self.model = None
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.min_confidence = min_confidence

        self.action_class_ids = action_class_ids or DEFAULT_ACTION_CLASS_IDS
        self.passenger_class_ids = (
            passenger_class_ids or DEFAULT_PASSENGER_CLASS_IDS
        )

        # Tum sinif ID'lerini birlestir
        self._all_class_ids: Dict[int, str] = {}
        self._all_class_ids.update(self.action_class_ids)
        self._all_class_ids.update(self.passenger_class_ids)

        # Tespit biriktirici
        self._detections: List[Dict[str, Any]] = []
        self.person_detector = None

        if model_path:
            self.load_model(model_path, device)

    def load_model(self, model_path: str, device: str = "auto"):
        """YOLOv12n modelini yukler."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[Modul 2] Model dosyasi bulunamadi: {model_path}"
            )

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)

            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model.to(device)
            self.device = device
            logger.info(f"[Modul 2] Model yuklendi: {model_path} ({device})")

        except Exception as e:
            raise RuntimeError(f"[Modul 2] Model yuklenemedi: {e}")

    def reset(self):
        """Yeni bir video icin tum birikimleri sifirlar."""
        self._detections.clear()

    # ==================================================================
    # Ana Analiz Fonksiyonu
    # ==================================================================
    def analyze(
        self,
        windshield_crop: np.ndarray,
        frame_time: float,
        full_frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        On cam kirpintisi uzerinde kabin ici analiz yapar.

        Args:
            windshield_crop: On cam kirpintisi (Modul 1 ciktisi).
            frame_time: Karenin video icerisindeki zamani (saniye).
            full_frame: Tam kare (yolcu konumlandirmasi icin gerekebilir).

        Returns:
            Bu karede bulunan tespitlerin listesi.
        """
        frame_detections: List[Dict[str, Any]] = []

        if self.model is None:
            logger.warning("[Modul 2] Model yuklenmemis, analiz atlaniyor.")
            return frame_detections

        if windshield_crop is None or windshield_crop.size == 0:
            return frame_detections

        try:
            predictions = self.model.predict(
                windshield_crop,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False,
            )

            if not predictions or len(predictions) == 0:
                return frame_detections

            pred = predictions[0]
            if pred.boxes is None:
                return frame_detections

            for box in pred.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id not in self._all_class_ids:
                    continue

                if confidence < self.min_confidence:
                    continue

                label = self._all_class_ids[class_id]
                kategori = get_category_for_class(label)

                detection = {
                    "zaman_saniye": round(frame_time, 2),
                    "kategori": kategori,
                    "etiket": sanitize_label(label),
                    "confidence_score": clamp_confidence(confidence),
                }

                frame_detections.append(detection)
                self._detections.append(detection)

        except Exception as e:
            logger.warning(f"[Modul 2] Kare islenirken hata: {e}")

        # Geometrik Sabit ROI Tabanlı Yolcu Tarama Kuralı (FTR §Tablo2)
        target_img = full_frame if full_frame is not None else windshield_crop
        p_dets = self.detect_passengers_roi(target_img, frame_time)
        frame_detections.extend(p_dets)

        return frame_detections

    # ==================================================================
    # Geometrik ROI Yolcu Saptayıcı (YOLOv11n Fallback)
    # ==================================================================
    def detect_passengers_roi(
        self, image: Any, frame_time: float
    ) -> List[Dict[str, Any]]:
        """
        Kabin içi sabit koltuk koordinatlarını tarayarak kişi varlığına göre etiket üretir.
          Sağ Üst ROI -> on_koltuk (19)
          Sol Alt ROI -> arka_koltuk_1 (17)
          Sağ Alt ROI -> arka_koltuk_2 (18)
        """
        results: List[Dict[str, Any]] = []
        if image is None or image.size == 0:
            return results

        if self.person_detector is None:
            try:
                from ultralytics import YOLO
                self.person_detector = YOLO("yolo11n.pt")
            except Exception:
                return results

        try:
            h, w = image.shape[:2]
            preds = self.person_detector.predict(image, conf=0.35, verbose=False)
            if not preds or not preds[0].boxes:
                return results

            for box in preds[0].boxes:
                if int(box.cls[0]) != 0:
                    continue
                conf = float(box.conf[0])
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cx, cy = (bx1 + bx2) / (2.0 * w), (by1 + by2) / (2.0 * h)

                seat = None
                if cx > 0.48 and cy < 0.65:
                    seat = "on_koltuk"
                elif cx < 0.50 and cy >= 0.50:
                    seat = "arka_koltuk_1"
                elif cx >= 0.50 and cy >= 0.50:
                    seat = "arka_koltuk_2"

                if seat:
                    det = {
                        "zaman_saniye": round(frame_time, 2),
                        "kategori": "yolcular",
                        "etiket": sanitize_label(seat),
                        "confidence_score": clamp_confidence(conf),
                    }
                    results.append(det)
                    self._detections.append(det)
        except Exception:
            pass

        return results

    def analyze_with_shared_results(
        self,
        yolo_result: Any,
        frame_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Paylasimli model sonuclari ile kabin analizi yapar.

        Gelistirme asamasinda tek model kullanildiginda, Modul 1'in
        YOLO sonuclari dogrudan bu fonksiyona aktarilir.

        Args:
            yolo_result: YOLO cikarim sonucu (ultralytics Result).
            frame_time: Karenin zamani (saniye).

        Returns:
            Bu karede bulunan tespitlerin listesi.
        """
        frame_detections: List[Dict[str, Any]] = []

        if yolo_result is None:
            return frame_detections

        boxes = yolo_result.boxes
        if boxes is None:
            return frame_detections

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id not in self._all_class_ids:
                continue

            if confidence < self.min_confidence:
                continue

            label = self._all_class_ids[class_id]
            kategori = get_category_for_class(label)

            detection = {
                "zaman_saniye": round(frame_time, 2),
                "kategori": kategori,
                "etiket": sanitize_label(label),
                "confidence_score": clamp_confidence(confidence),
            }

            frame_detections.append(detection)
            self._detections.append(detection)

        return frame_detections

    # ==================================================================
    # Sonuc Erisicileri
    # ==================================================================
    def get_all_detections(self) -> List[Dict[str, Any]]:
        """Tum video boyunca biriken tespitleri zamana gore sirali dondurur."""
        return sorted(self._detections, key=lambda d: d["zaman_saniye"])

    def get_detections_by_category(
        self, category: str
    ) -> List[Dict[str, Any]]:
        """Belirli bir kategorideki tespitleri filtreler."""
        return [d for d in self._detections if d["kategori"] == category]

    def get_summary(self) -> Dict[str, Any]:
        """Tespit istatistiklerinin ozetini dondurur."""
        return {
            "toplam_tespit": len(self._detections),
            "sofor_eylemi": len(
                self.get_detections_by_category("sofor_eylemi")
            ),
            "yolcular": len(self.get_detections_by_category("yolcular")),
            "benzersiz_etiketler": list(
                set(d["etiket"] for d in self._detections)
            ),
        }
