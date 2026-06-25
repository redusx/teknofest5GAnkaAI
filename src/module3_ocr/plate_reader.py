#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Modul 3: Plaka Okuma (OCR) Agi (Plate Reader)

Plaka ROI kirpintisi uzerinde:
  - Perspektif duzeltme (perspective warp)
  - Optik karakter tanima (Fast-Plate-OCR / CCT-S-V2)
  - Turkiye plaka standardi regex dogrulamasi

Hedef Model: Fast-Plate-OCR (Kompakt Evrisimli Transformer)
  - T4 GPU uzerinde ~0.67 ms cikarim suresi
  - ONNX Runtime (TensorRT backend) ile optimize
"""

import logging
from typing import Optional

import cv2
import numpy as np

from src.utils import validate_plate

logger = logging.getLogger("module3_ocr")


class PlateReader:
    """
    Plaka okuma (OCR) modulu.

    Modul 1'in cikardigi plaka ROI kirpintisini alir ve:
      1. Goruntu on-isleme (kontrast, keskinlestirme)
      2. Perspektif duzeltme (opsiyonel)
      3. OCR ile metin cikarim
      4. Turkiye plaka regex dogrulamasi

    NOT: Eger plaka ROI bulunamazsa (Modul 1 plaka tespit edemediyse)
         bu modul calistirilmaz (kosullu dallanma).

    Attributes:
        ocr_engine: OCR motoru (Fast-Plate-OCR veya EasyOCR).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        ocr_backend: str = "auto",
    ):
        """
        Args:
            model_path: OCR model dosyasi yolu (ONNX veya .pt).
            ocr_backend: OCR motoru secimi:
                "fast_plate_ocr" - Fast-Plate-OCR (CCT-S-V2)
                "easyocr"        - EasyOCR
                "auto"           - Kullanilabilir olani otomatik sec
        """
        self.model_path = model_path
        self.ocr_backend = ocr_backend
        self.ocr_engine = None

        self._initialize_ocr()

    def _initialize_ocr(self):
        """OCR motorunu baslatir."""
        # --- Fast-Plate-OCR (Tercih edilen) ---
        if self.ocr_backend in ("fast_plate_ocr", "auto"):
            try:
                from fast_plate_ocr import LicensePlateRecognizer

                self.ocr_engine = LicensePlateRecognizer("cct-s-v2-global-model")
                self.ocr_backend = "fast_plate_ocr"
                logger.info("[Modul 3] Fast-Plate-OCR baslatildi.")
                return
            except Exception as e:
                if self.ocr_backend == "fast_plate_ocr":
                    logger.error(f"[Modul 3] fast-plate-ocr baslatilamadı: {e}")
                    return
                logger.debug(f"[Modul 3] fast-plate-ocr baslatilamadı ({e}), diger yontemler deneniyor...")

        # --- EasyOCR (Yedek) ---
        if self.ocr_backend in ("easyocr", "auto"):
            try:
                import easyocr

                self.ocr_engine = easyocr.Reader(
                    ["tr", "en"],
                    gpu=True,
                    verbose=False,
                )
                self.ocr_backend = "easyocr"
                logger.info("[Modul 3] EasyOCR baslatildi.")
                return
            except ImportError:
                if self.ocr_backend == "easyocr":
                    logger.error(
                        "[Modul 3] easyocr paketi bulunamadi. "
                        "pip install easyocr"
                    )
                    return
                logger.debug("[Modul 3] EasyOCR bulunamadi.")

        logger.warning(
            "[Modul 3] Hicbir OCR motoru baslatılamadi. "
            "Plaka metni cikarilmayacak."
        )

    # ==================================================================
    # Ana Okuma Fonksiyonu
    # ==================================================================
    def read(
        self,
        plate_crop: np.ndarray,
        apply_perspective: bool = True,
    ) -> Optional[str]:
        """
        Plaka kirpintisindan metin cikarir.

        Args:
            plate_crop: Plaka bolgesi kirpintisi (Modul 1 ciktisi).
            apply_perspective: Perspektif duzeltme uygula (True/False).

        Returns:
            Normalize edilmis plaka metni veya None.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None

        try:
            # Fast-Plate-OCR kendi on-islemesini yapar (BGR/RGB 3 kanal bekler)
            if self.ocr_backend == "fast_plate_ocr":
                input_img = plate_crop
            else:
                input_img = self._preprocess(plate_crop)

            # OCR
            raw_text = self._run_ocr(input_img)

            if raw_text:
                validated = validate_plate(raw_text)
                logger.debug(f"[Modul 3] Plaka: '{raw_text}' -> '{validated}'")
                return validated

        except Exception as e:
            logger.warning(f"[Modul 3] Plaka okuma hatasi: {e}")

        return None

    # ==================================================================
    # Goruntu On-Isleme
    # ==================================================================
    def _preprocess(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Plaka kirpintisini OCR icin hazirlar.

        Adimlar:
          1. Gri tonlamaya cevir
          2. CLAHE kontrast iyilestirme
          3. Keskinlestirme (unsharp mask)
          4. Boyutlandirma (standart yukseklik)

        Args:
            plate_crop: Ham plaka kirpintisi.

        Returns:
            On islenmis goruntu.
        """
        img = plate_crop.copy()

        # Gri tonlama
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # CLAHE kontrast iyilestirme
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Keskinlestirme
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        # Boyutlandirma (yukseklik 64px standart)
        h, w = sharpened.shape[:2]
        if h > 0:
            target_h = 64
            scale = target_h / h
            target_w = int(w * scale)
            sharpened = cv2.resize(
                sharpened,
                (target_w, target_h),
                interpolation=cv2.INTER_CUBIC,
            )

        return sharpened

    # ==================================================================
    # OCR Motor Cagirimi
    # ==================================================================
    def _run_ocr(self, processed_image: np.ndarray) -> Optional[str]:
        """
        OCR motorunu calistirarak metin cikarir.

        Args:
            processed_image: On islenmis plaka goruntusu.

        Returns:
            Ham plaka metni veya None.
        """
        if self.ocr_engine is None:
            return None

        try:
            if self.ocr_backend == "fast_plate_ocr":
                # Fast-Plate-OCR
                result = self.ocr_engine.run(processed_image)
                if result and len(result) > 0:
                    return str(result[0].plate).strip()

            elif self.ocr_backend == "easyocr":
                # EasyOCR
                results = self.ocr_engine.readtext(
                    processed_image,
                    detail=1,
                    paragraph=False,
                )
                if results:
                    # En yuksek confidence'li sonuc
                    best = max(results, key=lambda x: x[2])
                    return best[1].strip()

        except Exception as e:
            logger.warning(f"[Modul 3] OCR motor hatasi: {e}")

        return None
