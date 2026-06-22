#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Yardimci Fonksiyonlar (Utility Functions)

Goruntu on-isleme, cikti formatlama, etiket sanitizasyonu ve
plaka regex dogrulamasi gibi ortak isleri icerir.
"""

import re
from typing import Dict, List, Optional, Any


# ==============================================================================
# Turkce Karakter -> ASCII Donusum Tablosu
# ==============================================================================
TR_CHAR_MAP: Dict[str, str] = {
    "ç": "c", "Ç": "C",
    "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I",
    "ö": "o", "Ö": "O",
    "ş": "s", "Ş": "S",
    "ü": "u", "Ü": "U",
}

# Turkiye plaka regex deseni (FTR Model Kilavuz §Tablo1)
PLATE_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"((\s?[a-zA-Z]\s?)(\d{4,5})"
    r"|(\s?[a-zA-Z]{2}\s?)(\d{3,4})"
    r"|(\s?[a-zA-Z]{3}\s?)(\d{2,3}))$"
)

# Gecerli degerler (FTR Model Kilavuz §Tablo1 ve §Tablo2)
VALID_VEHICLE_TYPES = [
    "sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon"
]

VALID_COLORS = [
    "beyaz", "siyah", "gri", "kirmizi", "mavi",
    "sari", "yesil", "turuncu", "kahverengi"
]

VALID_DRIVER_ACTIONS = [
    "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
    "telefonla_konusma", "slalom", "etrafa_bakinma", "emniyet_kemeri_ihlali"
]

VALID_OBJECTS = ["teknocan", "bilgisayar"]

VALID_PASSENGERS = ["arka_koltuk_1", "arka_koltuk_2", "on_koltuk"]

# Kategori esleme
LABEL_TO_CATEGORY: Dict[str, str] = {}
for label in VALID_DRIVER_ACTIONS:
    LABEL_TO_CATEGORY[label] = "sofor_eylemi"
for label in VALID_OBJECTS:
    LABEL_TO_CATEGORY[label] = "nesneler"
for label in VALID_PASSENGERS:
    LABEL_TO_CATEGORY[label] = "yolcular"


def sanitize_label(text: str) -> str:
    """
    Etiketi ASCII-safe ve kucuk harfli formata donusturur.

    Kurallar (FTR Model Kilavuz §1, §5):
      - Tum harfler kucuk olacak
      - Turkce karakterler ASCII karsiliklarina donusturulecek

    Args:
        text: Donusturulecek etiket metni.

    Returns:
        ASCII-safe, kucuk harfli etiket.
    """
    for tr_char, ascii_char in TR_CHAR_MAP.items():
        text = text.replace(tr_char, ascii_char)
    return text.lower().strip()


def validate_plate(plate_text: str) -> str:
    """
    Plaka metnini dogrular ve normalize eder.

    FTR Model Kilavuz §Tablo1:
      Regex: ^(0[1-9]|[1-7][0-9]|8[01])(...)$
      Ornek: 34ABC123

    Args:
        plate_text: Ham plaka metni.

    Returns:
        Normalize edilmis plaka metni (bosluklari kaldirilmis, buyuk harf).
    """
    # Bosluklari kaldir ve buyuk harfe cevir
    normalized = plate_text.replace(" ", "").upper()

    # Regex dogrulama
    if PLATE_REGEX.match(normalized):
        return normalized
    else:
        # Regex eslesmese bile plaka metnini dondur, log'da uyari ver
        return normalized


def preprocess_frame(frame, target_size: tuple = (640, 640)):
    """
    Goruntu on-isleme adimlari.

    Args:
        frame: OpenCV ile okunan goruntu (numpy array).
        target_size: Hedef boyut (genislik, yukseklik).

    Returns:
        On-islenmis goruntu.
    """
    import cv2
    import numpy as np

    if frame is None:
        return None

    # Boyutlandirma (YOLO standart girdi boyutu)
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    return resized


def format_output(
    video_id: str,
    vehicle_info: Optional[Dict[str, Any]],
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Yarisma standartlarina uygun JSON semasini hazirlar.

    FTR Model Kilavuz §4.3 - Konsolide Cikti Formati:
    {
        "video_id": "video.mp4",
        "arac_bilgisi": {
            "tip": "sedan",
            "plaka": "34ABC123",
            "renk": "beyaz",
            "confidence_score": 0.94
        },
        "tespitler": [
            {
                "zaman_saniye": 14.5,
                "kategori": "sofor_eylemi",
                "etiket": "telefonla_konusma",
                "confidence_score": 0.89
            }
        ]
    }

    Args:
        video_id: Video dosya adi.
        vehicle_info: Arac bilgisi sozlugu veya None.
        detections: Tespit listesi.

    Returns:
        Yarisma formatinda JSON-uyumlu sozluk.
    """
    output = {
        "video_id": video_id,
    }

    # Arac bilgisi
    if vehicle_info:
        output["arac_bilgisi"] = {
            "tip": sanitize_label(vehicle_info.get("tip", "sedan")),
            "plaka": validate_plate(vehicle_info.get("plaka", "")),
            "renk": sanitize_label(vehicle_info.get("renk", "")),
            "confidence_score": float(vehicle_info.get("confidence_score", 0.0)),
        }
    else:
        output["arac_bilgisi"] = {
            "tip": "sedan",
            "plaka": "",
            "renk": "",
            "confidence_score": 0.0,
        }

    # Tespitler
    formatted_detections = []
    for det in detections:
        etiket = sanitize_label(det.get("etiket", ""))
        kategori = det.get("kategori", LABEL_TO_CATEGORY.get(etiket, ""))

        formatted_detections.append({
            "zaman_saniye": round(float(det.get("zaman_saniye", 0.0)), 2),
            "kategori": kategori,
            "etiket": etiket,
            "confidence_score": round(float(det.get("confidence_score", 0.0)), 4),
        })

    output["tespitler"] = formatted_detections

    return output


def get_category_for_class(class_name: str) -> str:
    """
    Sinif adina gore JSON cikti kategorisini dondurur.

    Args:
        class_name: Sinif adi (orn: "telefonla_konusma").

    Returns:
        Kategori adi (orn: "sofor_eylemi").
    """
    return LABEL_TO_CATEGORY.get(class_name, "bilinmeyen")


def clamp_confidence(score: float) -> float:
    """
    Guven skorunu 0.0 - 1.0 araligina sinirlar.

    Args:
        score: Ham guven skoru.

    Returns:
        Sinirlandirilmis guven skoru.
    """
    return max(0.0, min(1.0, float(score)))
