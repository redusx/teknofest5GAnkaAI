#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Son Islem Motoru (Post-Processor)

Tum modullerden gelen ham tespitleri yarisma standartlarina uygun
JSON ciktisina donusturur.

Sorumluluklar:
  1. Zamansal 1D NMS (Olu Bolge) — ayni etiketi tekrar raporlama onlenir
  2. Arac bilgisi icin Zaman-Agirlikli Medyan confidence hesaplama
  3. ASCII normalizasyon ve etiket sterilizasyonu
  4. Konsolide results.json uretimi
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np

from src.utils import (
    sanitize_label,
    validate_plate,
    clamp_confidence,
    LABEL_TO_CATEGORY,
)

logger = logging.getLogger("postprocessor")


class PostProcessor:
    """
    Tum modullerin ciktilarini toplayarak yarisma formatinda JSON uretir.

    Attributes:
        cooldown_seconds: Ayni etiket icin tekrar raporlama bekleme suresi.
        min_confidence: Raporlanacak minimum guven esigi.
    """

    def __init__(
        self,
        cooldown_seconds: float = 3.0,
        min_confidence: float = 0.25,
    ):
        """
        Args:
            cooldown_seconds: Zamansal 1D NMS olu bolge suresi (saniye).
                Ayni etiketin bu sure icerisindeki tekrarlari filtrelenir.
            min_confidence: Minimum kabul edilebilir guven skoru.
        """
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence

    # ==================================================================
    # Zamansal 1D NMS (Non-Maximum Suppression)
    # ==================================================================
    def apply_temporal_nms(
        self, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Zamansal Olu Bolge algoritmasi ile ardisik tekrarlari filtreler.

        Algoritma:
          1. Bir olay tespit edildiginde kaydedilir.
          2. Ayni etiket, cooldown_seconds suresi icerisinde tekrar
             tespit edilirse yeni kayit ACILMAZ.
          3. Cooldown suresi gecerse ve tekrar tespit edilirse,
             yeni bir olay olarak kaydedilir.

        Args:
            detections: Ham tespit listesi (zaman_saniye siralı).

        Returns:
            Filtrelenmis tespit listesi.
        """
        if not detections:
            return []

        # Zamana gore sirala
        sorted_dets = sorted(detections, key=lambda d: d["zaman_saniye"])

        filtered: List[Dict[str, Any]] = []
        last_times: Dict[str, float] = {}

        for det in sorted_dets:
            etiket = det["etiket"]
            zaman = det["zaman_saniye"]
            conf = det.get("confidence_score", 0.0)

            # Minimum guven kontrolu
            if conf < self.min_confidence:
                continue

            # Cooldown kontrolu
            if etiket in last_times:
                elapsed = zaman - last_times[etiket]
                if elapsed < self.cooldown_seconds:
                    continue

            filtered.append(det)
            last_times[etiket] = zaman

        logger.info(
            f"Zamansal NMS: {len(detections)} -> {len(filtered)} tespit"
        )
        return filtered

    # ==================================================================
    # Arac Bilgisi Toplayicisi
    # ==================================================================
    def aggregate_vehicle_info(
        self,
        type_votes: Dict[str, List[float]],
        color_votes: Dict[str, List[float]],
        plate_detections: List[tuple],
    ) -> Dict[str, Any]:
        """
        Video boyunca biriktirilen arac tespitlerinden nihai bilgiyi uretir.

        Arac tipi ve renk icin cogunluk oyu (majority voting) kullanilir.
        Genel confidence_score icin Zaman-Agirlikli Medyan hesaplanir.

        Args:
            type_votes: {arac_tipi: [confidence_listesi]} esleme tablosu.
            color_votes: {renk: [confidence_listesi]} esleme tablosu.
            plate_detections: [(plaka_metni, confidence)] listesi.

        Returns:
            Yarisma formatinda arac_bilgisi sozlugu:
            {
                "tip": "sedan",
                "plaka": "34ABC123",
                "renk": "beyaz",
                "confidence_score": 0.94
            }
        """
        result = {
            "tip": "sedan",
            "plaka": "tespit edilemedi",
            "renk": "",
            "confidence_score": 0.0,
        }

        all_confidences: List[float] = []

        # --- Arac Tipi: En cok oy alan tip ---
        if type_votes:
            best_type = max(
                type_votes, key=lambda t: len(type_votes[t])
            )
            conf_list = type_votes[best_type]
            # Zaman-Agirlikli Medyan (outlier'ları dışlar)
            median_conf = float(np.median(conf_list))
            result["tip"] = sanitize_label(best_type)
            all_confidences.append(median_conf)

        # --- Renk: En cok oy alan renk ---
        if color_votes:
            best_color = max(
                color_votes, key=lambda c: len(color_votes[c])
            )
            conf_list = color_votes[best_color]
            median_conf = float(np.median(conf_list))
            result["renk"] = sanitize_label(best_color)
            all_confidences.append(median_conf)

        # --- Plaka: En yuksek confidence'li tespit ---
        if plate_detections:
            best_plate = max(plate_detections, key=lambda x: x[1])
            result["plaka"] = validate_plate(best_plate[0])
            all_confidences.append(best_plate[1])

        # --- Genel confidence_score ---
        if all_confidences:
            result["confidence_score"] = clamp_confidence(
                float(np.median(all_confidences))
            )

        return result

    # ==================================================================
    # Konsolide JSON Cikti Uretimi
    # ==================================================================
    def build_output(
        self,
        video_id: str,
        vehicle_info: Dict[str, Any],
        detections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Yarisma standartlarina uygun konsolide JSON ciktisini olusturur.

        FTR Model Kilavuz §4.3 formatina birebir uyar:
        {
            "video_id": "video.mp4",
            "arac_bilgisi": {...},
            "tespitler": [...]
        }

        Args:
            video_id: Video dosya adi.
            vehicle_info: Arac bilgisi sozlugu.
            detections: Filtrelenmis tespit listesi.

        Returns:
            JSON-uyumlu sozluk.
        """
        # Arac bilgisini sterilize et
        arac_bilgisi = {
            "tip": sanitize_label(vehicle_info.get("tip", "sedan")),
            "plaka": validate_plate(vehicle_info.get("plaka", "")),
            "renk": sanitize_label(vehicle_info.get("renk", "")),
            "confidence_score": round(
                float(vehicle_info.get("confidence_score", 0.0)), 4
            ),
        }

        # Tespitleri sterilize et
        formatted_detections = []
        for det in detections:
            etiket = sanitize_label(det.get("etiket", ""))
            kategori = det.get(
                "kategori", LABEL_TO_CATEGORY.get(etiket, "")
            )

            formatted_detections.append(
                {
                    "zaman_saniye": round(
                        float(det.get("zaman_saniye", 0.0)), 2
                    ),
                    "kategori": kategori,
                    "etiket": etiket,
                    "confidence_score": round(
                        float(det.get("confidence_score", 0.0)), 4
                    ),
                }
            )

        output = {
            "video_id": video_id,
            "arac_bilgisi": arac_bilgisi,
            "tespitler": formatted_detections,
        }

        logger.info(
            f"JSON cikti olusturuldu: "
            f"arac={arac_bilgisi['tip']}, "
            f"plaka={arac_bilgisi['plaka']}, "
            f"tespit_sayisi={len(formatted_detections)}"
        )

        return output
