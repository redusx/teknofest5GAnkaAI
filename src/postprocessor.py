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
        bbox_aspect_ratio: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Video boyunca biriktirilen arac tespitlerinden nihai bilgiyi uretir.

        Arac tipi icin en-boy orani evristigi destekli cogunluk oyu,
        Renk icin Parlaklik Agirlikli Oylama (Luminance-Weighted Voting),
        Plaka icin Zamansal Konsensus Oylama (Temporal Consensus Voting) kullanilir.
        """
        import re

        result = {
            "tip": "sedan",
            "plaka": "tespit edilemedi",
            "renk": "",
            "confidence_score": 0.0,
        }

        all_confidences: List[float] = []

        # --- Arac Tipi: En-Boy Orani Evristigi + Cogunluk Oyu ---
        if type_votes:
            best_type = max(type_votes, key=lambda t: len(type_votes[t]))
            conf_list = type_votes[best_type]
            median_conf = float(np.median(conf_list))

            # FTR Mimarisi §Aspect-Ratio Heuristics (Tepe kamerasi SUV tavan basikligi uyarisi)
            if best_type == "sedan" and bbox_aspect_ratio < 1.45:
                alt_candidates = [t for t in ["suv", "minibus", "panelvan", "pickup"] if t in type_votes]
                if alt_candidates:
                    best_type = max(alt_candidates, key=lambda t: len(type_votes[t]))
                    logger.info(f"[PostProcessor] Aspect-ratio evristigi (w/h={bbox_aspect_ratio:.2f} < 1.45) ile Sedan -> {best_type} secildi.")

            result["tip"] = sanitize_label(best_type)
            all_confidences.append(median_conf)

        # --- Renk: Parlaklik Agirlikli Oylama (Luminance-Weighted Voting) ---
        if color_votes:
            # Her renk icin toplam agirlik skoru
            color_scores = {c: sum(scores) for c, scores in color_votes.items()}
            best_color = max(color_scores, key=lambda c: color_scores[c])
            conf_list = color_votes[best_color]
            median_conf = float(np.median(conf_list))
            result["renk"] = sanitize_label(best_color)
            all_confidences.append(median_conf)
            logger.debug(f"[PostProcessor] Renk skoru: {best_color} (toplam agirlik: {color_scores[best_color]:.2f})")

        # --- Plaka: Zamansal Konsensus Oylama (Temporal Consensus OCR) ---
        if plate_detections:
            cleaned_pool = []
            for text, conf in plate_detections:
                clean_text = re.sub(r'[^A-Z0-9]', '', str(text).upper())
                if len(clean_text) >= 2:
                    cleaned_pool.append((clean_text, float(conf)))

            if cleaned_pool:
                # >= 6 karakterli okumalara oncelik ver (eksik plaka kirpintilarini ele)
                long_plates = [p for p in cleaned_pool if len(p[0]) >= 6]
                active_pool = long_plates if long_plates else cleaned_pool

                plate_votes = {}
                for text, conf in active_pool:
                    plate_votes[text] = plate_votes.get(text, 0.0) + conf

                best_plate_raw = max(plate_votes, key=lambda p: plate_votes[p])
                best_plate_conf = min(1.0, plate_votes[best_plate_raw] / max(1, len(active_pool)))

                result["plaka"] = validate_plate(best_plate_raw)
                all_confidences.append(best_plate_conf)
            else:
                best_plate = max(plate_detections, key=lambda x: x[1])
                result["plaka"] = validate_plate(best_plate[0])
                all_confidences.append(float(best_plate[1]))

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
