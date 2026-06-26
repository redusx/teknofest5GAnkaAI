#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 - ANKAAI
Modul 4: Kinematik Yorunge ve Slalom Tespiti (Trajectory Tracker)

ByteTrack + Genisletilmis Kalman Filtresi (EKF) ile:
  - Arac merkez koordinatlarinin zamansal takibi
  - Kalman filtresi ile yorunge duzlestirme (smoothing)
  - Yanal varyans ve ivme analizi ile slalom tespiti

Slalom Tespit Algoritması:
  1. Arac bounding box merkez koordinatlari (x, y) zaman damgalariyla kaydedilir
  2. Kalman filtresi ile gurultu filtrelenir
  3. Duzlestirmis yorunge uzerinde yanal (x ekseni) standart sapma hesaplanir
  4. Belirli bir zaman penceresi icerisinde yuksek yanal varyans ve
     isaret degistiren ivme tespit edilirse "slalom" olarak etiketlenir
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

import numpy as np

from src.utils import sanitize_label, clamp_confidence

logger = logging.getLogger("module4_trajectory")


# ==============================================================================
# Kalman Filtresi (Basitlestirilmis 2D)
# ==============================================================================
class SimpleKalmanFilter2D:
    """
    Basitlestirilmis 2D Kalman Filtresi.

    Durum vektoru: [x, y, vx, vy] (konum + hiz)
    Olcum vektoru: [x, y] (konum)

    Arac kinematiklerini modelleyerek bounding box merkez
    koordinatlarini duzlestirir.
    """

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
    ):
        """
        Args:
            process_noise: Surecin gurultu katsayisi (Q).
            measurement_noise: Olcum gurultu katsayisi (R).
        """
        # Durum vektoru: [x, y, vx, vy]
        self.state = np.zeros(4)

        # Durum gecis matrisi (sabit hiz modeli)
        self.F = np.eye(4)
        self.F[0, 2] = 1.0  # x += vx * dt
        self.F[1, 3] = 1.0  # y += vy * dt

        # Olcum matrisi
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Kovaryans matrisleri
        self.P = np.eye(4) * 100.0
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise

        self.initialized = False

    def predict(self) -> np.ndarray:
        """Bir sonraki durumu tahmin eder."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Olcum ile durumu gunceller.

        Args:
            measurement: [x, y] olcum degerleri.

        Returns:
            Guncellenmis [x, y] konum tahmini.
        """
        if not self.initialized:
            self.state[0] = measurement[0]
            self.state[1] = measurement[1]
            self.initialized = True
            return self.state[:2].copy()

        # Tahmin adimi
        self.predict()

        # Guncelleme adimi
        z = measurement
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.state[:2].copy()


# ==============================================================================
# Yorunge Takipci ve Slalom Tespitcisi
# ==============================================================================
class TrajectoryTracker:
    """
    Arac yorunge takibi ve slalom tespiti modulu.

    Modul 1'den gelen arac bounding box merkezlerini zaman icinde
    takip eder ve Kalman filtresi ile duzlestirerek slalom davranisini
    matematiksel olarak saptar.

    Slalom Tespiti Kriterleri:
      - Yanal (x ekseni) standart sapma esik degerini asarsa
      - Ivme isaret degistirme frekansi esik degerini asarsa
      - Belirlenen zaman penceresi icerisinde yukarıdaki kosullar saglanirsa

    Attributes:
        kalman_filter: 2D Kalman filtresi.
        trajectory: Duzlestirmis yorunge noktalari.
    """

    def __init__(
        self,
        window_seconds: float = 5.0,
        lateral_std_threshold: float = 30.0,
        min_direction_changes: int = 3,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
    ):
        """
        Args:
            window_seconds: Slalom analiz penceresi suresi (saniye).
            lateral_std_threshold: Yanal standart sapma esigi (piksel).
            min_direction_changes: Minimum yon degistirme sayisi.
            process_noise: Kalman filtresi surec gurultusu.
            measurement_noise: Kalman filtresi olcum gurultusu.
        """
        self.window_seconds = window_seconds
        self.lateral_std_threshold = lateral_std_threshold
        self.min_direction_changes = min_direction_changes

        self.kalman_filter = SimpleKalmanFilter2D(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )

        # Yorunge verileri: (zaman, x_filtered, y_filtered, x_raw, y_raw)
        self.trajectory: List[Tuple[float, float, float, float, float]] = []

        # Slalom olaylari
        self._slalom_events: List[Dict[str, Any]] = []

        # Son slalom tespiti zamani (cooldown icin)
        self._last_slalom_time: float = -999.0
        self._slalom_cooldown: float = 5.0

    def reset(self):
        """Yeni bir video icin tum verileri sifirlar."""
        self.trajectory.clear()
        self._slalom_events.clear()
        self._last_slalom_time = -999.0
        self.kalman_filter = SimpleKalmanFilter2D()

    # ==================================================================
    # Yorunge Guncelleme
    # ==================================================================
    def update(
        self,
        vehicle_bbox: Optional[Tuple[int, int, int, int]],
        frame_time: float,
    ):
        """
        Yeni bir kare icin arac konumunu gunceller.

        Args:
            vehicle_bbox: Arac bounding box (x1, y1, x2, y2) veya None.
            frame_time: Karenin zamani (saniye).
        """
        if vehicle_bbox is None:
            return

        x1, y1, x2, y2 = vehicle_bbox

        # Merkez koordinatlari
        cx_raw = (x1 + x2) / 2.0
        cy_raw = (y1 + y2) / 2.0

        # Kalman filtresi ile duzlestir
        measurement = np.array([cx_raw, cy_raw])
        filtered = self.kalman_filter.update(measurement)
        cx_filtered, cy_filtered = filtered

        # Yorungeye ekle
        self.trajectory.append(
            (frame_time, cx_filtered, cy_filtered, cx_raw, cy_raw)
        )

        # Slalom analizi (yeterli veri toplandiysa)
        if len(self.trajectory) >= 10:
            self._check_slalom(frame_time)

    # ==================================================================
    # Slalom Tespit Algoritmasi
    # ==================================================================
    def _check_slalom(self, current_time: float):
        """
        Zaman penceresi icerisindeki yorungeyi analiz ederek slalom saptar.

        Algoritma:
          1. Son window_seconds saniyelik veriyi al
          2. Yanal (x) eksende standart sapma hesapla
          3. Yanal hiz isaretinin yon degistirmelerini say
          4. Her iki kosul da saglanirsa "slalom" olarak etiketle

        Args:
            current_time: Mevcut kare zamani.
        """
        # Cooldown kontrolu
        if (current_time - self._last_slalom_time) < self._slalom_cooldown:
            return

        # Zaman penceresi icerisindeki verileri filtrele
        window_start = current_time - self.window_seconds
        window_data = [
            t for t in self.trajectory if t[0] >= window_start
        ]

        if len(window_data) < 10:
            return

        # Yanal (x) eksendeki duzlestirmis koordinatlar
        x_values = np.array([t[1] for t in window_data])

        # Kriter 1: Yanal standart sapma
        lateral_std = np.std(x_values)

        # Kriter 2: Gürültüden etkilenmeyen (deadzone korumalı) türev & zigzag hesabı
        # Küçük titreşimleri (< 1.5 piksel) sıfıra yuvarlayarak sahte yön değişimlerini eler
        dx = np.diff(x_values)
        significant_dx = np.where(np.abs(dx) > 1.5, dx, 0)
        active_dirs = np.sign(significant_dx[significant_dx != 0])
        sign_changes = int(np.sum(np.abs(np.diff(active_dirs)) > 0)) if len(active_dirs) > 1 else 0

        # Slalom tespiti
        if (
            lateral_std > self.lateral_std_threshold
            and sign_changes >= self.min_direction_changes
        ):
            slalom_event = {
                "zaman_saniye": round(current_time, 2),
                "kategori": "sofor_eylemi",
                "etiket": sanitize_label("slalom"),
                "confidence_score": clamp_confidence(
                    min(1.0, lateral_std / (self.lateral_std_threshold * 2))
                ),
            }

            self._slalom_events.append(slalom_event)
            self._last_slalom_time = current_time

            logger.info(
                f"[Modul 4] SLALOM tespit edildi @ {current_time:.1f}s "
                f"(std={lateral_std:.1f}, yon_degisim={sign_changes})"
            )

    # ==================================================================
    # Görselleştirme (Teknofest Jüri Gösterimi)
    # ==================================================================
    def draw_trajectory(self, frame: Any, current_time: float):
        """
        Araç merkezinin son 5 saniyelik yörünge kuyruğunu (tail) ekrana çizer.
        Slalom tespiti varsa kırmızı zigzag uyarısı verir.
        """
        import cv2
        window_start = current_time - self.window_seconds
        pts = [
            (int(t[1]), int(t[2])) for t in self.trajectory if t[0] >= window_start
        ]
        if len(pts) < 2:
            return

        is_slalom_active = (current_time - self._last_slalom_time) < 2.5
        color = (0, 0, 255) if is_slalom_active else (255, 200, 0)

        for i in range(1, len(pts)):
            thickness = max(1, int(4 * (i / len(pts))))
            cv2.line(frame, pts[i-1], pts[i], color, thickness)

        if is_slalom_active:
            cv2.putText(
                frame, "⚠️ SLALOM IHLALI (ZIGZAG)", 
                (max(10, pts[-1][0] - 80), max(30, pts[-1][1] - 40)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

    # ==================================================================
    # Sonuc Erisicileri
    # ==================================================================
    def get_slalom_events(self) -> List[Dict[str, Any]]:
        """Tespit edilen slalom olaylarini dondurur."""
        return self._slalom_events

    def get_trajectory_stats(self) -> Dict[str, Any]:
        """Yorunge istatistiklerini dondurur."""
        if not self.trajectory:
            return {
                "toplam_nokta": 0,
                "sure": 0.0,
                "yanal_std": 0.0,
                "slalom_sayisi": len(self._slalom_events),
            }

        x_values = [t[1] for t in self.trajectory]
        return {
            "toplam_nokta": len(self.trajectory),
            "sure": self.trajectory[-1][0] - self.trajectory[0][0],
            "yanal_std": float(np.std(x_values)),
            "slalom_sayisi": len(self._slalom_events),
        }
