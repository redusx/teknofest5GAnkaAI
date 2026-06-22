# MIMARİ TASARIM: 4 KADEMELİ KASKAT (CASCADE) MİKRO-MODELLER

## Genel Bakış

Sistem, birbiriyle ölçek ve odak noktası bakımından çatışan görevleri (bütünsel araç
kinematikleri vs. ufak kabin içi nesneler) ayrıştırmak için **4 Kademeli Kaskat
Mikro-Model** mimarisi kullanır. Her modül bağımsız olarak optimize edilebilir ve
TensorRT FP16 formatında çalıştırılır.

## Veri Akış Şeması

```
Video Karesi (1920x1080)
         │
         ▼
┌─────────────────────────┐
│  MODÜL 1: Küresel       │  YOLOv12s — TensorRT FP16
│  Bağlam ve Makro Tespit │  ~2.6 ms / kare
│                         │
│  Çıktılar:              │
│  ├─ Araç tipi + renk    │
│  ├─ Teknocan, Bilgisayar│
│  ├─ Plaka ROI (bbox)  ──┼──────────────────────┐
│  └─ Ön Cam ROI (bbox) ──┼──────┐               │
└─────────────────────────┘      │               │
         │                       ▼               ▼
         │          ┌──────────────────┐  ┌──────────────────┐
         │          │ MODÜL 2: Kabin   │  │ MODÜL 3: Plaka   │
         │          │ İçi Analiz       │  │ OCR              │
         │          │                  │  │                  │
         │          │ YOLOv12n — FP16  │  │ Fast-Plate-OCR   │
         │          │ ~1.6 ms / kare   │  │ CCT-S-V2         │
         │          │                  │  │ ~0.67 ms          │
         │          │ Çıktılar:        │  │                  │
         │          │ ├─ Şoför eylemi  │  │ Çıktı:           │
         │          │ ├─ Yolcular      │  │ └─ Plaka metni   │
         │          │ └─ Emniyet kemeri│  │    (regex valid.) │
         │          └──────────────────┘  └──────────────────┘
         │
         ▼
┌─────────────────────────┐
│  MODÜL 4: Kinematik     │  ByteTrack + Kalman Filtresi
│  Yörünge Takibi         │  CPU tabanlı (minimal maliyet)
│                         │
│  Çıktı:                 │
│  └─ Slalom tespiti      │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  POST-PROCESSOR         │
│                         │
│  ├─ Zamansal 1D NMS     │
│  ├─ Confidence toplama  │
│  ├─ ASCII normalizasyon │
│  └─ results.json üretimi│
└─────────────────────────┘
```

## Modül Detayları

### Modül 1: Küresel Bağlam ve Makro Tespit Ağı
- **Model**: YOLOv12s (Small) — TensorRT FP16
- **Girdi**: Ham video karesi (1920x1080)
- **Görevler**:
  - Araç tespiti, tip ve renk sınıflandırması
  - Teknocan ve bilgisayar nesnelerinin tespiti
  - Plaka bölgesi (Plate ROI) koordinat çıkarımı
  - Ön cam bölgesi (Windshield ROI) koordinat çıkarımı
- **Gecikme**: ~2.6 ms (T4 GPU)
- **VRAM**: ~1.2 GB

### Modül 2: Kabin İçi Davranış ve Yolcu Analizi
- **Model**: YOLOv12n (Nano) — TensorRT FP16
- **Girdi**: Ön cam kırpıntısı (Modül 1 çıktısı)
- **Görevler**:
  - Şoför eylemleri: telefon, su, sigara, esneme, arkaya bakma, etrafa bakınma
  - Emniyet kemeri ihlali
  - Yolcu konumlandırması (heuristik kural seti)
- **Gecikme**: ~1.6 ms (T4 GPU)
- **VRAM**: ~0.6 GB

### Modül 3: Plaka Okuma (OCR) Ağı
- **Model**: Fast-Plate-OCR (CCT-S-V2)
- **Girdi**: Plaka kırpıntısı + perspektif düzeltme (Modül 1 çıktısı)
- **Görevler**:
  - Plaka metni çıkarımı
  - Türkiye plaka regex doğrulaması
- **Gecikme**: ~0.67 ms (T4 GPU)
- **VRAM**: ~0.4 GB

### Modül 4: Kinematik Yörünge ve Slalom Tespiti
- **Algoritma**: ByteTrack + Genişletilmiş Kalman Filtresi (EKF)
- **Girdi**: Araç bounding box merkezleri (Modül 1 çıktısı)
- **Görevler**:
  - Araç merkez koordinatlarının zamansal takibi
  - Kalman filtresi ile yörünge düzleştirme
  - Yanal varyans analizi ile slalom tespiti
- **Gecikme**: Minimal (CPU tabanlı matris işlemi)
- **VRAM**: 0 (CPU only)

## Toplam Kaynak Tüketimi (T4 GPU)
- **VRAM**: ~3 GB / 16 GB (devasa güvenlik marjı)
- **Kare başı gecikme**: ~15 ms (tüm modüller + veri transferi)
- **10 FPS @ 5 dk video**: ~45 sn çıkarım ≪ 10 dk limit

## Koşullu Dallanma (Conditional Execution)
- Ön cam tespit edilmediğinde → Modül 2 ÇALIŞMAZ
- Plaka tespit edilmediğinde → Modül 3 ÇALIŞMAZ
- Bu sayede gereksiz hesaplama önlenir ve zaman kazanılır.
