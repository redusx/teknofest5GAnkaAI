# TEKNOFEST 2026 – 5G & Yapay Zeka ile Akilli Yol Guvenligi

## Takim: ANKAAI

### Proje Ozeti

Bu proje, karayolu guvenligini artirmak amaciyla 5G teknolojisinin dusuk gecikme
ve yuksek bant genisligi yeteneklerini yapay zeka tabanli nesne tespitiyle
birlestiren uctan uca bir akilli yol guvenligi sistemi sunmaktadir.

### Calistirma

```bash
# 1) Docker imajini olustur
docker build -t teknofest/ankaai:latest .

# 2) Tesla T4 uzerinde calistir
docker run --rm --gpus all \
  -v <video-dosyasi>:/app/data/input/video.mp4 \
  -v <run-klasoru>:/app/data/output \
  teknofest/ankaai:latest
```

### Dizin Yapisi

```
ankaai/
├── Dockerfile
├── requirements.txt
├── main.py
├── README.md
├── src/
│   ├── __init__.py
│   ├── predict.py
│   ├── utils.py
│   ├── vehicle_detector.py
│   └── action_detector.py
├── tools/
│   └── data_merger.py
├── weights/
│   └── best_model.pt
└── configs/
    └── class_config.yaml
```

### Giris / Cikis Yollari

| Yon    | Yol                             |
|--------|---------------------------------|
| Girdi  | `/app/data/input/video.mp4`     |
| Cikti  | `/app/data/output/results.json` |
| Model  | `/app/weights/best_model.pt`    |

### Cikti JSON Semasi

```json
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
```
