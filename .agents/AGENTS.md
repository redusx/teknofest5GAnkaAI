# ANKAAI Proje Özel Kuralları (Project Rules)

## 1. Donanım Optimizasyonu (YOLO Eğitimleri)
* **GPU / RAM Profili:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM) & 16 GB Sistem RAM.
* **Eğitim Parametreleri:** YOLOv12 / YOLO11 model eğitimlerinde (`train_models.py`), VRAM kapasitesini tam kullanmak ve OOM hatasını önlemek için varsayılan olarak **`batch=32`** ve **`workers=4`** tercih edilmelidir. `cache=False` ve `amp=True` sabit tutulmalıdır.

## 2. Windows Konsol ve Encoding Uyumluluğu
* **Terminal Kodlaması:** Windows konsol varsayılan kodlaması (cp1254) nedeniyle betiklerin en üstünde mutlaka stdout yapılandırması yapılmalıdır:
  ```python
  import sys
  if hasattr(sys.stdout, "reconfigure"):
      sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
  ```
* **Log Mesajları:** `logger.info` veya `print` çıktıları içerisinde `→`, `—` gibi ASCII dışı özel ok/tire sembolleri yerine standart `->` ve `-` kullanılmalıdır.

## 3. Ultralytics `data.yaml` Dosya Yolları
* Veri seti hazırlama betiklerinde (`prepare_datasets.py`, `prepare_cabin_datasets.py`), oluşturulan `data.yaml` dosyasındaki `path` anahtarı görece (`.`) değil, mutlaka **mutlak posix yolu** olarak yazılmalıdır:
  ```python
  path: {MERGED_DIR.as_posix()}
  ```
