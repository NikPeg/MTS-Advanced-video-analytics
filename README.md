# Advanced Video Analytics

Курс по продвинутой видеоаналитике: от обработки видеосигналов до распознавания действий и временной локализации событий.

## 📚 Структура проекта

```
.
├── Theory/                    # Лекционные материалы
│   ├── 1-Lecture.md          # Видео как сигнал
│   ├── 3-Ot-analiza-kadrov-k-analizu-dvizheniya.md
│   ├── 4-Videodetekciya-kak-prostranstvenno-vremennaya-zadacha.md
│   ├── 5-Videosegmentaciya-kak-vremennaya-zadacha.md
│   ├── 6-Vremennaya-lokalizaciya-sobytij-i-video-anomalii.md
│   ├── 1-code_example.py     # Примеры работы с CFR/VFR
│   └── 4-code-example.py     # Примеры детекции и трекинга
├── HW1.md                     # Семинар: Пайплайн данных
├── HW2.md                     # Домашнее задание: Видеосегментация
└── requirements.txt           # Зависимости проекта
```

## 📖 Лекции

### 1. [Видео как сигнал](Theory/1-Lecture.md)
Основы работы с видеоданными:
- Дискретизация видео по пространству и времени
- CFR vs VFR (Constant/Variable Frame Rate)
- Цветовые пространства (RGB, Y'CbCr, BT.709)
- Кодеки и компрессия (H.264, H.265, AV1)
- Метрики качества видео (PSNR, SSIM, VMAF)
- Декодирование и пайплайны обработки

**Примеры кода:** [1-code_example.py](Theory/1-code_example.py) — работа с CFR/VFR видео

### 2. [От анализа кадров к анализу движения](Theory/3-Ot-analiza-kadrov-k-analizu-dvizheniya.md)
Переход от статических изображений к видеопотокам:
- Пространственно-временные фильтры
- 3D CNN (C3D, R(2+1)D, I3D)
- SlowFast: двухпотоковая архитектура
- Video Vision Transformers (TimeSformer, VideoSwin)
- Предобучение и transfer learning

### 3. [Видеодетекция как пространственно-временная задача](Theory/4-Videodetekciya-kak-prostranstvenno-vremennaya-zadacha.md)
Детекция объектов в видео:
- Временная когерентность и дрейф
- Tracking-by-detection
- Фильтр Калмана
- Многообъектный трекинг (MOT)
- DeepSORT, ByteTrack, OC-SORT
- Метрики трекинга (MOTA, MOTP, HOTA, IDF1)

**Примеры кода:** [4-code-example.py](Theory/4-code-example.py) — детекция и трекинг с YOLO и ByteTrack

### 4. [Видеосегментация как временная задача](Theory/5-Videosegmentaciya-kak-vremennaya-zadacha.md)
Сегментация объектов в видео:
- Временная согласованность масок
- Оптический поток и warping
- Space-Time Memory Networks (STM)
- Cross-frame attention
- Метрики сегментации (IoU, Boundary F-score, J&F)

### 5. [Временная локализация событий и видео-аномалии](Theory/6-Vremennaya-lokalizaciya-sobytij-i-video-anomalii.md)
Локализация событий во времени:
- Temporal Convolutional Networks (TCN)
- LSTM и рекуррентные модели
- Трансформеры для временной локализации
- Метрики локализации (temporal IoU, mAP)
- Детекция видео-аномалий

## 💻 Практические задания

### [Семинар: Пайплайн данных](HW1.md)
Практика построения эффективного видеопайплайна:
- Параллельное чтение и сэмплирование кадров
- Реализация Dataset для видеоклипов
- Профилирование latency и оптимизация throughput
- Работа с PyAV и PyTorch

### [Домашнее задание: Видеосегментация](HW2.md)
Практическая работа по видео-сегментации:
- Временная стабилизация масок (anti-flicker)
- Semi-supervised Video Object Segmentation (VOS)
- Перенос масок через оптический поток
- Оценка качества сегментации

**Реализация**: [HW2/](HW2/) - Вариант A (Anti-Flicker система)

## 🔧 Примеры кода

### [Пример 1: Работа с CFR/VFR видео](Theory/1-code_example.py)
Демонстрирует:
- Создание видео с фиксированной частотой кадров (CFR)
- Создание видео с переменной частотой кадров (VFR)
- Использование FFmpeg для обработки видео

**Использование:**
```bash
python Theory/1-code_example.py <input_video>
```

### [Пример 2: Детекция и трекинг](Theory/4-code-example.py)
Демонстрирует:
- Детекция объектов с помощью YOLO
- Многообъектный трекинг с ByteTrack
- Визуализация траекторий объектов
- Сохранение результата в видео

**Требования:** ultralytics, opencv-python

## 📋 Дополнительные материалы

В папке `Theory/` также находятся:
- `1-Seminar.md` — материалы первого семинара
- `2-Seminar.md` — материалы второго семинара
- `5-Seminar_5.md` — материалы пятого семинара
- `1-Конспект.md` — конспект первой лекции

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Основные библиотеки

- **PyAV** — декодирование видео
- **PyTorch** — глубокое обучение
- **OpenCV** — обработка изображений
- **ultralytics** — YOLO модели
- **FFmpeg** — обработка видео (требуется установка отдельно)

### Запуск примеров

```bash
# Пример работы с CFR/VFR
python Theory/1-code_example.py video.mp4

# Пример детекции и трекинга
python Theory/4-code-example.py
```

## 📝 Темы курса

1. **Основы видео** — дискретизация, кодеки, цветовые пространства
2. **Распознавание действий** — 3D CNN, трансформеры, архитектуры
3. **Видеодетекция** — детекция объектов, трекинг, MOT
4. **Видеосегментация** — сегментация объектов, временная согласованность
5. **Локализация событий** — временная локализация, детекция аномалий

## 🔗 Полезные ссылки

- [PyAV Documentation](https://pyav.org/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 Лицензия

Материалы курса предназначены для образовательных целей.

