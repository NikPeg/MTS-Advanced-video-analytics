# Быстрый старт HW2

## Установка зависимостей

```bash
pip install torch torchvision opencv-python numpy matplotlib scipy tqdm jupyter
```

## Запуск

1. Откройте Jupyter Notebook:
```bash
cd HW2
jupyter notebook video_segmentation.ipynb
```

2. Выполните все ячейки последовательно (Cell → Run All)

3. Результаты будут сохранены в папку `results/`

## Ожидаемое время выполнения

- На GPU: ~3-5 минут
- На CPU: ~8-12 минут

## Результаты

После выполнения в папке `results/` появятся:
- `first_frame.png` - Первый кадр видео
- `original_masks.png` - Исходные маски
- `metrics_comparison.png` - Графики метрик
- `comparison_best.png` - Сравнение до/после
- `metrics.json` - Численные результаты
- `summary.json` - Итоговая сводка

## Примечания

- Используется 100 кадров из видео для ускорения
- Модель DeepLabV3 загружается автоматически при первом запуске
- Для полного видео увеличьте `max_frames` в ячейке загрузки

