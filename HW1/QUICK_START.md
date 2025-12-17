# Быстрый старт - HW1

## Установка

```bash
# Основные зависимости
pip install -r HW1/requirements.txt

# Опционально: decord для задачи 7 (аппаратное декодирование)
# Если pip install не работает, попробуйте:
# conda install -c conda-forge decord
# или
# pip install decord2
```

## Быстрая проверка

### 1. Тест базовой функциональности

```bash
python HW1/test_pipeline.py <путь_к_видео>
```

Это проверит задачи 1 и 2 (декодер и Dataset).

### 2. Запуск всех задач

```bash
python HW1/video_pipeline.py --video <путь_к_видео> --all
```

### 3. Запуск отдельной задачи

```bash
python HW1/video_pipeline.py --video <путь_к_видео> --task <1-10>
```

## Что проверять

После запуска проверьте:

1. **Консольный вывод** - должны быть метрики и результаты
2. **Папка `HW1/results/`** - должны создаться графики и файлы результатов
3. **Отсутствие ошибок** - все задачи должны выполниться без критических ошибок

## Примеры команд

```bash
# Тест базовой функциональности
python HW1/test_pipeline.py sample.mp4

# Задача 3: Параллельная загрузка
python HW1/video_pipeline.py --video sample.mp4 --task 3

# Задача 4: Профилирование (затем смотрите в TensorBoard)
python HW1/video_pipeline.py --video sample.mp4 --task 4
tensorboard --logdir=HW1/results/profiler

# Все задачи
python HW1/video_pipeline.py --video sample.mp4 --all
```

## Ожидаемые результаты

- ✅ Все задачи выполняются без ошибок
- ✅ Создаются графики в `HW1/results/`
- ✅ Метрики выводятся в консоль
- ✅ FPS увеличивается с оптимизациями
- ✅ GPU быстрее CPU (если доступна GPU)

Подробная инструкция: см. `HW1/README.md`

