"""
Семинар №2: Собираем быстрый пайплайн данных
Реализация всех 10 задач по построению эффективного видеопайплайна
"""

import os
import time
import queue
import threading
import multiprocessing
import json
import hashlib
from collections import deque
from typing import List, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import av
import cv2
from PIL import Image
import torchvision.transforms.v2 as transforms_v2
import matplotlib.pyplot as plt
from tqdm import tqdm

# decord будет импортирован лениво только в задаче 7, чтобы избежать конфликта с av
DECORD_AVAILABLE = None  # Будет установлено при первом использовании


# ============================================================================
# Задача 1: Базовый декодер видеокадров
# ============================================================================

def check_cfr_vfr(filename: str) -> Tuple[str, dict]:
    """
    Проверяет, является ли видео CFR (Constant Frame Rate) или VFR (Variable Frame Rate).
    Основано на задании 1 из семинара 1.
    
    Returns:
        ('CFR' или 'VFR', словарь с метриками)
    """
    container = av.open(filename)
    video_stream = container.streams.video[0]
    
    # Получаем метаданные (используем правильные атрибуты PyAV)
    avg_frame_rate = video_stream.average_rate
    time_base = video_stream.time_base
    
    # Читаем первые 30 кадров для анализа интервалов
    pts_list = []
    frame_count = 0
    
    for frame in container.decode(video_stream):
        if frame.pts is not None:
            pts_list.append(frame.pts)
        frame_count += 1
        if frame_count >= 30:
            break
    
    container.close()
    
    # Анализируем интервалы между кадрами
    metrics = {
        'avg_frame_rate': float(avg_frame_rate) if avg_frame_rate else None,
        'time_base': float(time_base),
    }
    
    if len(pts_list) > 1:
        # Вычисляем интервалы в секундах
        intervals = [(pts_list[i+1] - pts_list[i]) * float(time_base) 
                    for i in range(len(pts_list)-1)]
        metrics['intervals'] = intervals
        metrics['avg_interval'] = np.mean(intervals)
        metrics['std_interval'] = np.std(intervals)
        metrics['cv'] = metrics['std_interval'] / metrics['avg_interval'] if metrics['avg_interval'] > 0 else 0
        
        # Определяем CFR/VFR по коэффициенту вариации
        if metrics['cv'] > 0.1:
            result = 'VFR'
        else:
            result = 'CFR'
    else:
        result = 'Unknown'
        metrics['cv'] = None
    
    return result, metrics

def read_clip(filename: str, start: int = 0, num_frames: int = 16, stride: int = 2, verbose: bool = False) -> np.ndarray:
    """
    Читает клип из видео с помощью PyAV.
    Улучшенная версия с правильной обработкой временных меток (PTS).
    
    Args:
        filename: путь к видеофайлу
        start: начальный кадр (индекс)
        num_frames: количество кадров для чтения
        stride: шаг между кадрами
        verbose: выводить ли информацию о прочитанных кадрах
    
    Returns:
        numpy.ndarray формы (T, H, W, 3) в формате RGB
    """
    container = av.open(filename)
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames if video_stream.frames else 0
    
    # Получаем параметры потока
    time_base = video_stream.time_base
    duration = float(video_stream.duration * time_base) if video_stream.duration else 0
    fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
    
    # Проверяем границы
    if total_frames > 0 and start >= total_frames:
        container.close()
        # Возвращаем пустой массив правильной формы (будет дополнен padding)
        return np.array([]).reshape(0, 1080, 1920, 3)
    
    frames = []
    frame_indices = []
    pts_list = []  # Список временных меток для анализа
    
    # Вычисляем целевые индексы кадров
    target_indices = [start + i * stride for i in range(num_frames)]
    
    # Переходим к начальному кадру используя правильный seek по времени
    # (подход из семинара 1, задание 2)
    try:
        if start > 0 and duration > 0:
            # Вычисляем время начала в секундах
            start_time = start / fps if fps > 0 else 0
            # Seek использует timestamp в базовых единицах времени потока
            seek_pts = int(start_time * time_base.denominator / time_base.numerator)
            container.seek(seek_pts, stream=video_stream)
    except Exception as e:
        if verbose:
            print(f"Warning: seek failed, starting from beginning: {e}")
    
    frame_count = 0
    last_pts = None
    
    # Читаем кадры с учетом временных меток
    for frame in container.decode(video_stream):
        current_pts = frame.pts
        
        # Проверяем, нужен ли нам этот кадр
        if frame_count in target_indices:
            if len(frames) < num_frames:
                # Конвертируем в RGB numpy array
                img = frame.to_ndarray(format='rgb24')
                frames.append(img)
                frame_indices.append(frame_count)
                pts_list.append(current_pts)
            else:
                break
        
        frame_count += 1
        
        # Останавливаемся если прошли все нужные кадры
        if len(frames) >= num_frames:
            break
        
        # Защита от бесконечного цикла
        if total_frames > 0 and frame_count >= total_frames:
            break
    
    container.close()
    
    if len(frames) == 0:
        # Возвращаем пустой массив (будет дополнен padding)
        return np.array([]).reshape(0, 1080, 1920, 3)
    
    result = np.stack(frames, axis=0)  # (T, H, W, 3)
    
    if verbose:
        print(f"Read {len(frames)} frames. Indices: {frame_indices}")
        print(f"Actual FPS: {fps:.2f}")
        if len(pts_list) > 1:
            # Анализ временных интервалов (для проверки CFR/VFR)
            intervals = [(pts_list[i+1] - pts_list[i]) * float(time_base) 
                        for i in range(len(pts_list)-1)]
            if intervals:
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                print(f"Frame intervals: avg={avg_interval:.4f}s, std={std_interval:.4f}s")
                if std_interval / avg_interval > 0.1:
                    print("  → VFR detected (variable frame rate)")
                else:
                    print("  → CFR detected (constant frame rate)")
    
    return result


# ============================================================================
# Задача 2: Реализация Dataset для видеоклипов
# ============================================================================

class VideoDataset(Dataset):
    """Dataset для загрузки видеоклипов."""
    
    def __init__(
        self,
        video_files: List[str],
        clip_len: int = 16,
        stride: int = 2,
        transform: Optional[Callable] = None
    ):
        self.video_files = video_files
        self.clip_len = clip_len
        self.stride = stride
        self.transform = transform
        
        # Предвычисляем количество клипов в каждом видео
        self.clip_counts = []
        for video_file in video_files:
            container = av.open(video_file)
            video_stream = container.streams.video[0]
            total_frames = video_stream.frames
            container.close()
            
            # Максимальный стартовый индекс для клипа
            max_start = max(0, total_frames - (clip_len * stride))
            num_clips = max(1, (max_start // stride) + 1)
            self.clip_counts.append(num_clips)
    
    def __len__(self):
        return sum(self.clip_counts)
    
    def __getitem__(self, idx):
        # Определяем, из какого видео брать клип
        video_idx = 0
        clip_idx = idx
        
        for i, count in enumerate(self.clip_counts):
            if clip_idx < count:
                video_idx = i
                break
            clip_idx -= count
        
        video_file = self.video_files[video_idx]
        
        # Вычисляем стартовый кадр
        start_frame = clip_idx * self.stride
        
        # Читаем клип
        clip = read_clip(video_file, start=start_frame, num_frames=self.clip_len, stride=self.stride, verbose=False)
        
        # Убеждаемся, что клип имеет правильную длину (padding если нужно)
        if len(clip) == 0:
            # Если не удалось прочитать кадры, пропускаем этот клип
            # Возвращаем нулевой тензор правильной формы
            if self.transform:
                # Создаем нулевой тензор формы (T, C, H, W)
                clip = torch.zeros(self.clip_len, 3, 224, 224, dtype=torch.float32)
            else:
                # Создаем нулевой numpy array и конвертируем
                clip = np.zeros((self.clip_len, 1080, 1920, 3), dtype=np.uint8)
        elif len(clip) < self.clip_len:
            # Дополняем последним кадром
            if len(clip) > 0:
                last_frame = clip[-1]
                padding_frames = [last_frame] * (self.clip_len - len(clip))
                clip = np.concatenate([clip, np.stack(padding_frames, axis=0)], axis=0)
            else:
                # Если нет кадров, создаем нулевой массив
                clip = np.zeros((self.clip_len, clip.shape[1] if len(clip.shape) > 1 else 1080, 
                               clip.shape[2] if len(clip.shape) > 2 else 1920, 3), dtype=np.uint8)
        
        # Обрезаем если больше нужного
        clip = clip[:self.clip_len]
        
        # Применяем трансформации
        if self.transform:
            # Transform ожидает 3D (H, W, C), а клип 4D (T, H, W, C)
            # Применяем transform к каждому кадру отдельно
            transformed_frames = []
            for frame in clip:
                # frame имеет форму (H, W, 3)
                frame_tensor = self.transform(frame)
                transformed_frames.append(frame_tensor)
            # Собираем обратно в тензор формы (T, C, H, W)
            clip = torch.stack(transformed_frames, dim=0)
        else:
            # Если нет transform, конвертируем в тензор
            clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float()  # (T, H, W, 3) -> (T, C, H, W)
        
        return clip


# ============================================================================
# Задача 3: Параллельная загрузка данных
# ============================================================================

def measure_throughput(dataloader: DataLoader, num_iterations: int = 10) -> Tuple[float, float]:
    """
    Измеряет throughput (кадров/с) для DataLoader.
    
    Returns:
        (mean_fps, std_fps)
    """
    times = []
    frame_counts = []
    
    # Вычисляем общее количество батчей для progress bar
    total_batches = len(dataloader)
    
    for iteration in tqdm(range(num_iterations), desc="  Iterations", leave=False):
        start_time = time.time()
        frames_processed = 0
        
        for batch in tqdm(dataloader, desc=f"    Epoch {iteration+1}/{num_iterations}", 
                         total=total_batches, leave=False, unit="batch"):
            frames_processed += batch.shape[0] * batch.shape[1]  # batch_size * clip_len
        
        elapsed = time.time() - start_time
        fps = frames_processed / elapsed if elapsed > 0 else 0
        
        times.append(elapsed)
        frame_counts.append(frames_processed)
    
    total_frames = sum(frame_counts)
    total_time = sum(times)
    mean_fps = total_frames / total_time if total_time > 0 else 0
    
    return mean_fps, np.std([fc / t for fc, t in zip(frame_counts, times)])


def task3_parallel_loading(video_files: List[str], output_dir: str = "HW1/results", force_rerun: bool = False):
    """Задача 3: Измерение throughput при разных num_workers.
    
    Args:
        video_files: список путей к видеофайлам
        output_dir: директория для сохранения результатов
        force_rerun: если True, перезапускает измерения даже если есть сохраненные результаты
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем уникальный ключ для чекпоинта на основе параметров
    config_hash = hashlib.md5(
        (str(sorted(video_files)) + "16_2_4_5").encode()
    ).hexdigest()[:8]
    checkpoint_file = f"{output_dir}/task3_checkpoint_{config_hash}.json"
    
    # Пытаемся загрузить сохраненные результаты
    if not force_rerun and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                saved_data = json.load(f)
                results = [(r['workers'], r['mean_fps'], r['std_fps']) for r in saved_data['results']]
                print(f"✓ Loaded checkpoint from {checkpoint_file}")
                print(f"  Previous results: {len(results)} configurations")
                
                # Проверяем, что все конфигурации есть
                num_workers_list = [1, 2, 4, 8]
                if len(results) == len(num_workers_list):
                    # Выводим результаты и строим график
                    workers, fps_values, _ = zip(*results)
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(workers, fps_values, 'o-', linewidth=2, markersize=8)
                    plt.xlabel('Number of Workers', fontsize=12)
                    plt.ylabel('Throughput (FPS)', fontsize=12)
                    plt.title('Throughput vs Number of Workers', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f"{output_dir}/task3_throughput.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"\nResults saved to {output_dir}/task3_throughput.png")
                    
                    max_fps = max(fps_values)
                    saturation_workers = next((w for w, f in zip(workers, fps_values) if f >= 0.95 * max_fps), workers[-1])
                    print(f"Saturation point: {saturation_workers} workers")
                    
                    return results
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}. Running fresh measurements...")
    
    # Выполняем измерения
    transform = transforms_v2.Compose([
        transforms_v2.ToImage(),
        transforms_v2.Resize((224, 224)),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VideoDataset(video_files, clip_len=16, stride=2, transform=transform)
    
    num_workers_list = [1, 2, 4, 8]
    results = []
    
    for num_workers in tqdm(num_workers_list, desc="Testing num_workers", unit="config"):
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=num_workers,
            shuffle=False
        )
        
        print(f"\nTesting with num_workers={num_workers}...")
        mean_fps, std_fps = measure_throughput(dataloader, num_iterations=5)
        results.append((num_workers, mean_fps, std_fps))
        print(f"  ✓ Mean FPS: {mean_fps:.2f} ± {std_fps:.2f}")
        
        # Сохраняем промежуточные результаты после каждой конфигурации
        checkpoint_data = {
            'results': [
                {'workers': w, 'mean_fps': float(fps), 'std_fps': float(std)}
                for w, fps, std in results
            ],
            'timestamp': time.time()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    # Построение графика
    workers, fps_values, _ = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(workers, fps_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Workers', fontsize=12)
    plt.ylabel('Throughput (FPS)', fontsize=12)
    plt.title('Throughput vs Number of Workers', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/task3_throughput.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to {output_dir}/task3_throughput.png")
    print(f"Checkpoint saved to {checkpoint_file}")
    
    # Определение точки насыщения
    max_fps = max(fps_values)
    saturation_workers = next((w for w, f in zip(workers, fps_values) if f >= 0.95 * max_fps), workers[-1])
    print(f"Saturation point: {saturation_workers} workers")
    
    return results


# ============================================================================
# Задача 4: Профилирование этапов пайплайна
# ============================================================================

def task4_profiling(video_files: List[str], output_dir: str = "HW1/results"):
    """Задача 4: Профилирование с torch.profiler."""
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms_v2.Compose([
        transforms_v2.ToImage(),
        transforms_v2.Resize((224, 224)),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VideoDataset(video_files, clip_len=16, stride=2, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    
    # Заглушка модели
    model = lambda x: x.mean()
    
    # Профилирование
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/profiler"),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for i, batch in enumerate(dataloader):
            if isinstance(batch, torch.Tensor):
                batch = batch.to('cuda' if torch.cuda.is_available() else 'cpu')
            result = model(batch)
            prof.step()
            
            if i >= 10:  # Ограничиваем количество итераций
                break
    
    print(f"Profiling results saved to {output_dir}/profiler")
    print("View with: tensorboard --logdir=" + output_dir + "/profiler")
    
    # Анализ соотношения времени
    key_averages = prof.key_averages()
    
    decode_time = 0
    prep_time = 0
    infer_time = 0
    
    for event in key_averages:
        name = event.key.lower()
        # cpu_time_total уже в микросекундах, делим на 1000 для миллисекунд
        cpu_time_ms = event.cpu_time_total / 1000 if hasattr(event, 'cpu_time_total') else 0
        
        if 'read_clip' in name or 'decode' in name:
            decode_time += cpu_time_ms
        elif 'transform' in name or 'resize' in name or 'normalize' in name:
            prep_time += cpu_time_ms
        elif 'mean' in name or 'model' in name:
            infer_time += cpu_time_ms
    
    total_time = decode_time + prep_time + infer_time
    if total_time > 0:
        print(f"\nTime distribution:")
        print(f"  Decode: {decode_time:.2f} ms ({100*decode_time/total_time:.1f}%)")
        print(f"  Preprocess: {prep_time:.2f} ms ({100*prep_time/total_time:.1f}%)")
        print(f"  Infer: {infer_time:.2f} ms ({100*infer_time/total_time:.1f}%)")
        print(f"\nRatio L_dec:L_prep:L_inf = {decode_time:.2f}:{prep_time:.2f}:{infer_time:.2f}")


# ============================================================================
# Задача 5: Prefetch и pinned memory
# ============================================================================

def task5_prefetch_pinned_memory(video_files: List[str], output_dir: str = "HW1/results", force_rerun: bool = False):
    """Задача 5: Сравнение с/без prefetch и pinned memory.
    
    Args:
        video_files: список путей к видеофайлам
        output_dir: директория для сохранения результатов
        force_rerun: если True, перезапускает измерения даже если есть сохраненные результаты
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем уникальный ключ для чекпоинта
    config_hash = hashlib.md5(
        (str(sorted(video_files)) + "16_2_4_10").encode()
    ).hexdigest()[:8]
    checkpoint_file = f"{output_dir}/task5_checkpoint_{config_hash}.json"
    
    # Пытаемся загрузить сохраненные результаты
    if not force_rerun and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                saved_data = json.load(f)
                results = [(r['name'], r['mean_fps'], r['std_fps'], r['jitter']) for r in saved_data['results']]
                print(f"✓ Loaded checkpoint from {checkpoint_file}")
                print(f"  Previous results: {len(results)} configurations")
                
                # Выводим результаты
                print("\n" + "="*60)
                print("Comparison Results:")
                print("="*60)
                for name, mean_fps, std_fps, jitter in results:
                    print(f"{name:20s} | FPS: {mean_fps:7.2f} ± {std_fps:6.2f} | Jitter: {jitter:.4f}")
                
                return results
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}. Running fresh measurements...")
    
    transform = transforms_v2.Compose([
        transforms_v2.ToImage(),
        transforms_v2.Resize((224, 224)),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VideoDataset(video_files, clip_len=16, stride=2, transform=transform)
    
    configs = [
        ("Baseline", {"prefetch_factor": None, "pin_memory": False}),
        ("Prefetch", {"prefetch_factor": 2, "pin_memory": False}),
        ("Pinned Memory", {"prefetch_factor": None, "pin_memory": True}),
        ("Both", {"prefetch_factor": 2, "pin_memory": True}),
    ]
    
    results = []
    
    for name, config in tqdm(configs, desc="Testing configurations", unit="config"):
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=4,
            shuffle=False,
            **config
        )
        
        print(f"\nTesting {name}...")
        
        fps_list = []
        latencies = []
        
        for iteration in tqdm(range(10), desc=f"  {name} iterations", leave=False):
            start = time.time()
            for batch in dataloader:
                if torch.cuda.is_available() and config.get("pin_memory", False):
                    batch = batch.to('cuda', non_blocking=True)
                pass  # Просто загружаем данные
            elapsed = time.time() - start
            fps_list.append(len(dataset) * 16 / elapsed)  # примерный FPS
            latencies.append(elapsed)
        
        mean_fps = np.mean(fps_list)
        std_fps = np.std(fps_list)
        jitter = np.std(latencies) / np.mean(latencies) if np.mean(latencies) > 0 else 0
        
        results.append((name, mean_fps, std_fps, jitter))
        print(f"  Mean FPS: {mean_fps:.2f} ± {std_fps:.2f}")
        print(f"  Jitter: {jitter:.4f}")
        
        # Сохраняем промежуточные результаты
        checkpoint_data = {
            'results': [
                {'name': n, 'mean_fps': float(fps), 'std_fps': float(std), 'jitter': float(j)}
                for n, fps, std, j in results
            ],
            'timestamp': time.time()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    # Вывод результатов
    print("\n" + "="*60)
    print("Comparison Results:")
    print("="*60)
    for name, mean_fps, std_fps, jitter in results:
        print(f"{name:20s} | FPS: {mean_fps:7.2f} ± {std_fps:6.2f} | Jitter: {jitter:.4f}")
    
    print(f"\nCheckpoint saved to {checkpoint_file}")
    return results


# ============================================================================
# Задача 6: Pipeline overlap
# ============================================================================

class PipelineOverlap:
    """Реализация перекрытия декодирования и инференса."""
    
    def __init__(self, video_file: str, model: Callable, batch_size: int = 4):
        self.video_file = video_file
        self.model = model
        self.batch_size = batch_size
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
    def decode_thread(self):
        """Поток декодирования."""
        container = av.open(self.video_file)
        video_stream = container.streams.video[0]
        
        frame_count = 0
        batch = []
        
        for frame in container.decode(video_stream):
            img = frame.to_ndarray(format='rgb24')
            batch.append(img)
            
            if len(batch) >= self.batch_size:
                self.frame_queue.put(np.stack(batch))
                batch = []
            
            frame_count += 1
            if frame_count >= 100:  # Ограничение для теста
                break
        
        if batch:
            self.frame_queue.put(np.stack(batch))
        
        self.frame_queue.put(None)  # Сигнал окончания
        container.close()
    
    def infer_thread(self):
        """Поток инференса."""
        while True:
            batch = self.frame_queue.get()
            if batch is None:
                self.result_queue.put(None)
                break
            
            # Имитация инференса
            result = self.model(torch.from_numpy(batch).float())
            self.result_queue.put(result)
    
    def run_sequential(self):
        """Последовательное выполнение."""
        start = time.time()
        
        container = av.open(self.video_file)
        video_stream = container.streams.video[0]
        
        frame_count = 0
        for frame in container.decode(video_stream):
            img = frame.to_ndarray(format='rgb24')
            batch_tensor = torch.from_numpy(img).float().unsqueeze(0)
            result = self.model(batch_tensor)
            frame_count += 1
            if frame_count >= 100:
                break
        
        container.close()
        elapsed = time.time() - start
        return elapsed / frame_count if frame_count > 0 else 0
    
    def run_overlapped(self):
        """Перекрытое выполнение."""
        start = time.time()
        
        decode_thread = threading.Thread(target=self.decode_thread)
        infer_thread = threading.Thread(target=self.infer_thread)
        
        decode_thread.start()
        infer_thread.start()
        
        results = []
        while True:
            result = self.result_queue.get()
            if result is None:
                break
            results.append(result)
        
        decode_thread.join()
        infer_thread.join()
        
        elapsed = time.time() - start
        return elapsed / len(results) if results else 0


def task6_pipeline_overlap(video_file: str, output_dir: str = "HW1/results"):
    """Задача 6: Сравнение последовательного и перекрытого выполнения."""
    os.makedirs(output_dir, exist_ok=True)
    
    model = lambda x: x.mean()
    pipeline = PipelineOverlap(video_file, model)
    
    print("Running sequential pipeline...")
    seq_latency = pipeline.run_sequential()
    print(f"  Average latency: {seq_latency*1000:.2f} ms")
    
    print("Running overlapped pipeline...")
    overlap_latency = pipeline.run_overlapped()
    print(f"  Average latency: {overlap_latency*1000:.2f} ms")
    
    speedup = seq_latency / overlap_latency if overlap_latency > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    
    return seq_latency, overlap_latency, speedup


# ============================================================================
# Задача 7: Аппаратное декодирование
# ============================================================================

def read_clip_decord(filename: str, start: int = 0, num_frames: int = 16, stride: int = 2, gpu: bool = False) -> np.ndarray:
    """Чтение клипа с помощью decord (GPU/CPU). Ленивый импорт для избежания конфликта с av."""
    global DECORD_AVAILABLE
    
    # Ленивый импорт decord только когда он действительно нужен
    if DECORD_AVAILABLE is None:
        try:
            import decord
            DECORD_AVAILABLE = True
        except ImportError:
            DECORD_AVAILABLE = False
            raise ImportError("decord not available. Install with: pip install decord or conda install -c conda-forge decord")
    
    if not DECORD_AVAILABLE:
        raise ImportError("decord not available")
    
    import decord  # Импортируем здесь, чтобы избежать конфликта при загрузке модуля
    
    ctx = decord.gpu(0) if gpu else decord.cpu(0)
    vr = decord.VideoReader(filename, ctx=ctx)
    
    frame_indices = [start + i * stride for i in range(num_frames)]
    frames = vr.get_batch(frame_indices).asnumpy()
    
    return frames


def task7_hardware_decoding(video_file: str, output_dir: str = "HW1/results"):
    """Задача 7: Сравнение PyAV (CPU) и decord (GPU)."""
    os.makedirs(output_dir, exist_ok=True)
    
    num_frames = 100
    results = []
    
    # PyAV (CPU)
    print("Testing PyAV (CPU)...")
    times = []
    for _ in range(5):
        start = time.time()
        read_clip(video_file, start=0, num_frames=num_frames, stride=1)
        times.append((time.time() - start) * 1000)  # мс
    
    avg_time_pyav = np.mean(times)
    fps_pyav = num_frames / (avg_time_pyav / 1000)
    results.append(("PyAV (CPU)", avg_time_pyav, fps_pyav))
    print(f"  Average time: {avg_time_pyav:.2f} ms")
    print(f"  FPS: {fps_pyav:.2f}")
    
    # decord (CPU) - ленивый импорт
    global DECORD_AVAILABLE
    if DECORD_AVAILABLE is None:
        try:
            import decord
            DECORD_AVAILABLE = True
        except ImportError:
            DECORD_AVAILABLE = False
    
    if DECORD_AVAILABLE:
        print("\nTesting decord (CPU)...")
        times = []
        for _ in range(5):
            start = time.time()
            read_clip_decord(video_file, start=0, num_frames=num_frames, stride=1, gpu=False)
            times.append((time.time() - start) * 1000)
        
        avg_time_decord_cpu = np.mean(times)
        fps_decord_cpu = num_frames / (avg_time_decord_cpu / 1000)
        results.append(("decord (CPU)", avg_time_decord_cpu, fps_decord_cpu))
        print(f"  Average time: {avg_time_decord_cpu:.2f} ms")
        print(f"  FPS: {fps_decord_cpu:.2f}")
        
        # decord (GPU) - если доступно
        if torch.cuda.is_available():
            print("\nTesting decord (GPU)...")
            times = []
            for _ in range(5):
                start = time.time()
                read_clip_decord(video_file, start=0, num_frames=num_frames, stride=1, gpu=True)
                times.append((time.time() - start) * 1000)
            
            avg_time_decord_gpu = np.mean(times)
            fps_decord_gpu = num_frames / (avg_time_decord_gpu / 1000)
            results.append(("decord (GPU)", avg_time_decord_gpu, fps_decord_gpu))
            print(f"  Average time: {avg_time_decord_gpu:.2f} ms")
            print(f"  FPS: {fps_decord_gpu:.2f}")
    
    # Вывод таблицы
    print("\n" + "="*60)
    print("Decoding Performance Comparison:")
    print("="*60)
    print(f"{'Method':<20} | {'Time (ms)':<15} | {'FPS':<10}")
    print("-" * 60)
    for method, avg_time, fps in results:
        print(f"{method:<20} | {avg_time:>13.2f} | {fps:>8.2f}")
    
    return results


# ============================================================================
# Задача 8: Оптимизация препроцессинга
# ============================================================================

def task8_gpu_preprocessing(video_files: List[str], output_dir: str = "HW1/results"):
    """Задача 8: Сравнение CPU и GPU препроцессинга."""
    os.makedirs(output_dir, exist_ok=True)
    
    video_file = video_files[0]  # Используем первое видео
    
    # CPU препроцессинг
    cpu_transform = transforms_v2.Compose([
        transforms_v2.ToImage(),
        transforms_v2.Resize((224, 224)),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # GPU препроцессинг
    gpu_transform = transforms_v2.Compose([
        transforms_v2.ToImage(),
        transforms_v2.Resize((224, 224), antialias=True),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = lambda x: x.mean()
    
    # CPU вариант
    print("Testing CPU preprocessing...")
    decode_times = []
    prep_times = []
    infer_times = []
    
    for i in range(10):
        # Декодирование
        decode_start = time.time()
        clip = read_clip(video_file, start=i*32, num_frames=16, stride=2, verbose=False)
        if len(clip) == 0:
            continue
        decode_time = time.time() - decode_start
        
        # Препроцессинг
        prep_start = time.time()
        # Применяем transform к каждому кадру (clip имеет форму (T, H, W, 3))
        processed_frames = []
        for frame_idx in range(clip.shape[0]):
            frame = clip[frame_idx]  # (H, W, 3) numpy array uint8
            processed_frame = cpu_transform(frame)
            processed_frames.append(processed_frame)
        processed = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
        prep_time = time.time() - prep_start
        
        # Инференс
        infer_start = time.time()
        result = model(processed)
        infer_time = time.time() - infer_start
        
        decode_times.append(decode_time * 1000)
        prep_times.append(prep_time * 1000)
        infer_times.append(infer_time * 1000)
    
    cpu_decode = np.mean(decode_times)
    cpu_prep = np.mean(prep_times)
    cpu_infer = np.mean(infer_times)
    
    print(f"  Decode: {cpu_decode:.2f} ms")
    print(f"  Preprocess: {cpu_prep:.2f} ms")
    print(f"  Infer: {cpu_infer:.2f} ms")
    
    # GPU вариант
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device != 'cpu':
        print(f"\nTesting GPU preprocessing (device: {device})...")
        decode_times_gpu = []
        prep_times_gpu = []
        infer_times_gpu = []
        
        for i in range(10):
            # Декодирование
            decode_start = time.time()
            clip = read_clip(video_file, start=i*32, num_frames=16, stride=2, verbose=False)
            if len(clip) == 0:
                continue
            decode_time = time.time() - decode_start
            
            # Препроцессинг
            prep_start = time.time()
            # Применяем transform к каждому кадру (clip имеет форму (T, H, W, 3))
            processed_frames = []
            for frame_idx in range(clip.shape[0]):
                frame = clip[frame_idx]  # (H, W, 3) numpy array uint8
                processed_frame = gpu_transform(frame)
                if device == 'cuda':
                    processed_frame = processed_frame.cuda()
                elif device == 'mps':
                    processed_frame = processed_frame.to('mps')
                processed_frames.append(processed_frame)
            processed = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
            prep_time = time.time() - prep_start
            
            # Инференс
            infer_start = time.time()
            result = model(processed)
            infer_time = time.time() - infer_start
            
            decode_times_gpu.append(decode_time * 1000)
            prep_times_gpu.append(prep_time * 1000)
            infer_times_gpu.append(infer_time * 1000)
        
        gpu_decode = np.mean(decode_times_gpu)
        gpu_prep = np.mean(prep_times_gpu)
        gpu_infer = np.mean(infer_times_gpu)
        
        print(f"  Decode: {gpu_decode:.2f} ms")
        print(f"  Preprocess: {gpu_prep:.2f} ms")
        print(f"  Infer: {gpu_infer:.2f} ms")
        
        # График
        categories = ['Decode', 'Preprocess', 'Infer']
        cpu_times = [cpu_decode, cpu_prep, cpu_infer]
        gpu_times = [gpu_decode, gpu_prep, gpu_infer]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, cpu_times, width, label='CPU', alpha=0.8)
        bars2 = ax.bar(x + width/2, gpu_times, width, label='GPU', alpha=0.8)
        
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title('CPU vs GPU Pipeline Stages', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(f"{output_dir}/task8_gpu_preprocessing.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to {output_dir}/task8_gpu_preprocessing.png")
        
        return {
            'cpu': (cpu_decode, cpu_prep, cpu_infer),
            'gpu': (gpu_decode, gpu_prep, gpu_infer)
        }
    else:
        print("\nGPU not available, skipping GPU preprocessing test.")
        return {'cpu': (cpu_decode, cpu_prep, cpu_infer)}


# ============================================================================
# Задача 9: Измерение стабильности FPS
# ============================================================================

def task9_fps_stability(video_files: List[str], output_dir: str = "HW1/results", force_rerun: bool = False):
    """Задача 9: Измерение стабильности FPS.
    
    Args:
        video_files: список путей к видеофайлам
        output_dir: директория для сохранения результатов
        force_rerun: если True, перезапускает измерения даже если есть сохраненные результаты
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем уникальный ключ для чекпоинта
    config_hash = hashlib.md5(
        (str(sorted(video_files)) + "16_2_2_4_8_1_2_4_100").encode()
    ).hexdigest()[:8]
    checkpoint_file = f"{output_dir}/task9_checkpoint_{config_hash}.json"
    
    # Пытаемся загрузить сохраненные результаты
    if not force_rerun and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                saved_data = json.load(f)
                results = [(r['batch_size'], r['prefetch_factor'], r['mean_fps'], r['cv']) 
                           for r in saved_data['results']]
                print(f"✓ Loaded checkpoint from {checkpoint_file}")
                print(f"  Previous results: {len(results)} configurations")
                
                # Строим графики и выводим результаты
                batch_sizes = [2, 4, 8]
                
                # График FPS
                fig, ax = plt.subplots(figsize=(12, 6))
                for batch_size in batch_sizes:
                    fps_values = [fps for bs, pf, fps, cv in results if bs == batch_size]
                    prefetch_values = [pf for bs, pf, fps, cv in results if bs == batch_size]
                    if fps_values:
                        ax.plot(prefetch_values, fps_values, 'o-', label=f'Batch={batch_size}', linewidth=2)
                ax.set_xlabel('Prefetch Factor', fontsize=12)
                ax.set_ylabel('FPS', fontsize=12)
                ax.set_title('FPS Stability Analysis', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/task9_fps_stability.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                # График CV
                fig, ax = plt.subplots(figsize=(12, 6))
                for batch_size in batch_sizes:
                    cv_values = [cv for bs, pf, fps, cv in results if bs == batch_size]
                    prefetch_values = [pf for bs, pf, fps, cv in results if bs == batch_size]
                    if cv_values:
                        ax.plot(prefetch_values, cv_values, 'o-', label=f'Batch={batch_size}', linewidth=2)
                ax.axhline(y=0.05, color='r', linestyle='--', label='Stability threshold (CV=0.05)')
                ax.set_xlabel('Prefetch Factor', fontsize=12)
                ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
                ax.set_title('FPS Stability (CV)', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/task9_fps_cv.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                # Определение стабильных конфигураций
                stable_configs = [(bs, pf, fps, cv) for bs, pf, fps, cv in results if cv < 0.05]
                print(f"\nStable configurations (CV < 0.05): {len(stable_configs)}")
                for bs, pf, fps, cv in stable_configs:
                    print(f"  Batch={bs}, Prefetch={pf}: FPS={fps:.2f}, CV={cv:.4f}")
                
                print(f"\nResults saved to {output_dir}/task9_fps_stability.png and task9_fps_cv.png")
                return results
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}. Running fresh measurements...")
    
    transform = transforms_v2.Compose([
        transforms_v2.ToImage(),
        transforms_v2.Resize((224, 224)),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    batch_sizes = [2, 4, 8]
    prefetch_factors = [1, 2, 4]
    
    results = []
    total_configs = len(batch_sizes) * len(prefetch_factors)
    config_count = 0
    
    for batch_size in batch_sizes:
        for prefetch_factor in prefetch_factors:
            config_count += 1
            dataset = VideoDataset(video_files, clip_len=16, stride=2, transform=transform)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=4,
                prefetch_factor=prefetch_factor,
                pin_memory=True
            )
            
            fps_list = []
            
            pbar = tqdm(enumerate(dataloader), total=min(100, len(dataloader)), 
                       desc=f"  Batch={batch_size}, Prefetch={prefetch_factor} ({config_count}/{total_configs})",
                       leave=False)
            
            for i, batch in pbar:
                if i >= 100:  # 100 итераций
                    break
                
                start = time.time()
                # Имитация обработки
                _ = batch.mean()
                elapsed = time.time() - start
                
                frames_in_batch = batch.shape[0] * batch.shape[1]
                fps = frames_in_batch / elapsed if elapsed > 0 else 0
                fps_list.append(fps)
                
                # Обновляем описание progress bar
                if fps_list:
                    current_fps = np.mean(fps_list)
                    pbar.set_postfix({'FPS': f'{current_fps:.2f}'})
            
            if fps_list:
                mean_fps = np.mean(fps_list)
                std_fps = np.std(fps_list)
                cv = std_fps / mean_fps if mean_fps > 0 else float('inf')
                
                results.append((batch_size, prefetch_factor, mean_fps, cv))
                
                print(f"  ✓ Batch={batch_size}, Prefetch={prefetch_factor}: "
                      f"FPS={mean_fps:.2f}, CV={cv:.4f} {'✓' if cv < 0.05 else '✗'}")
                
                # Сохраняем промежуточные результаты
                checkpoint_data = {
                    'results': [
                        {'batch_size': bs, 'prefetch_factor': pf, 'mean_fps': float(fps), 'cv': float(cv)}
                        for bs, pf, fps, cv in results
                    ],
                    'timestamp': time.time()
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
    
    # График FPS
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for batch_size in batch_sizes:
        fps_values = [fps for bs, pf, fps, cv in results if bs == batch_size]
        prefetch_values = [pf for bs, pf, fps, cv in results if bs == batch_size]
        ax.plot(prefetch_values, fps_values, 'o-', label=f'Batch={batch_size}', linewidth=2)
    
    ax.set_xlabel('Prefetch Factor', fontsize=12)
    ax.set_ylabel('FPS', fontsize=12)
    ax.set_title('FPS Stability Analysis', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/task9_fps_stability.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # График CV
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for batch_size in batch_sizes:
        cv_values = [cv for bs, pf, fps, cv in results if bs == batch_size]
        prefetch_values = [pf for bs, pf, fps, cv in results if bs == batch_size]
        ax.plot(prefetch_values, cv_values, 'o-', label=f'Batch={batch_size}', linewidth=2)
    
    ax.axhline(y=0.05, color='r', linestyle='--', label='Stability threshold (CV=0.05)')
    ax.set_xlabel('Prefetch Factor', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_title('FPS Stability (CV)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/task9_fps_cv.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Определение стабильных конфигураций
    stable_configs = [(bs, pf, fps, cv) for bs, pf, fps, cv in results if cv < 0.05]
    
    print(f"\nStable configurations (CV < 0.05): {len(stable_configs)}")
    for bs, pf, fps, cv in stable_configs:
        print(f"  Batch={bs}, Prefetch={pf}: FPS={fps:.2f}, CV={cv:.4f}")
    
    print(f"\nResults saved to {output_dir}/task9_fps_stability.png and task9_fps_cv.png")
    print(f"Checkpoint saved to {checkpoint_file}")
    
    return results


# ============================================================================
# Задача 10: Финальное задание - мини-RT пайплайн
# ============================================================================

class RealTimePipeline:
    """Near-real-time пайплайн с двухуровневой очередью."""
    
    def __init__(
        self,
        video_source: str,
        model: Callable,
        frame_queue_size: int = 30,
        clip_queue_size: int = 5,
        clip_len: int = 16,
        stride: int = 2
    ):
        self.video_source = video_source
        self.model = model
        self.frame_queue_size = frame_queue_size
        self.clip_queue_size = clip_queue_size
        self.clip_len = clip_len
        self.stride = stride
        
        self.frame_queue = queue.Queue(maxsize=frame_queue_size)
        self.clip_queue = queue.Queue(maxsize=clip_queue_size)
        
        self.running = False
        self.stats = {
            'fps': deque(maxlen=100),
            'latencies': deque(maxlen=100),
            'frame_times': deque(maxlen=100)
        }
    
    def decode_thread(self):
        """Поток декодирования кадров."""
        if self.video_source.startswith('rtsp://') or self.video_source.startswith('http://'):
            cap = cv2.VideoCapture(self.video_source)
        else:
            container = av.open(self.video_source)
            video_stream = container.streams.video[0]
            cap = None
        
        frame_buffer = deque(maxlen=self.clip_len * self.stride)
        
        try:
            if cap is None:
                # PyAV для файлов
                for frame in container.decode(video_stream):
                    if not self.running:
                        break
                    
                    img = frame.to_ndarray(format='rgb24')
                    frame_buffer.append(img)
                    
                    if len(frame_buffer) >= self.clip_len * self.stride:
                        # Формируем клип
                        clip_frames = [frame_buffer[i] for i in range(0, len(frame_buffer), self.stride)][:self.clip_len]
                        clip = np.stack(clip_frames, axis=0)
                        
                        try:
                            self.clip_queue.put(clip, timeout=0.1)
                        except queue.Full:
                            pass  # Пропускаем если очередь полна
            else:
                # OpenCV для потоков
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_buffer.append(frame_rgb)
                    
                    if len(frame_buffer) >= self.clip_len * self.stride:
                        clip_frames = [frame_buffer[i] for i in range(0, len(frame_buffer), self.stride)][:self.clip_len]
                        clip = np.stack(clip_frames, axis=0)
                        
                        try:
                            self.clip_queue.put(clip, timeout=0.1)
                        except queue.Full:
                            pass
        finally:
            if cap:
                cap.release()
            if 'container' in locals():
                container.close()
    
    def process_thread(self):
        """Поток обработки клипов."""
        transform = transforms_v2.Compose([
            transforms_v2.ToImage(),
            transforms_v2.Resize((224, 224)),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        while self.running:
            try:
                clip = self.clip_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                # Препроцессинг
                # Применяем transform к каждому кадру отдельно
                processed_frames = []
                for frame_idx in range(clip.shape[0]):
                    frame = clip[frame_idx]  # (H, W, 3) numpy array uint8
                    processed_frame = transform(frame)
                    processed_frames.append(processed_frame)
                processed = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
                
                if device == 'cuda':
                    processed = processed.cuda()
                elif device == 'mps':
                    processed = processed.to('mps')
                
                # Инференс
                result = self.model(processed)
                
                latency = time.time() - start_time
                
                # Обновление статистики
                self.stats['latencies'].append(latency)
                fps = self.clip_len / latency if latency > 0 else 0
                self.stats['fps'].append(fps)
                
            except queue.Empty:
                continue
    
    def run(self, duration: float = 30.0):
        """Запуск пайплайна на указанное время."""
        self.running = True
        
        decode_thread = threading.Thread(target=self.decode_thread, daemon=True)
        process_thread = threading.Thread(target=self.process_thread, daemon=True)
        
        decode_thread.start()
        process_thread.start()
        
        time.sleep(duration)
        self.running = False
        
        decode_thread.join(timeout=5)
        process_thread.join(timeout=5)
    
    def get_stats(self):
        """Получение статистики."""
        if not self.stats['fps']:
            return None
        
        fps_array = np.array(self.stats['fps'])
        latencies_array = np.array(self.stats['latencies'])
        
        mean_fps = np.mean(fps_array)
        p95_latency = np.percentile(latencies_array, 95) * 1000  # мс
        
        # Jitter как стандартное отклонение латентности
        jitter = np.std(latencies_array) / np.mean(latencies_array) if np.mean(latencies_array) > 0 else 0
        
        return {
            'mean_fps': mean_fps,
            'p95_latency_ms': p95_latency,
            'jitter': jitter,
            'fps_history': list(fps_array),
            'latency_history': list(latencies_array * 1000)  # мс
        }


def task10_realtime_pipeline(video_file: str, output_dir: str = "HW1/results"):
    """Задача 10: Финальный near-real-time пайплайн."""
    os.makedirs(output_dir, exist_ok=True)
    
    model = lambda x: x.mean()
    
    pipeline = RealTimePipeline(
        video_file,
        model,
        frame_queue_size=30,
        clip_queue_size=5,
        clip_len=16,
        stride=2
    )
    
    print("Running real-time pipeline for 30 seconds...")
    pipeline.run(duration=30.0)
    
    stats = pipeline.get_stats()
    
    if stats:
        print("\n" + "="*60)
        print("Real-Time Pipeline Statistics:")
        print("="*60)
        print(f"Mean FPS: {stats['mean_fps']:.2f}")
        print(f"P95 Latency: {stats['p95_latency_ms']:.2f} ms")
        print(f"Jitter: {stats['jitter']:.4f}")
        
        # Графики
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # FPS over time
        axes[0].plot(stats['fps_history'], linewidth=1, alpha=0.7)
        axes[0].axhline(y=stats['mean_fps'], color='r', linestyle='--', label=f'Mean: {stats["mean_fps"]:.2f}')
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('FPS', fontsize=12)
        axes[0].set_title('FPS Over Time', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Latency over time
        axes[1].plot(stats['latency_history'], linewidth=1, alpha=0.7)
        axes[1].axhline(y=stats['p95_latency_ms'], color='r', linestyle='--', 
                       label=f'P95: {stats["p95_latency_ms"]:.2f} ms')
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Latency (ms)', fontsize=12)
        axes[1].set_title('Latency Over Time', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/task10_realtime_stats.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to {output_dir}/task10_realtime_stats.png")
        
        return stats
    else:
        print("No statistics collected.")
        return None


# ============================================================================
# Мини-ДЗ: Offline vs Near-Real-Time режимы
# ============================================================================

def minihw_offline_vs_realtime(video_file: str, output_dir: str = "HW1/results"):
    """Сравнение offline и near-real-time режимов."""
    os.makedirs(output_dir, exist_ok=True)
    
    model = lambda x: x.mean()
    
    # Offline режим
    print("Testing offline mode...")
    transform = transforms_v2.Compose([
        transforms_v2.ToImage(),
        transforms_v2.Resize((224, 224)),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VideoDataset([video_file], clip_len=16, stride=2, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, prefetch_factor=2, pin_memory=True)
    
    offline_fps_list = []
    offline_latencies = []
    
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= 50:
            break
        
        iter_start = time.time()
        result = model(batch)
        iter_latency = time.time() - iter_start
        
        frames = batch.shape[0] * batch.shape[1]
        fps = frames / iter_latency if iter_latency > 0 else 0
        
        offline_fps_list.append(fps)
        offline_latencies.append(iter_latency)
    
    offline_total_time = time.time() - start_time
    offline_mean_fps = np.mean(offline_fps_list)
    offline_p95_latency = np.percentile(offline_latencies, 95) * 1000
    offline_jitter = np.std(offline_latencies) / np.mean(offline_latencies) if np.mean(offline_latencies) > 0 else 0
    
    print(f"  Mean FPS: {offline_mean_fps:.2f}")
    print(f"  P95 Latency: {offline_p95_latency:.2f} ms")
    print(f"  Jitter: {offline_jitter:.4f}")
    
    # Near-Real-Time режим
    print("\nTesting near-real-time mode...")
    pipeline = RealTimePipeline(video_file, model, clip_len=16, stride=2)
    pipeline.run(duration=10.0)
    
    realtime_stats = pipeline.get_stats()
    
    if realtime_stats:
        print(f"  Mean FPS: {realtime_stats['mean_fps']:.2f}")
        print(f"  P95 Latency: {realtime_stats['p95_latency_ms']:.2f} ms")
        print(f"  Jitter: {realtime_stats['jitter']:.4f}")
        
        # Таблица сравнения
        print("\n" + "="*60)
        print("Offline vs Near-Real-Time Comparison:")
        print("="*60)
        print(f"{'Metric':<20} | {'Offline':<15} | {'Near-RT':<15}")
        print("-" * 60)
        print(f"{'Mean FPS':<20} | {offline_mean_fps:>13.2f} | {realtime_stats['mean_fps']:>13.2f}")
        print(f"{'P95 Latency (ms)':<20} | {offline_p95_latency:>13.2f} | {realtime_stats['p95_latency_ms']:>13.2f}")
        print(f"{'Jitter':<20} | {offline_jitter:>13.4f} | {realtime_stats['jitter']:>13.4f}")
        
        return {
            'offline': {
                'mean_fps': offline_mean_fps,
                'p95_latency_ms': offline_p95_latency,
                'jitter': offline_jitter
            },
            'realtime': realtime_stats
        }
    
    return None


# ============================================================================
# Основная функция для запуска всех задач
# ============================================================================

def main():
    """Главная функция для запуска всех задач."""
    import argparse
    
    parser = argparse.ArgumentParser(description='HW1: Video Pipeline Optimization')
    parser.add_argument('video', type=str, nargs='?', help='Path to video file (positional argument)')
    parser.add_argument('--video', type=str, dest='video_file', help='Path to video file (alternative)')
    parser.add_argument('--task', type=int, choices=range(1, 11), help='Task number (1-10)')
    parser.add_argument('--output', type=str, default='HW1/results', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Run all tasks')
    parser.add_argument('--force', action='store_true', help='Force rerun even if checkpoint exists')
    
    args = parser.parse_args()
    
    # Определяем путь к видео: сначала позиционный аргумент, потом --video
    video_path = args.video or args.video_file
    
    if not video_path:
        parser.error('Video file path is required. Use: python video_pipeline.py <video_file> [--all] or python video_pipeline.py --video <video_file> [--all]')
    
    video_files = [video_path]
    output_dir = args.output
    
    if args.all:
        print("Running all tasks...")
        print("\n" + "="*60)
        print("Task 1: Basic frame decoder")
        print("="*60)
        
        # Проверка CFR/VFR (из семинара 1, задание 1)
        print("\nChecking video format (CFR/VFR)...")
        video_type, metrics = check_cfr_vfr(video_path)
        print(f"Video type: {video_type}")
        if metrics.get('avg_frame_rate'):
            print(f"Average frame rate: {metrics['avg_frame_rate']:.2f} fps")
        if metrics.get('cv') is not None:
            print(f"Coefficient of variation: {metrics['cv']:.4f}")
        
        # Чтение клипа с подробным выводом
        print("\nReading clip...")
        clip = read_clip(video_path, start=0, num_frames=16, stride=2, verbose=True)
        print(f"Clip shape: {clip.shape}")
        print(f"Clip dtype: {clip.dtype}")
        print(f"Value range: [{clip.min()}, {clip.max()}]")
        
        # Визуализация первого и последнего кадра
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(clip[0])
        axes[0].set_title('First frame')
        axes[0].axis('off')
        axes[1].imshow(clip[-1])
        axes[1].set_title('Last frame')
        axes[1].axis('off')
        plt.savefig(f"{output_dir}/task1_frames.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Frames saved to {output_dir}/task1_frames.png")
        
        print("\n" + "="*60)
        print("Task 3: Parallel loading")
        print("="*60)
        task3_parallel_loading(video_files, output_dir, force_rerun=args.force)
        
        print("\n" + "="*60)
        print("Task 4: Profiling")
        print("="*60)
        task4_profiling(video_files, output_dir)
        
        print("\n" + "="*60)
        print("Task 5: Prefetch and pinned memory")
        print("="*60)
        task5_prefetch_pinned_memory(video_files, output_dir, force_rerun=args.force)
        
        print("\n" + "="*60)
        print("Task 6: Pipeline overlap")
        print("="*60)
        task6_pipeline_overlap(video_path, output_dir)
        
        print("\n" + "="*60)
        print("Task 7: Hardware decoding")
        print("="*60)
        task7_hardware_decoding(video_path, output_dir)
        
        print("\n" + "="*60)
        print("Task 8: GPU preprocessing")
        print("="*60)
        task8_gpu_preprocessing(video_files, output_dir)
        
        print("\n" + "="*60)
        print("Task 9: FPS stability")
        print("="*60)
        task9_fps_stability(video_files, output_dir, force_rerun=args.force)
        
        print("\n" + "="*60)
        print("Task 10: Real-time pipeline")
        print("="*60)
        task10_realtime_pipeline(video_path, output_dir)
        
        print("\n" + "="*60)
        print("Mini-HW: Offline vs Real-time")
        print("="*60)
        minihw_offline_vs_realtime(video_path, output_dir)
        
    elif args.task:
        if args.task == 1:
            # Проверка CFR/VFR
            print("Checking video format (CFR/VFR)...")
            video_type, metrics = check_cfr_vfr(video_path)
            print(f"Video type: {video_type}")
            if metrics.get('avg_frame_rate'):
                print(f"Average frame rate: {metrics['avg_frame_rate']:.2f} fps")
            
            # Чтение клипа
            clip = read_clip(video_path, start=0, num_frames=16, stride=2, verbose=True)
            print(f"Clip shape: {clip.shape}")
            print(f"Clip dtype: {clip.dtype}")
        elif args.task == 3:
            task3_parallel_loading(video_files, output_dir, force_rerun=args.force)
        elif args.task == 4:
            task4_profiling(video_files, output_dir)
        elif args.task == 5:
            task5_prefetch_pinned_memory(video_files, output_dir, force_rerun=args.force)
        elif args.task == 6:
            task6_pipeline_overlap(video_path, output_dir)
        elif args.task == 7:
            task7_hardware_decoding(video_path, output_dir)
        elif args.task == 8:
            task8_gpu_preprocessing(video_files, output_dir)
        elif args.task == 9:
            task9_fps_stability(video_files, output_dir, force_rerun=args.force)
        elif args.task == 10:
            task10_realtime_pipeline(video_path, output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

