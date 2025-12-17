"""
Простой скрипт для быстрой проверки работы пайплайна
"""

import sys
import os

# Добавляем путь к модулю
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_pipeline import read_clip, VideoDataset
import numpy as np

def test_task1(video_file):
    """Тест задачи 1: базовый декодер"""
    print("="*60)
    print("Testing Task 1: Basic Frame Decoder")
    print("="*60)
    
    try:
        clip = read_clip(video_file, start=0, num_frames=8, stride=2)
        print(f"✓ Successfully read clip")
        print(f"  Shape: {clip.shape}")
        print(f"  Dtype: {clip.dtype}")
        print(f"  Value range: [{clip.min():.1f}, {clip.max():.1f}]")
        
        assert clip.shape[0] == 8, f"Expected 8 frames, got {clip.shape[0]}"
        assert len(clip.shape) == 4, f"Expected 4D array, got {len(clip.shape)}D"
        assert clip.shape[3] == 3, f"Expected 3 channels, got {clip.shape[3]}"
        
        print("✓ All assertions passed!")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_task2(video_file):
    """Тест задачи 2: VideoDataset"""
    print("\n" + "="*60)
    print("Testing Task 2: VideoDataset")
    print("="*60)
    
    try:
        dataset = VideoDataset([video_file], clip_len=8, stride=2)
        print(f"✓ Dataset created")
        print(f"  Length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample shape: {sample.shape}")
            print(f"  Sample dtype: {sample.dtype}")
            
            assert len(sample.shape) == 4, "Expected 4D array"
            assert sample.shape[0] == 8, "Expected 8 frames"
            
            print("✓ All assertions passed!")
            return True
        else:
            print("⚠ Dataset is empty (video might be too short)")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <video_file>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    
    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        sys.exit(1)
    
    print(f"Testing with video: {video_file}\n")
    
    results = []
    results.append(("Task 1", test_task1(video_file)))
    results.append(("Task 2", test_task2(video_file)))
    
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    for task, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{task}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

