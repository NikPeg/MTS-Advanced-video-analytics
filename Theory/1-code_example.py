import os
import subprocess
import sys

def make_vfr_cfr(input_path, fps=30):
    base, ext = os.path.splitext(input_path)
    cfr_path = f"{base}_CFR{ext}"
    vfr_path = f"{base}_VFR{ext}"

    # 1. Создание CFR-видео (фиксированная частота кадров)
    cmd_cfr = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"fps={fps}",
        "-vsync", "cfr",
        "-pix_fmt", "yuv420p",
        cfr_path
    ]
    subprocess.run(cmd_cfr, check=True)

    # 2. Создание VFR-видео (искусственное варьирование FPS)
    # Мы изменяем скорость некоторых фрагментов через фильтр setpts.
    # Пример: 0–3 c в 1.0×, 3–6 c в 0.5× (замедленно), 6–9 c в 1.5× (ускорено)
    # Это создаёт переменные интервалы PTS и реальную VFR-структуру.
    cmd_vfr = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex",
        "[0:v]trim=0:3,setpts=PTS-STARTPTS[v0];"
        "[0:v]trim=3:6,setpts=2*PTS[v1];"     # 0.5× скорость
        "[0:v]trim=6:9,setpts=0.66*PTS[v2];"  # 1.5× скорость
        "[v0][v1][v2]concat=n=3:v=1:a=0[v]",
        "-map", "[v]",
        "-vsync", "vfr",
        "-pix_fmt", "yuv420p",
        vfr_path
    ]
    subprocess.run(cmd_vfr, check=True)

    print(f" CFR video saved: {cfr_path}")
    print(f" VFR video saved: {vfr_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_vfr_cfr.py <input_video>")
        sys.exit(1)
    make_vfr_cfr(sys.argv[1])
