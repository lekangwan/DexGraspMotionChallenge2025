"""把真实 Isaac Gym MP4 整理为 PPT 用短视频、GIF 和四帧故事板。"""

from pathlib import Path
import shutil

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def collect_retarget_videos():
    """输入：最终重定向录像目录；输出：复制到统一资产目录的 MP4；作用：便于 PPT 打包。"""
    source = ROOT / "retarget_research/reports/videos/final_retargeting"
    target = ASSETS / "videos/retarget"
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*/*.mp4"):
        if path.stat().st_size > 1024:
            shutil.copy2(path, target / f"{path.parent.name}_{path.name}")


def video_frames(path):
    """输入：MP4；输出：RGB 帧列表；作用：为 GIF 和故事板提供统一数据。"""
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def presentation_crop(frame):
    """输入：Isaac 原始帧；输出：裁去上下空白的 8:5 画面；作用：放大手与物体。"""
    height = frame.shape[0]
    return frame[int(height * 0.10):int(height * 0.95)]


def make_gif(path, output):
    """输入：MP4；输出：12 fps GIF；作用：PPT 中无需播放器即可循环展示。"""
    frames = video_frames(path)
    if not frames:
        return
    step = max(round(len(frames) / 72), 1)
    images = [Image.fromarray(presentation_crop(frame)).resize(
              (640, 400), Image.Resampling.LANCZOS)
              for frame in frames[::step]]
    images[0].save(output, save_all=True, append_images=images[1:], duration=83,
                   loop=0, optimize=True)


def make_storyboard(path, output):
    """输入：MP4；输出：接近/闭合/抬升/末帧拼图；作用：静态页面也能讲清动作过程。"""
    frames = video_frames(path)
    if not frames:
        return
    indices = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    labels = ["接近", "闭合", "抬升", "末帧"]
    font = ImageFont.truetype(FONT_PATH, 27)
    tiles = []
    for index, label in zip(indices, labels):
        image = Image.fromarray(presentation_crop(frames[index])).resize(
            (480, 300), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (480, 348), "white")
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(((480 - box[2]) / 2, 308), label, font=font, fill="#243447")
        tiles.append(canvas)
    sheet = Image.new("RGB", (1920, 348), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (480 * i, 0))
    sheet.save(output, quality=95)


def make_keyframe(path, output):
    """输入：MP4；输出：闭合阶段单帧；作用：作为论文流程图中的真实仿真缩略图。"""
    frames = video_frames(path)
    if not frames:
        return
    frame = presentation_crop(frames[len(frames) // 3])
    Image.fromarray(frame).save(output)


def main():
    """输入：已录好的 MP4；输出：GIF 与故事板；作用：一键制作展示衍生资产。"""
    collect_retarget_videos()
    gif_dir = ASSETS / "animations"
    sheet_dir = ASSETS / "storyboards"
    keyframe_dir = ASSETS / "keyframes"
    gif_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((ASSETS / "videos").glob("**/*.mp4")):
        if path.stat().st_size <= 1024:
            continue
        name = "_".join(path.relative_to(ASSETS / "videos").with_suffix("").parts)
        make_gif(path, gif_dir / f"{name}.gif")
        make_storyboard(path, sheet_dir / f"{name}.jpg")
        make_keyframe(path, keyframe_dir / f"{name}.png")
    print("VIDEO_DERIVATIVES=COMPLETE")


if __name__ == "__main__":
    main()
