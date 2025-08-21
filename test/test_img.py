import os
import time
from datetime import datetime
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

# Target resolution for uniform display
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720


def resize_image_to_hd(image):
    """Resize image to HD resolution (1280x720) while maintaining aspect ratio"""
    h, w = image.shape[:2]

    # Calculate scaling factor
    scale_w = TARGET_WIDTH / w
    scale_h = TARGET_HEIGHT / h
    scale = min(scale_w, scale_h)  # Use smaller scale to fit within bounds

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create canvas with target size and center the image
    canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

    # Calculate position to center the image
    start_x = (TARGET_WIDTH - new_w) // 2
    start_y = (TARGET_HEIGHT - new_h) // 2

    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    return canvas


def draw_label_with_bg(img, label, position=(10, 50), text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    x, y = position
    w, h = text_size
    cv2.rectangle(img, (x, y - h - 10), (x + w + 10, y + 10), bg_color, -1)
    cv2.putText(img, label, (x + 5, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def process_image(model, image_path, model_name, conf=0.25):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    start_time = time.time()
    results = model(image, conf=conf, verbose=False)
    elapsed = time.time() - start_time

    fire, smoke = 0, 0
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        detections.append({'class': cls, 'confidence': conf, 'bbox': box.xyxy[0].tolist()})
        if cls == 0:
            fire += 1
        elif cls == 1:
            smoke += 1

    annotated = results[0].plot()

    # Resize to HD resolution
    annotated = resize_image_to_hd(annotated)

    draw_label_with_bg(annotated, model_name)

    stats = {
        'inference_time': elapsed,
        'fire_count': fire,
        'smoke_count': smoke,
        'detections': detections
    }
    return annotated, stats


def show_comparison(image_path, image_dict):
    model_names = list(image_dict.keys())
    if len(model_names) < 2:
        return
    img1 = image_dict[model_names[0]]
    img2 = image_dict[model_names[1]]
    if img1 is None or img2 is None:
        return

    # Both images are already HD resolution (1280x720), so we can directly combine them
    combined = np.hstack((img1.copy(), img2.copy()))

    # The combined image will be 2560x720, which might be too wide
    # Let's resize it to fit better on screen
    combined_resized = cv2.resize(combined, (1920, 540), interpolation=cv2.INTER_AREA)

    window_name = f"{Path(image_path).stem}: {model_names[0]} vs {model_names[1]}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 540)
    cv2.imshow(window_name, combined_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_stats_yaml(stats, output_dir):
    filtered = {}
    for model, images in stats.items():
        filtered[model] = {}
        for name, metrics in images.items():
            filtered[model][name] = {k: v for k, v in metrics.items() if k != 'detections'}
    with open(os.path.join(output_dir, "image_stats.yaml"), 'w') as f:
        yaml.dump(filtered, f)


def save_stats_table(stats, model_sizes, output_dir):
    data = {
        "Model": [],
        "Size (MB)": [],
        "Avg Inference (ms)": [],
        "Total Fire": [],
        "Total Smoke": [],
        "Fire/Img": [],
        "Smoke/Img": []
    }

    for model in stats:
        infs = [v['inference_time'] for v in stats[model].values()]
        fires = sum(v['fire_count'] for v in stats[model].values())
        smokes = sum(v['smoke_count'] for v in stats[model].values())
        count = len(stats[model])

        data["Model"].append(model)
        data["Size (MB)"].append(f"{model_sizes[model]:.2f}")
        data["Avg Inference (ms)"].append(f"{np.mean(infs) * 1000:.2f}")
        data["Total Fire"].append(str(fires))
        data["Total Smoke"].append(str(smokes))
        data["Fire/Img"].append(f"{fires / count:.2f}")
        data["Smoke/Img"].append(f"{smokes / count:.2f}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    table = ax.table(
        cellText=[list(row) for row in zip(*data.values())],
        colLabels=list(data.keys()),
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detailed_stats_table.png"), dpi=300)
    plt.close()


def main():
    print("Starting Image Evaluation with HD Resolution (1280x720)")

    MODEL_PATHS = {
        "YOLOv12": "../yolov12/runs/YOLOv12/weights/best.pt",
        "YOLO-GFL": "../yolo-gfl/runs/YOLO-GFL/weights/best.pt",
    }

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"Error: Model {name} not found in path: {path}")
            return

    print("All models loaded successfully")

    Tk().withdraw()
    image_paths = askopenfilenames(
        title="Choose images for inference",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )

    if not image_paths:
        print("No images selected. Exiting.")
        return

    print(f"Selected {len(image_paths)} images for inference")
    print(f"Output resolution will be standardized to HD: {TARGET_WIDTH}x{TARGET_HEIGHT}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("../test/output/images", f"batch_{timestamp}")
    output_img_dir = os.path.join(base_dir, "output")
    os.makedirs(output_img_dir, exist_ok=True)

    models = {}
    model_sizes = {}
    for name, path in MODEL_PATHS.items():
        models[name] = YOLO(path)
        model_sizes[name] = os.path.getsize(path) / (1024 * 1024)
        print(f"Model {name}: {model_sizes[name]:.2f} MB")

    stats = {name: {} for name in models}
    result_images = {}

    print("Running inference...")
    for img_path in image_paths:
        image_name = Path(img_path).stem
        result_images[image_name] = {}

        for model_name, model in models.items():
            annotated, stat = process_image(model, img_path, model_name)
            if annotated is not None:
                filename = f"{image_name}_{model_name}.png"
                save_path = os.path.join(output_img_dir, filename)
                cv2.imwrite(save_path, annotated)
                stats[model_name][image_name] = stat
                result_images[image_name][model_name] = annotated

    print("Generating comparison visualizations...")
    for img_path in image_paths:
        show_comparison(img_path, result_images[Path(img_path).stem])

    print("Saving statistics...")
    save_stats_yaml(stats, base_dir)
    save_stats_table(stats, model_sizes, base_dir)

    print(f"\nImage Evaluation Complete!")
    print(f"Results saved to: {base_dir}")
    print(f"Annotated images: {output_img_dir}")
    print(f"Statistics and tables: {base_dir}")
    print(f"All output images standardized to HD resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")


if __name__ == "__main__":
    main()