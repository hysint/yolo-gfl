import os
import time
from datetime import datetime
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO


def process_video(model, video_info, model_config, output_dir, stats):
    cap = cv2.VideoCapture(video_info['path'])
    video_name = Path(video_info['path']).stem

    if not cap.isOpened():
        print(f"Error: Could not open video {video_info['name']}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output directory
    model_video_dir = os.path.join(output_dir['videos'], model_config['name'])
    os.makedirs(model_video_dir, exist_ok=True)

    # Output video path with fixed codec
    video_output_path = os.path.join(model_video_dir, f"{video_name}_{model_config['name']}.mp4")

    # Use mp4v codec which is better supported for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed from XVID to mp4v
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # Optimization: Pre-compile model for faster inference
    print(f"Warming up model {model_config['name']}...")
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(3):  # Warm up iterations
        _ = model(dummy_frame, conf=model_config['conf_threshold'], verbose=False)

    # Statistics tracking
    frame_times = []
    fire_counts = []
    smoke_counts = []
    processed_frames = 0
    detections = []

    # FPS calculation for display
    fps_window = []
    fps_window_size = 30

    print(f"Processing {total_frames} frames...")

    # Process frames with optimizations
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        processed_frames += 1

        # Show progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count / total_frames * 100:.1f}%)")

        start_time = time.time()

        # Optimized inference with reduced verbosity
        results = model(frame,
                        conf=model_config['conf_threshold'],
                        verbose=False,  # Disable verbose output
                        save=False,  # Don't save intermediate results
                        show=False)  # Don't show intermediate results

        inference_time = time.time() - start_time
        frame_times.append(inference_time)

        # Calculate real-time FPS for display
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        fps_window.append(current_fps)
        if len(fps_window) > fps_window_size:
            fps_window.pop(0)
        avg_fps = sum(fps_window) / len(fps_window)

        # Count detections
        fire_count = 0
        smoke_count = 0
        frame_detections = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                frame_detections.append({
                    'class': cls,
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                })

                if cls == 0:
                    fire_count += 1
                elif cls == 1:
                    smoke_count += 1

        fire_counts.append(fire_count)
        smoke_counts.append(smoke_count)
        detections.append(frame_detections)

        # Optimized annotation - only annotate if there are detections
        if frame_detections:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame.copy()

        # Add overlay texts
        fps_text = f"FPS: {avg_fps:.1f}"
        det_text = f"Fire: {fire_count} | Smoke: {smoke_count}"

        # Get text sizes
        fps_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        det_size, _ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Padding and dimensions
        padding_x = 20
        padding_y = 10
        spacing_y = 5  # space between the two texts

        bg_width = max(fps_size[0], det_size[0]) + padding_x
        total_text_height = fps_size[1] + det_size[1] + spacing_y
        bg_height = total_text_height + padding_y

        # Draw unified background rectangle
        cv2.rectangle(
            annotated_frame,
            (5, 5),
            (5 + bg_width, 5 + bg_height),
            (0, 0, 0),
            -1
        )

        # Calculate text positions
        fps_text_y = 5 + fps_size[1] + padding_y // 2
        det_text_y = fps_text_y + det_size[1] + spacing_y

        # Draw FPS text
        cv2.putText(
            annotated_frame,
            fps_text,
            org=(10, fps_text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # Draw Detection Count text
        cv2.putText(
            annotated_frame,
            det_text,
            org=(10, det_text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # Optimized display - resize for faster rendering if needed
        display_frame = annotated_frame
        if width > 1280:  # Resize large frames for display
            scale = 1280 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_frame = cv2.resize(annotated_frame, (new_width, new_height))

        # Display frame
        cv2.imshow(f"Inference - {model_config['name']} - {video_info['name']}", display_frame)

        # Non-blocking key check for faster processing
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Processing interrupted by user")
            break

        # Write original size frame to video
        out.write(annotated_frame)

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video processing complete: {video_output_path}")

    # Store comprehensive statistics
    if frame_times:
        stats[model_config['name']][video_info['name']] = {
            'avg_fps': len(frame_times) / sum(frame_times),
            'total_fire': sum(fire_counts),
            'total_smoke': sum(smoke_counts),
            'max_fire_per_frame': max(fire_counts) if fire_counts else 0,
            'max_smoke_per_frame': max(smoke_counts) if smoke_counts else 0,
            'avg_inference': sum(frame_times) / len(frame_times),
            'min_inference': min(frame_times),
            'max_inference': max(frame_times),
            'total_frames': processed_frames,
            'detections': detections,
            'fire_per_frame': sum(fire_counts) / processed_frames if processed_frames > 0 else 0,
            'smoke_per_frame': sum(smoke_counts) / processed_frames if processed_frames > 0 else 0,
            'processing_efficiency': processed_frames / total_frames if total_frames > 0 else 0
        }
    else:
        print(f"Warning: No frames processed for {video_info['name']}")


def visualize_stats(stats, output_dir, model_configs):
    if not stats or not any(stats.values()):
        print("No statistics to visualize")
        return

    # Create performance comparison plots
    plt.style.use('default')  # Use default style for faster rendering

    videos = list(next(iter(stats.values())).keys())
    num_models = len(stats)

    # Main performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # 1. FPS Comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(videos))
    width = 0.35 if num_models == 2 else 0.25

    for i, (model_name, videos_stats) in enumerate(stats.items()):
        fps_values = [v['avg_fps'] for v in videos_stats.values()]
        bars = ax1.bar(x_pos + (i * width), fps_values, width, label=model_name, alpha=0.8)

        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, fps_values)):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + max(fps_values) * 0.01,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_title('Average FPS Performance', fontweight='bold')
    ax1.set_ylabel('Frames Per Second')
    ax1.set_xlabel('Videos')
    ax1.set_xticks(x_pos + width / 2)
    ax1.set_xticklabels(videos, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Detection Performance
    ax2 = axes[0, 1]
    x = np.arange(len(videos))
    width_det = 0.15

    for i, (model_name, videos_stats) in enumerate(stats.items()):
        fire_counts = [v['total_fire'] for v in videos_stats.values()]
        smoke_counts = [v['total_smoke'] for v in videos_stats.values()]

        fire_bars = ax2.bar(x + (i * width_det * 2), fire_counts, width_det,
                            label=f'{model_name} Fire', color='red', alpha=0.7)
        smoke_bars = ax2.bar(x + (i * width_det * 2) + width_det, smoke_counts, width_det,
                             label=f'{model_name} Smoke', color='gray', alpha=0.7)

        # Add count labels
        for bar, val in zip(fire_bars, fire_counts):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, val + max(fire_counts + smoke_counts) * 0.01,
                         str(val), ha='center', va='bottom', fontsize=8)

        for bar, val in zip(smoke_bars, smoke_counts):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, val + max(fire_counts + smoke_counts) * 0.01,
                         str(val), ha='center', va='bottom', fontsize=8)

    ax2.set_title('Total Detection Counts', fontweight='bold')
    ax2.set_ylabel('Detection Count')
    ax2.set_xlabel('Videos')
    ax2.set_xticks(x + width_det)
    ax2.set_xticklabels(videos, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Inference Time Analysis
    ax3 = axes[1, 0]
    for i, (model_name, videos_stats) in enumerate(stats.items()):
        inf_times = [v['avg_inference'] * 1000 for v in videos_stats.values()]  # Convert to ms
        min_times = [v['min_inference'] * 1000 for v in videos_stats.values()]
        max_times = [v['max_inference'] * 1000 for v in videos_stats.values()]

        ax3.errorbar(x + (i * 0.1), inf_times,
                     yerr=[np.array(inf_times) - np.array(min_times),
                           np.array(max_times) - np.array(inf_times)],
                     fmt='o-', label=model_name, capsize=5, capthick=2)

        # Add time labels
        for j, val in enumerate(inf_times):
            ax3.text(x[j] + (i * 0.1), val - 5, f'{val:.1f}ms',
                     ha='center', va='top', fontsize=8)

    ax3.set_title('Inference Time (with Min/Max Range)', fontweight='bold')
    ax3.set_ylabel('Time (milliseconds)')
    ax3.set_xlabel('Videos')
    ax3.set_xticks(x)
    ax3.set_xticklabels(videos, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Model Efficiency (Size vs Performance)
    ax4 = axes[1, 1]

    for model_name in stats.keys():
        model_config = next((c for c in model_configs if c['name'] == model_name), None)
        if model_config:
            size_mb = model_config['size_mb']
            avg_fps = np.mean([v['avg_fps'] for v in stats[model_name].values()])

            ax4.scatter(size_mb, avg_fps, s=200, alpha=0.7, label=model_name)
            ax4.annotate(f'{model_name}\n{size_mb:.1f}MB\n{avg_fps:.1f}FPS',
                         (size_mb, avg_fps), xytext=(10, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax4.set_title('Model Size vs Performance Efficiency', fontweight='bold')
    ax4.set_xlabel('Model Size (MB)')
    ax4.set_ylabel('Average FPS')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir['stats'], 'performance_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Generate detailed statistics table
    generate_stats_table(stats, model_configs, output_dir)

    # Save performance data
    save_performance_data(stats, model_configs, output_dir)

    print("Performance visualization completed successfully!")


def generate_stats_table(stats, model_configs, output_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Model', 'Size (MB)', 'Avg FPS', 'Inference (ms)',
               'Total Fire', 'Total Smoke', 'Fire/Frame', 'Smoke/Frame', 'Efficiency']

    for model_name in stats.keys():
        model_config = next((c for c in model_configs if c['name'] == model_name), None)
        size_mb = model_config['size_mb'] if model_config else 0

        # Calculate aggregated metrics
        all_videos = stats[model_name]
        avg_fps = np.mean([v['avg_fps'] for v in all_videos.values()])
        avg_inference = np.mean([v['avg_inference'] for v in all_videos.values()]) * 1000
        total_fire = sum([v['total_fire'] for v in all_videos.values()])
        total_smoke = sum([v['total_smoke'] for v in all_videos.values()])
        total_frames = sum([v['total_frames'] for v in all_videos.values()])

        fire_per_frame = total_fire / total_frames if total_frames > 0 else 0
        smoke_per_frame = total_smoke / total_frames if total_frames > 0 else 0
        efficiency = avg_fps / size_mb if size_mb > 0 else 0

        row = [
            model_name,
            f'{size_mb:.2f}',
            f'{avg_fps:.2f}',
            f'{avg_inference:.2f}',
            str(total_fire),
            str(total_smoke),
            f'{fire_per_frame:.3f}',
            f'{smoke_per_frame:.3f}',
            f'{efficiency:.2f}'
        ]
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Detailed Performance Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir['stats'], 'detailed_stats_table.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def save_performance_data(stats, model_configs, output_dir):
    # Prepare data for saving (exclude large detection arrays)
    save_data = {
        'models': {},
        'summary': {},
        'timestamp': datetime.now().isoformat()
    }

    for model_name in stats.keys():
        model_config = next((c for c in model_configs if c['name'] == model_name), None)

        # Model details
        save_data['models'][model_name] = {
            'config': {
                'size_mb': model_config['size_mb'] if model_config else 0,
                'conf_threshold': model_config['conf_threshold'] if model_config else 0
            },
            'videos': {}
        }

        # Video results (without detection arrays for file size)
        for video_name, metrics in stats[model_name].items():
            filtered_metrics = {k: v for k, v in metrics.items() if k != 'detections'}
            save_data['models'][model_name]['videos'][video_name] = filtered_metrics

        # Model summary
        all_videos = stats[model_name]
        save_data['summary'][model_name] = {
            'avg_fps': np.mean([v['avg_fps'] for v in all_videos.values()]),
            'total_detections': sum([v['total_fire'] + v['total_smoke'] for v in all_videos.values()]),
            'avg_inference_ms': np.mean([v['avg_inference'] for v in all_videos.values()]) * 1000,
            'videos_processed': len(all_videos)
        }

    # Save to file
    with open(os.path.join(output_dir['stats'], 'performance_data.yaml'), 'w') as f:
        yaml.dump(save_data, f, default_flow_style=False, indent=2)


def main():
    print("Starting Video Evaluation")

    MODEL_PATHS = {
        "YOLOv12": "../yolov12/runs/YOLOv12/weights/best.pt",
        "YOLO-GFL": "../yolo-gfl/runs/YOLO-GFL/weights/best.pt",
    }

    CONF_THRESHOLD = 0.25

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"Error: Model {name} not found at: {path}")
            return

    print("All models loaded successfully")

    Tk().withdraw()
    selected_path = askopenfilename(
        title="Select video file for inference",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )

    if not selected_path:
        print("No video file selected. Exiting.")
        return

    print(f"Selected video: {Path(selected_path).name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("../test/output/videos", f"batch_{timestamp}")
    video_output_dir = os.path.join(base_output_dir, "videos")
    stats_output_dir = os.path.join(base_output_dir, "stats", Path(selected_path).stem)

    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(stats_output_dir, exist_ok=True)

    print(f"Output directory: {base_output_dir}")

    model_configs = []
    for name, path in MODEL_PATHS.items():
        size_mb = os.path.getsize(path) / (1024 * 1024)
        model_configs.append({
            'name': name,
            'path': path,
            'conf_threshold': CONF_THRESHOLD,
            'size_mb': size_mb
        })
        print(f"Model {name}: {size_mb:.2f} MB")

    stats = {name: {} for name in MODEL_PATHS.keys()}

    for model_config in model_configs:
        print(f"\nProcessing with model: {model_config['name']}")
        model = YOLO(model_config['path'])
        process_video(
            model=model,
            video_info={'name': Path(selected_path).stem, 'path': selected_path},
            model_config=model_config,
            output_dir={'videos': video_output_dir, 'stats': stats_output_dir},
            stats=stats
        )

    print("\nGenerating performance analysis...")
    visualize_stats(stats, {'videos': video_output_dir, 'stats': stats_output_dir}, model_configs)

    print(f"\nVideo Evaluation Complete!")
    print(f"Results saved to: {base_output_dir}")
    print(f"Video outputs: {video_output_dir}")
    print(f"Statistics: {stats_output_dir}")


if __name__ == "__main__":
    main()