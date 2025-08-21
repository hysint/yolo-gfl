import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def calculate_performance(
        model,
        test_images_dir,
        conf_threshold=0.25,
        warmup_images=5,
        save_inference=True,
        output_dir="runs/inference_output"):

    print(f"Calculating performance metrics...")

    # Get all test images
    image_files = [f for f in os.listdir(test_images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(image_files) == 0:
        print(f"No images found in {test_images_dir}")
        return 0, 0, []

    print(f"Found {len(image_files)} test images")

    # Create output directory for this model if saving inference results
    model_name = getattr(model, 'model_name', 'unknown_model')
    if save_inference:
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"Inference results will be saved to: {model_output_dir}")

    # Warmup with first few images
    print("Warming up model...")
    for i in range(min(warmup_images, len(image_files))):
        try:
            image_path = os.path.join(test_images_dir, image_files[i])
            image = cv2.imread(image_path)
            if image is not None:
                _ = model(image, conf=conf_threshold, verbose=False)
        except Exception as e:
            print(f"Warning: Error during warmup with image {image_files[i]}: {str(e)}")

    # Measure inference times for all images
    inference_times = []
    processed_images = 0
    saved_images = 0

    print("Measuring inference performance...")
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            image_path = os.path.join(test_images_dir, image_file)
            image = cv2.imread(image_path)

            if image is not None:
                start_time = time.time()
                results = model(image, conf=conf_threshold, verbose=False)
                inference_time = time.time() - start_time

                inference_times.append(inference_time)
                processed_images += 1

                # Save inference result if enabled
                if save_inference and len(results) > 0:
                    try:
                        # Use YOLO's built-in plotting functionality
                        annotated_image = results[0].plot()

                        # Save annotated image
                        output_path = os.path.join(model_output_dir, f"result_{image_file}")
                        cv2.imwrite(output_path, annotated_image)
                        saved_images += 1

                    except Exception as e:
                        print(f"Warning: Could not save inference result for {image_file}: {str(e)}")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    if len(inference_times) == 0:
        print("No images were successfully processed")
        return 0, 0, []

    # Calculate metrics
    avg_inference_time = np.mean(inference_times)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    print(f"  Processed {processed_images} images")
    print(f"  Average inference time: {avg_inference_time * 1000:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    if save_inference:
        print(f"  Saved {saved_images} inference results to {model_output_dir}")

    return avg_inference_time, fps, inference_times


def save_inference_summary(results, output_dir="runs/inference_output"):
    summary_path = os.path.join(output_dir, "inference_summary.txt")

    with open(summary_path, 'w') as f:
        f.write("Inference Results Summary\n")
        f.write("=" * 50 + "\n\n")

        for result in results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"  - Images processed: {result['num_test_images']}\n")
            f.write(f"  - Average inference time: {result['avg_inference_time'] * 1000:.2f} ms\n")
            f.write(f"  - FPS: {result['fps']:.2f}\n")
            f.write(f"  - Model size: {result['model_size_mb']:.2f} MB\n")
            f.write(f"  - Inference results saved to: runs/inference_output/{result['model']}/\n")
            f.write("\n")

        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Inference summary saved to: {summary_path}")


def create_inference_comparison_grid(results, output_dir="runs/inference_output", max_images=6):
    print("Creating inference comparison grid...")

    # Find common images across all models
    all_model_dirs = []
    for result in results:
        model_dir = os.path.join(output_dir, result['model'])
        if os.path.exists(model_dir):
            all_model_dirs.append((result['model'], model_dir))

    if len(all_model_dirs) < 2:
        print("Need at least 2 models with inference results to create comparison grid")
        return

    # Get common images (images that exist in all model directories)
    common_images = None
    for model_name, model_dir in all_model_dirs:
        model_images = set([f for f in os.listdir(model_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if common_images is None:
            common_images = model_images
        else:
            common_images = common_images.intersection(model_images)

    if not common_images:
        print("No common images found across all models")
        return

    # Select subset of images for comparison
    selected_images = list(common_images)[:max_images]
    num_models = len(all_model_dirs)
    num_images = len(selected_images)

    # Create comparison grid
    fig, axes = plt.subplots(num_images, num_models, figsize=(5 * num_models, 5 * num_images))

    # Handle single row case
    if num_images == 1:
        axes = axes.reshape(1, -1)
    elif num_models == 1:
        axes = axes.reshape(-1, 1)

    for img_idx, image_name in enumerate(selected_images):
        for model_idx, (model_name, model_dir) in enumerate(all_model_dirs):
            try:
                image_path = os.path.join(model_dir, image_name)
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                ax = axes[img_idx, model_idx] if num_images > 1 else axes[model_idx]
                ax.imshow(image_rgb)
                ax.set_title(f"{model_name}\n{image_name}", fontsize=10)
                ax.axis('off')

            except Exception as e:
                print(f"Error loading {image_name} for {model_name}: {str(e)}")
                ax = axes[img_idx, model_idx] if num_images > 1 else axes[model_idx]
                ax.text(0.5, 0.5, 'Error loading\nimage',
                        horizontalalignment='center', verticalalignment='center')
                ax.set_title(f"{model_name}\n{image_name}", fontsize=10)

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'model_comparison_grid.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Inference comparison grid saved to: {comparison_path}")


def visualize_performance_metrics(results, output_dir="runs/performance_analysis"):
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for visualization
    model_names = [result['model'] for result in results]
    avg_times_ms = [result['avg_inference_time'] * 1000 for result in results]  # Convert to ms
    std_times_ms = [result['std_inference_time'] * 1000 for result in results]  # Convert to ms
    fps_values = [result['fps'] for result in results]

    # Create figure with subplots (stacked vertically)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))

    # 1. Average Inference Time with error bars
    bars1 = ax1.bar(model_names, avg_times_ms, yerr=std_times_ms, capsize=5,
                    color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(model_names)])
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Average Inference Time Comparison')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels inside bars
    for i, (bar, avg_time, std_time) in enumerate(zip(bars1, avg_times_ms, std_times_ms)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height / 2,
                 f'{avg_time:.1f}ms\n±{std_time:.1f}',
                 ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # 2. FPS Comparison
    bars2 = ax2.bar(model_names, fps_values,
                    color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(model_names)])
    ax2.set_xlabel('Model')
    ax2.set_ylabel('FPS (Frames Per Second)')
    ax2.set_title('FPS Comparison')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels inside bars
    for bar, fps in zip(bars2, fps_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height / 2,
                 f'{fps:.1f}',
                 ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # 3. Inference Time Distribution (Box plot)
    inference_time_data = [np.array(result['inference_times']) * 1000 for result in results]  # Convert to ms
    ax3.boxplot(inference_time_data, tick_labels=model_names)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('Inference Time Distribution')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Performance Summary Table
    ax4.axis('off')

    # Create table data
    table_data = []
    for result in results:
        table_data.append([
            result['model'],
            f"{result['avg_inference_time'] * 1000:.2f}",
            f"{result['std_inference_time'] * 1000:.2f}",
            f"{result['fps']:.2f}",
            f"{len(result['inference_times'])}"
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Model', 'Avg Time (ms)', 'Std Time (ms)', 'FPS', 'Images'],
        cellLoc='center',
        colLoc='center',
        loc='center',
        bbox=(0, 0, 1, 1)  # Full width and height of the axis
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Performance Summary')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Performance visualization saved to: {os.path.join(output_dir, 'performance_comparison.png')}")

    # Create a separate detailed comparison chart
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

    x = np.arange(len(model_names))
    width = 0.35

    # Normalize FPS to fit with inference timescale (for dual y-axis effect)
    fps_normalized = np.array(fps_values) * 10  # Scale factor for visibility

    bars1 = ax.bar(x - width / 2, avg_times_ms, width, label='Inference Time (ms)',
                   alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width / 2, fps_normalized, width, label='FPS (×10)',
                   alpha=0.8, color='orange')

    ax.set_xlabel('Model')
    ax.set_ylabel('Time (ms) / FPS (×10)')
    ax.set_title('Inference Time vs FPS Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    # Add value labels
    for bar, time_val in zip(bars1, avg_times_ms):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{time_val:.1f}ms', ha='center', va='bottom')

    for bar, fps_val in zip(bars2, fps_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{fps_val:.1f}fps', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_fps_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Time vs FPS comparison saved to: {os.path.join(output_dir, 'time_vs_fps_comparison.png')}")


def evaluate_model(model_path: str, data_path: str, project_name: str, run_name: str, test_images_dir: str):
    print(f"\nEvaluating Model: {run_name}")

    # Load model
    model = YOLO(model_path, task='detect')
    # Add model name attribute for inference saving
    model.model_name = run_name

    print("Model Info:")
    print(model.info())

    # Get model size
    try:
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        print(f"Model Size: {model_size_mb:.2f} MB")
    except Exception as e:
        print(f"Could not determine model size: {str(e)}")
        model_size_mb = 0

    print(f"\nUltralytics Evaluation")
    # Evaluation using Ultralytics framework
    start_time = time.time()
    metrics = model.val(
        data=data_path,
        save_json=True,
        project=project_name,
        name=run_name,
        split='test',  # Use test split for evaluation
    )
    ultralytics_eval_time = time.time() - start_time
    print(f"Ultralytics evaluation completed in {ultralytics_eval_time:.2f} seconds")

    print(f"\nPerformance Analysis with Inference Saving")
    avg_inference_time, fps, inference_times = calculate_performance(
        model, test_images_dir, conf_threshold=0.25, save_inference=True
    )

    # Calculate statistics
    std_inference_time = np.std(inference_times) if inference_times else 0
    min_inference_time = np.min(inference_times) if inference_times else 0
    max_inference_time = np.max(inference_times) if inference_times else 0

    # Return results
    return {
        'model': run_name,
        'model_path': model_path,
        'model_size_mb': model_size_mb,
        'ultralytics_metrics': metrics,
        'ultralytics_eval_time': ultralytics_eval_time,
        'avg_inference_time': avg_inference_time,
        'std_inference_time': std_inference_time,
        'min_inference_time': min_inference_time,
        'max_inference_time': max_inference_time,
        'fps': fps,
        'inference_times': inference_times,
        'num_test_images': len(inference_times)
    }


def print_performance_summary(results):
    print(f"\nPerformance Summary")

    # Print header
    print(f"{'Model':<15} {'Size(MB)':<10} {'Avg Time(ms)':<15} {'Std Time(ms)':<15} {'FPS':<8} {'Images':<8}")

    # Print each model's performance
    for result in results:
        print(f"{result['model']:<15} "
              f"{result['model_size_mb']:<10.2f} "
              f"{result['avg_inference_time'] * 1000:<15.2f} "
              f"{result['std_inference_time'] * 1000:<15.2f} "
              f"{result['fps']:<8.2f} "
              f"{result['num_test_images']:<8}")

    # Find best performing model
    if results:
        best_fps_model = max(results, key=lambda x: x['fps'])
        fastest_model = min(results, key=lambda x: x['avg_inference_time'])

        print(f"\nBest FPS: {best_fps_model['model']} ({best_fps_model['fps']:.2f} FPS)")
        print(f"Fastest Inference: {fastest_model['model']} ({fastest_model['avg_inference_time'] * 1000:.2f} ms)")


def main():
    # Configuration - Modify these paths according to your setup
    data_path = r"../dataset/HOME-FIRE/data.yaml"
    test_images_dir = r"../dataset/HOME-FIRE/test/images"  # Direct path to test images

    models_to_evaluate = [
        {
            "model_path": r'../yolov12/runs/YOLOv12/weights/best.pt',
            "project_name": 'runs/evaluation',
            "run_name": 'YOLOv12',
        },
        {
            "model_path": r'../yolo-gfl/runs/YOLO-GFL/weights/best.pt',
            "project_name": 'runs/evaluation',
            "run_name": 'YOLO-GFL',
        },
    ]

    # Verify test images directory exists
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found: {test_images_dir}")
        print("Please update the test_images_dir path in the main() function")
        return

    print(f"Starting Model Evaluation")
    print(f"Data config: {data_path}")
    print(f"Test images: {test_images_dir}")
    print(f"Models to evaluate: {len(models_to_evaluate)}")

    all_results = []

    for model_info in models_to_evaluate:
        try:
            result = evaluate_model(
                model_path=model_info["model_path"],
                data_path=data_path,
                project_name=model_info["project_name"],
                run_name=model_info["run_name"],
                test_images_dir=test_images_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error evaluating model {model_info['run_name']}: {str(e)}")
            continue

    if all_results:
        # Print performance summary
        print_performance_summary(all_results)

        # Generate performance visualizations
        print(f"\nGenerating performance visualizations...")
        visualize_performance_metrics(all_results)

        # Save inference summary and create comparison grid
        print(f"\nSaving inference results summary...")
        save_inference_summary(all_results)

        print(f"\nCreating inference comparison grid...")
        create_inference_comparison_grid(all_results)

        print("Evaluation Complete!")
    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    main()