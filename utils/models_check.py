import torch
import os
from thop import profile, clever_format
from ultralytics import YOLO
import pandas as pd


def create_descriptive_name(model_path):
    """Generate a descriptive model name from its path."""
    normalized_path = os.path.normpath(model_path)
    path_parts = normalized_path.split(os.sep)

    if 'runs' in path_parts:
        runs_index = path_parts.index('runs')
        if runs_index + 1 < len(path_parts):
            experiment_name = path_parts[runs_index + 1]
            return f"{experiment_name}_best.pt"

    if len(path_parts) >= 2:
        parent_folder = path_parts[-3] if len(path_parts) >= 3 else path_parts[-2]
        return f"{parent_folder}_best.pt"

    return os.path.basename(model_path)


def analyze_model(model_path):
    """Analyze a YOLO model and return its characteristics."""
    try:
        if not os.path.exists(model_path):
            return {"error": f"File not found: {model_path}"}

        model = YOLO(model_path)
        pytorch_model = model.model
        pytorch_model.eval()

        input_tensor = torch.randn(1, 3, 640, 640)

        flops, params = profile(pytorch_model, inputs=(input_tensor,), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        model_size_mb = sum(p.numel() * p.element_size() for p in pytorch_model.parameters()) / (1024 * 1024)

        model_name = create_descriptive_name(model_path)

        return {
            "model_name": model_name,
            "full_path": model_path,
            "parameters": params,
            "parameters_formatted": params_formatted,
            "gflops": flops / 1e9,
            "gflops_formatted": flops_formatted,
            "file_size_mb": file_size_mb,
            "memory_mb": model_size_mb,
            "task": model.task
        }

    except Exception as e:
        return {"error": f"Error analyzing {model_path}: {str(e)}"}


def compare_models(model_paths):
    """Compare multiple YOLO models and print a summary table."""
    print("Starting Model Analysis")

    results = []
    for model_path in model_paths:
        print(f"Analyzing model: {model_path}")
        result = analyze_model(model_path)
        if "error" not in result:
            results.append(result)
        else:
            print(f"Error: {result['error']}")

    if not results:
        print("No models were successfully analyzed.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values('parameters')

    print("\nComparison Results")
    print(f"{'Model':<25} {'Parameters':<12} {'GFLOPs':<12} {'File Size (MB)':<15} {'Memory (MB)':<12}")

    for _, row in df.iterrows():
        print(
            f"{row['model_name']:<25} "
            f"{row['parameters_formatted']:<12} "
            f"{row['gflops']:<12.3f} "
            f"{row['file_size_mb']:<15.2f} "
            f"{row['memory_mb']:<12.2f}"
        )

    print("\nModel Analysis Complete!")


def main():
    model_paths = [
        "../yolo-gfl/runs/YOLO-GFL/weights/best.pt",
        "../yolov12/runs/YOLOv12/weights/best.pt",
    ]

    print("Checking dependencies...")
    try:
        import thop
        import ultralytics
        import pandas
        print("Dependencies check: OK")
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Please install with: pip install thop ultralytics pandas")
        return

    compare_models(model_paths)


if __name__ == "__main__":
    main()