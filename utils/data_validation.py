import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split


def load_yaml(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def check_corrupted_files(dataset_path):
    """Check for corrupted image files and annotation files."""
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')

    corrupted_images = []
    corrupted_annotations = []
    normal_count = 0
    missing_annotations = 0

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        return corrupted_images, corrupted_annotations, normal_count, missing_annotations

    for image_file in os.listdir(images_path):
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(images_path, image_file)
        label_path = os.path.join(labels_path, os.path.splitext(image_file)[0] + '.txt')

        try:
            img = Image.open(image_path)
            img.verify()
            img.close()

            if not os.path.exists(label_path):
                missing_annotations += 1
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            valid = True
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        valid = False
                        break

            if valid:
                normal_count += 1
            else:
                corrupted_annotations.append(image_file)

        except (IOError, SyntaxError, ValueError):
            corrupted_images.append(image_file)

    return corrupted_images, corrupted_annotations, normal_count, missing_annotations


def get_class_distribution(dataset_path, classes):
    """Get distribution of classes in the dataset."""
    class_counts = {cls: 0 for cls in classes}
    background_count = 0
    invalid_annotations = 0

    labels_path = os.path.join(dataset_path, 'labels')
    if not os.path.exists(labels_path):
        return class_counts, background_count, invalid_annotations

    for label_file in os.listdir(labels_path):
        if not label_file.endswith('.txt'):
            continue

        file_path = os.path.join(labels_path, label_file)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if not lines or all(not line.strip() for line in lines):
                background_count += 1
                continue

            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) < 1:
                        continue
                    try:
                        class_id = int(parts[0])
                        if class_id < len(classes):
                            class_counts[classes[class_id]] += 1
                    except (ValueError, IndexError):
                        invalid_annotations += 1
                        continue

        except Exception as e:
            print(f"Warning: Error processing {file_path}: {str(e)}")
            invalid_annotations += 1
            continue

    return class_counts, background_count, invalid_annotations


def plot_distribution(data_dict, title, labels, output_path, colors=None):
    """Generic distribution plot function with consistent style."""
    plt.figure(figsize=(12, 6))

    if colors is None:
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(labels)]

    bars = plt.bar(labels, [data_dict[label] for label in labels], color=colors)

    plt.title(title, fontsize=14)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_combined_distribution(distributions, background_counts, splits, classes, output_dir):
    """Plot combined class distribution for all splits with consistent style."""
    os.makedirs(output_dir, exist_ok=True)

    # Colors for different classes
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'][:len(classes) + 1]

    # Create subplots for each split
    fig, axes = plt.subplots(len(splits) + 1, 1, figsize=(12, 5 * (len(splits) + 1)))

    # Plot each split's distribution
    for i, split in enumerate(splits):
        labels = classes + ['Background']
        counts = [distributions[split][cls] for cls in classes] + [background_counts[split]]
        bars = axes[i].bar(labels, counts, color=colors)

        axes[i].set_title(f'{split.capitalize()} Split Distribution', fontsize=12)
        axes[i].set_ylabel('Count', fontsize=10)

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                             f'{int(height)}', ha='center', va='bottom', fontsize=10)

    # Combined plot for all splits
    x = np.arange(len(classes) + 1)
    bar_width = 0.8 / len(splits)
    all_labels = classes + ['Background']

    for i, split in enumerate(splits):
        counts = [distributions[split][cls] for cls in classes] + [background_counts[split]]
        bars = axes[-1].bar(x + i * bar_width - 0.4 + bar_width / 2, counts,
                            width=bar_width, label=split.capitalize(), alpha=0.7)

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[-1].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                              f'{int(height)}', ha='center', va='bottom', fontsize=8)

    axes[-1].set_title('Combined Class Distribution Across Splits', fontsize=12)
    axes[-1].set_ylabel('Count', fontsize=10)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(all_labels, fontsize=10)
    axes[-1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()


def plot_corruption_stats(corruption_stats, splits, output_dir):
    """Plot corruption statistics with same style as distribution plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    categories = ['Normal', 'Corrupted Images', 'Corrupted Annotations', 'Missing Annotations', 'Invalid Annotations']
    colors = ['#99ff99', '#ff9999', '#ffcc99', '#c2c2f0', '#ff6666']

    # Create subplots for each split
    fig, axes = plt.subplots(len(splits) + 1, 1, figsize=(12, 5 * (len(splits) + 1)))

    # Plot each split's corruption stats
    for i, split in enumerate(splits):
        stats = corruption_stats[split]
        labels = categories
        counts = [
            stats['normal'],
            stats['corrupted_images'],
            stats['corrupted_annotations'],
            stats['missing_annotations'],
            stats['invalid_annotations']
        ]

        bars = axes[i].bar(labels, counts, color=colors)

        axes[i].set_title(f'{split.capitalize()} File Status', fontsize=12)
        axes[i].set_ylabel('Count', fontsize=10)

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                             f'{int(height)}', ha='center', va='bottom', fontsize=10)

    # Combined plot for all splits
    x = np.arange(len(categories))
    bar_width = 0.8 / len(splits)

    for i, split in enumerate(splits):
        stats = corruption_stats[split]
        counts = [
            stats['normal'],
            stats['corrupted_images'],
            stats['corrupted_annotations'],
            stats['missing_annotations'],
            stats['invalid_annotations']
        ]

        bars = axes[-1].bar(x + i * bar_width - 0.4 + bar_width / 2, counts,
                            width=bar_width, label=split.capitalize(), alpha=0.7)

        # Add count labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[-1].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                              f'{int(height)}', ha='center', va='bottom', fontsize=8)

    axes[-1].set_title('Combined File Status Across Splits', fontsize=12)
    axes[-1].set_ylabel('Count', fontsize=10)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(categories, fontsize=10)
    axes[-1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'corruption_stats.png'))
    plt.close()


def visualize_samples(dataset_base, splits, classes, num_samples, output_dir):
    """Visualize random samples from each split with bounding boxes and counts."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a single figure for all split samples
    rows = len(splits)
    cols = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # Handle the case when there's only one sample or one split
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for row, split in enumerate(splits):
        split_path = os.path.join(dataset_base, split)
        images_path = os.path.join(split_path, 'images')
        labels_path = os.path.join(split_path, 'labels')

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            for col in range(cols):
                axes[row, col].text(0.5, 0.5, f"No images in {split}",
                                    horizontalalignment='center', verticalalignment='center')
                axes[row, col].axis('off')
            continue

        # Get all image files
        image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            for col in range(cols):
                axes[row, col].text(0.5, 0.5, f"No images in {split}",
                                    horizontalalignment='center', verticalalignment='center')
                axes[row, col].axis('off')
            continue

        # Sample random images
        samples = random.sample(image_files, min(num_samples, len(image_files)))

        for col, sample in enumerate(samples):
            img_path = os.path.join(images_path, sample)
            label_path = os.path.join(labels_path, os.path.splitext(sample)[0] + '.txt')

            # Load image
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Could not read image")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f"Corrupted image\n{sample}",
                                    horizontalalignment='center', verticalalignment='center')
                axes[row, col].axis('off')
                continue

            h, w, _ = img.shape

            # Initialize class counts for this image
            image_class_counts = {cls: 0 for cls in classes}

            # Load and draw bounding boxes
            if not os.path.exists(label_path):
                cv2.putText(img, "MISSING ANNOTATION", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) != 5:
                                raise ValueError("Invalid annotation format")

                            class_id = int(parts[0])

                            if class_id < len(classes):
                                class_name = classes[class_id]
                                image_class_counts[class_name] += 1

                                x_center, y_center, box_width, box_height = map(float, parts[1:5])

                                # Convert to pixel coordinates
                                x1 = int((x_center - box_width / 2) * w)
                                y1 = int((y_center - box_height / 2) * h)
                                x2 = int((x_center + box_width / 2) * w)
                                y2 = int((y_center + box_height / 2) * h)

                                # Draw rectangle (different color for each class)
                                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                                color = colors[class_id % len(colors)]
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                                # Add label
                                cv2.putText(img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                except Exception as e:
                    # Mark as corrupted annotation
                    cv2.putText(img, "CORRUPTED ANNOTATION", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            axes[row, col].imshow(img)

            # Create title with counts
            count_str = ", ".join([f"{cls}: {count}" for cls, count in image_class_counts.items() if count > 0])
            title = f"{split.capitalize()} sample {col + 1}\n{count_str}"
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples_visualization.png'))
    plt.close()


def save_distribution_stats(distributions, background_counts, invalid_counts, splits, classes, output_dir):
    """Save distribution statistics to YAML file."""
    statistics = {
        'class_distribution': {},
        'annotation_quality': {}
    }

    # Class distribution stats
    class_stats = {}
    for split in splits:
        total_split = sum(distributions[split].values()) + background_counts[split]
        split_stats = {
            'total_instances': total_split,
            'classes': {}
        }

        for cls in classes:
            count = distributions[split][cls]
            percentage = (count / total_split * 100) if total_split > 0 else 0
            split_stats['classes'][cls] = {
                'count': count,
                'percentage': f"{percentage:.2f}%"
            }

        # Add background stats
        split_stats['classes']['Background'] = {
            'count': background_counts[split],
            'percentage': f"{(background_counts[split] / total_split * 100) if total_split > 0 else 0:.2f}%"
        }

        class_stats[split] = split_stats

    # Combined class stats
    total_instances = {cls: sum(distributions[split][cls] for split in splits) for cls in classes}
    total_background = sum(background_counts.values())
    total_all = sum(total_instances.values()) + total_background

    combined_class_stats = {
        'total_instances': total_all,
        'classes': {}
    }

    for cls in classes:
        count = total_instances[cls]
        percentage = (count / total_all * 100) if total_all > 0 else 0
        combined_class_stats['classes'][cls] = {
            'count': count,
            'percentage': f"{percentage:.2f}%"
        }

    combined_class_stats['classes']['Background'] = {
        'count': total_background,
        'percentage': f"{(total_background / total_all * 100) if total_all > 0 else 0:.2f}%"
    }

    statistics['class_distribution'] = {
        'splits': class_stats,
        'combined': combined_class_stats
    }

    # Annotation quality stats
    annotation_stats = {}
    for split in splits:
        annotation_stats[split] = {
            'invalid_annotations': invalid_counts[split]
        }

    statistics['annotation_quality'] = {
        'splits': annotation_stats,
        'combined': {
            'invalid_annotations': sum(invalid_counts.values())
        }
    }

    # Save to YAML file
    with open(os.path.join(output_dir, 'distribution.yaml'), 'w') as f:
        yaml.dump(statistics, f, sort_keys=False, default_flow_style=False)


def save_corruption_stats(corruption_stats, splits, output_dir):
    """Save corruption statistics to YAML file with same structure as distribution stats."""
    statistics = {}

    # Per split statistics
    split_stats = {}
    for split in splits:
        stats = corruption_stats[split]
        total_files = stats['normal'] + stats['corrupted_images'] + stats['corrupted_annotations'] + stats[
            'missing_annotations']

        file_stats = {
            'normal': {
                'count': stats['normal'],
                'percentage': f"{(stats['normal'] / total_files * 100) if total_files > 0 else 0:.2f}%"
            },
            'corrupted_images': {
                'count': stats['corrupted_images'],
                'percentage': f"{(stats['corrupted_images'] / total_files * 100) if total_files > 0 else 0:.2f}%"
            },
            'corrupted_annotations': {
                'count': stats['corrupted_annotations'],
                'percentage': f"{(stats['corrupted_annotations'] / total_files * 100) if total_files > 0 else 0:.2f}%"
            },
            'missing_annotations': {
                'count': stats['missing_annotations'],
                'percentage': f"{(stats['missing_annotations'] / total_files * 100) if total_files > 0 else 0:.2f}%"
            }
        }

        split_stats[split] = {
            'total_files': total_files,
            'file_status': file_stats
        }

    # Combined statistics
    combined_normal = sum(stats['normal'] for stats in corruption_stats.values())
    combined_corrupted_images = sum(stats['corrupted_images'] for stats in corruption_stats.values())
    combined_corrupted_annotations = sum(stats['corrupted_annotations'] for stats in corruption_stats.values())
    combined_missing = sum(stats['missing_annotations'] for stats in corruption_stats.values())
    total_all = combined_normal + combined_corrupted_images + combined_corrupted_annotations + combined_missing

    combined_stats = {
        'total_files': total_all,
        'file_status': {
            'normal': {
                'count': combined_normal,
                'percentage': f"{(combined_normal / total_all * 100) if total_all > 0 else 0:.2f}%"
            },
            'corrupted_images': {
                'count': combined_corrupted_images,
                'percentage': f"{(combined_corrupted_images / total_all * 100) if total_all > 0 else 0:.2f}%"
            },
            'corrupted_annotations': {
                'count': combined_corrupted_annotations,
                'percentage': f"{(combined_corrupted_annotations / total_all * 100) if total_all > 0 else 0:.2f}%"
            },
            'missing_annotations': {
                'count': combined_missing,
                'percentage': f"{(combined_missing / total_all * 100) if total_all > 0 else 0:.2f}%"
            }
        }
    }

    # Combine all statistics
    statistics = {
        'splits': split_stats,
        'combined': combined_stats
    }

    # Save to YAML file
    with open(os.path.join(output_dir, 'corruption.yaml'), 'w') as f:
        yaml.dump(statistics, f, sort_keys=False, default_flow_style=False)


def analyze_dataset(config):
    """Analyze dataset based on configuration."""
    print("Starting Dataset Analysis")

    dataset_base = config['dataset_base']
    data_yaml_path = config['data_yaml_path']
    splits = config['splits']
    num_samples = config['num_samples_per_split']
    output_dir = config['output_dir']

    if not check_dataset_splitting(dataset_base):
        print("Dataset not split yet. Performing splitting with ratio 60:20:20...")
        if not split_dataset(dataset_base):
            print("Error: Failed to split dataset. Please check your dataset structure.")
            return
    else:
        print("Dataset already split into train/val/test folders")

    os.makedirs(output_dir, exist_ok=True)

    yaml_data = load_yaml(data_yaml_path)
    classes = yaml_data.get('names', [])

    distributions = {}
    background_counts = {}
    invalid_annotations_counts = {}
    corruption_stats = {}

    for split in splits:
        split_path = os.path.join(dataset_base, split)

        if not os.path.exists(split_path):
            print(f"Warning: Split folder '{split}' not found in dataset directory")
            distributions[split] = {cls: 0 for cls in classes}
            background_counts[split] = 0
            invalid_annotations_counts[split] = 0
            corruption_stats[split] = {
                'corrupted_images': 0,
                'corrupted_annotations': 0,
                'missing_annotations': 0,
                'normal': 0,
                'invalid_annotations': 0
            }
            continue

        print(f"\nAnalyzing {split} split...")

        dist, bg_count, invalid_annos = get_class_distribution(split_path, classes)
        distributions[split] = dist
        background_counts[split] = bg_count
        invalid_annotations_counts[split] = invalid_annos

        corrupted_images, corrupted_annotations, normal_count, missing_annotations = check_corrupted_files(split_path)
        corruption_stats[split] = {
            'corrupted_images': len(corrupted_images),
            'corrupted_annotations': len(corrupted_annotations) + invalid_annos,
            'missing_annotations': missing_annotations,
            'normal': normal_count,
            'invalid_annotations': invalid_annos
        }

        print(f"Summary for {split}:")
        print(f"  - Normal files: {normal_count}")
        print(f"  - Corrupted images: {len(corrupted_images)}")
        print(f"  - Corrupted annotations: {len(corrupted_annotations)}")
        print(f"  - Invalid annotations: {invalid_annos}")
        print(f"  - Missing annotations: {missing_annotations}")
        print(f"  - Background images (no objects): {bg_count}")
        print("  Class distribution:")
        for cls, count in dist.items():
            print(f"    - {cls}: {count}")

    print("\nGenerating distribution plots...")
    plot_combined_distribution(distributions, background_counts, splits, classes, output_dir)

    print("Generating corruption statistics plots...")
    plot_corruption_stats(corruption_stats, splits, output_dir)

    print("Generating sample visualizations...")
    visualize_samples(dataset_base, splits, classes, num_samples, output_dir)

    print("Saving statistics to YAML files...")
    save_distribution_stats(distributions, background_counts, invalid_annotations_counts, splits, classes, output_dir)
    save_corruption_stats(corruption_stats, splits, output_dir)

    print("\nDataset Analysis Complete!")
    print(f"Results saved to: {output_dir}")


def check_dataset_splitting(dataset_base):
    """Check if dataset is already split into train/val/test folders."""
    required_folders = ['train', 'val', 'test']
    for folder in required_folders:
        if not os.path.exists(os.path.join(dataset_base, folder)):
            return False
    return True


def split_dataset(dataset_base, split_ratio=(0.6, 0.2, 0.2)):
    """Split dataset into train, val, test sets if not already split."""
    if check_dataset_splitting(dataset_base):
        print("Dataset already split into train/val/test folders")
        return True

    os.makedirs(os.path.join(dataset_base, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_base, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dataset_base, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_base, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dataset_base, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_base, 'test', 'labels'), exist_ok=True)

    root_images = os.path.join(dataset_base, 'images')
    root_labels = os.path.join(dataset_base, 'labels')

    if not os.path.exists(root_images) or not os.path.exists(root_labels):
        print("Error: No unsplit dataset found in root directory. Please check your dataset structure.")
        return False

    image_files = [f for f in os.listdir(root_images) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("Error: No images found in the dataset.")
        return False

    train_val, test = train_test_split(image_files, test_size=split_ratio[2], random_state=42)
    train, val = train_test_split(train_val, test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]),
                                  random_state=42)

    def copy_files(files, split_name):
        for file in files:
            src_img = os.path.join(root_images, file)
            dst_img = os.path.join(dataset_base, split_name, 'images', file)
            shutil.copy2(src_img, dst_img)

            label_file = os.path.splitext(file)[0] + '.txt'
            src_label = os.path.join(root_labels, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(dataset_base, split_name, 'labels', label_file)
                shutil.copy2(src_label, dst_label)

    copy_files(train, 'train')
    copy_files(val, 'val')
    copy_files(test, 'test')

    print(f"Dataset successfully split: Train ({len(train)}), Val ({len(val)}), Test ({len(test)})")
    return True


if __name__ == "__main__":
    CONFIG = {
        'dataset_base': '../dataset/HOME-FIRE',
        'data_yaml_path': '../dataset/HOME-FIRE/data.yaml',
        'splits': ['train', 'val', 'test'],
        'target_classes': ['Fire', 'Smoke'],
        'num_samples_per_split': 2,
        'output_dir': '../utils/figures/distribution'
    }

    analyze_dataset(CONFIG)