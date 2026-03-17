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
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def check_corrupted_files(dataset_path):
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
    class_counts = {cls: 0 for cls in classes}
    background_count = 0
    invalid_annotations = 0
    bbox_per_image = {}

    labels_path = os.path.join(dataset_path, 'labels')
    if not os.path.exists(labels_path):
        return class_counts, background_count, invalid_annotations, bbox_per_image

    for label_file in os.listdir(labels_path):
        if not label_file.endswith('.txt'):
            continue

        file_path = os.path.join(labels_path, label_file)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            valid_lines = [line for line in lines if line.strip()]

            if not valid_lines:
                background_count += 1
                continue

            image_bbox_count = 0

            for line in valid_lines:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                try:
                    class_id = int(parts[0])
                    if class_id < len(classes):
                        class_counts[classes[class_id]] += 1
                        image_bbox_count += 1
                except (ValueError, IndexError):
                    invalid_annotations += 1
                    continue

            if image_bbox_count > 0:
                if image_bbox_count not in bbox_per_image:
                    bbox_per_image[image_bbox_count] = 0
                bbox_per_image[image_bbox_count] += 1

        except Exception as e:
            print(f"Warning: Error processing {file_path}: {str(e)}")
            invalid_annotations += 1
            continue

    return class_counts, background_count, invalid_annotations, bbox_per_image


def plot_distribution(data_dict, title, output_path, colors=None):
    labels = list(data_dict.keys())

    plt.figure(figsize=(10, 6))

    if colors is None:
        colors = ['#ff6b6b', '#4ecdc4'][:len(labels)]

    bars = plt.bar(labels, [data_dict[label] for label in labels], color=colors, edgecolor='black', linewidth=1.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Classes', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height + max(data_dict.values()) * 0.01,
                     f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_bbox_per_image_distribution(bbox_distributions, splits, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    num_splits = len(splits)
    fig, axes = plt.subplots(num_splits + 1, 1, figsize=(12, 5 * (num_splits + 1)))

    if num_splits == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes

    split_colors = ['#e74c3c', '#3498db', '#2ecc71'] 

    for i, split in enumerate(splits):
        bbox_dist = bbox_distributions[split]

        if not bbox_dist:
            axes[i].text(0.5, 0.5, f"No data for {split}",
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(f'{split.capitalize()} - Bounding Boxes per Image',
                              fontsize=13, fontweight='bold')
            axes[i].axis('off')
            continue

        sorted_bbox_counts = sorted(bbox_dist.items())
        bbox_nums = [x[0] for x in sorted_bbox_counts]
        image_counts = [x[1] for x in sorted_bbox_counts]

        bars = axes[i].bar(bbox_nums, image_counts, color=split_colors[i % len(split_colors)],
                           edgecolor='black', linewidth=1.5, alpha=0.8)

        axes[i].set_title(f'{split.capitalize()} - Bounding Boxes per Image',
                          fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Number of Bounding Boxes', fontsize=11)
        axes[i].set_ylabel('Number of Images', fontsize=11)
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')

        if bbox_nums:
            axes[i].set_xticks(range(min(bbox_nums), max(bbox_nums) + 1))

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[i].text(bar.get_x() + bar.get_width() / 2., height + max(image_counts) * 0.01,
                             f'{int(height)}', ha='center', va='bottom',
                             fontsize=10, fontweight='bold')

        total_images = sum(image_counts)
        total_bboxes = sum(k * v for k, v in bbox_dist.items())
        avg_bbox = total_bboxes / total_images if total_images > 0 else 0
        max_bbox = max(bbox_nums) if bbox_nums else 0
        min_bbox = min(bbox_nums) if bbox_nums else 0

        stats_text = f'Total Images: {total_images} | Total BBoxes: {total_bboxes} | Avg: {avg_bbox:.2f} | Min: {min_bbox} | Max: {max_bbox}'
        axes[i].text(0.5, 0.95, stats_text, transform=axes[i].transAxes,
                     ha='center', va='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    all_bbox_nums = set()
    for bbox_dist in bbox_distributions.values():
        all_bbox_nums.update(bbox_dist.keys())

    if all_bbox_nums:
        all_bbox_nums = sorted(all_bbox_nums)
        x = np.arange(len(all_bbox_nums))
        bar_width = 0.25

        for i, split in enumerate(splits):
            bbox_dist = bbox_distributions[split]
            counts = [bbox_dist.get(num, 0) for num in all_bbox_nums]

            offset = (i - len(splits) / 2 + 0.5) * bar_width
            bars = axes[-1].bar(x + offset, counts, width=bar_width,
                                label=split.capitalize(), alpha=0.8,
                                color=split_colors[i % len(split_colors)],
                                edgecolor='black', linewidth=1)

            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    axes[-1].text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                                  f'{int(height)}', ha='center', va='bottom', fontsize=8)

        axes[-1].set_title('Combined Distribution - Bounding Boxes per Image',
                           fontsize=13, fontweight='bold')
        axes[-1].set_xlabel('Number of Bounding Boxes', fontsize=11)
        axes[-1].set_ylabel('Number of Images', fontsize=11)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(all_bbox_nums, fontsize=10)
        axes[-1].legend(fontsize=10, loc='upper right')
        axes[-1].grid(axis='y', alpha=0.3, linestyle='--')
    else:
        axes[-1].text(0.5, 0.5, "No annotated images found",
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[-1].transAxes, fontsize=12)
        axes[-1].set_title('Combined Distribution - Bounding Boxes per Image',
                           fontsize=13, fontweight='bold')
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbox_per_image_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_distribution(distributions, splits, classes, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    colors = ['#ff6b6b', '#4ecdc4']

    fig, axes = plt.subplots(len(splits) + 1, 1, figsize=(10, 5 * (len(splits) + 1)))

    if len(splits) == 1:
        axes = [axes]

    for i, split in enumerate(splits):
        counts = [distributions[split][cls] for cls in classes]
        bars = axes[i].bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)

        axes[i].set_title(f'{split.capitalize()} Split - Class Distribution', fontsize=13, fontweight='bold')
        axes[i].set_ylabel('Count', fontsize=11)
        axes[i].set_xlabel('Classes', fontsize=11)
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[i].text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                             f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    x = np.arange(len(classes))
    bar_width = 0.25
    split_colors = ['#e74c3c', '#3498db', '#2ecc71']

    for i, split in enumerate(splits):
        counts = [distributions[split][cls] for cls in classes]
        offset = (i - len(splits) / 2 + 0.5) * bar_width
        bars = axes[-1].bar(x + offset, counts, width=bar_width,
                            label=split.capitalize(), alpha=0.8,
                            color=split_colors[i % len(split_colors)],
                            edgecolor='black', linewidth=1)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[-1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                              f'{int(height)}', ha='center', va='bottom', fontsize=9)

    axes[-1].set_title('Combined Class Distribution Across Splits', fontsize=13, fontweight='bold')
    axes[-1].set_ylabel('Count', fontsize=11)
    axes[-1].set_xlabel('Classes', fontsize=11)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(classes, fontsize=11)
    axes[-1].legend(fontsize=10, loc='upper right')
    axes[-1].grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_corruption_stats(corruption_stats, splits, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    categories = ['Normal', 'Corrupted Images', 'Corrupted Annotations', 'Missing Annotations', 'Invalid Annotations']
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#e67e22']

    fig, axes = plt.subplots(len(splits) + 1, 1, figsize=(12, 5 * (len(splits) + 1)))

    if len(splits) == 1:
        axes = [axes]

    for i, split in enumerate(splits):
        stats = corruption_stats[split]
        counts = [
            stats['normal'],
            stats['corrupted_images'],
            stats['corrupted_annotations'],
            stats['missing_annotations'],
            stats['invalid_annotations']
        ]

        bars = axes[i].bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)

        axes[i].set_title(f'{split.capitalize()} - File Status', fontsize=13, fontweight='bold')
        axes[i].set_ylabel('Count', fontsize=11)
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[i].text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                             f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    x = np.arange(len(categories))
    bar_width = 0.25
    split_colors = ['#e74c3c', '#3498db', '#2ecc71']

    for i, split in enumerate(splits):
        stats = corruption_stats[split]
        counts = [
            stats['normal'],
            stats['corrupted_images'],
            stats['corrupted_annotations'],
            stats['missing_annotations'],
            stats['invalid_annotations']
        ]

        offset = (i - len(splits) / 2 + 0.5) * bar_width
        bars = axes[-1].bar(x + offset, counts, width=bar_width,
                            label=split.capitalize(), alpha=0.8,
                            color=split_colors[i % len(split_colors)],
                            edgecolor='black', linewidth=1)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[-1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                              f'{int(height)}', ha='center', va='bottom', fontsize=8)

    axes[-1].set_title('Combined File Status Across Splits', fontsize=13, fontweight='bold')
    axes[-1].set_ylabel('Count', fontsize=11)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(categories, fontsize=10, rotation=15, ha='right')
    axes[-1].legend(fontsize=10, loc='upper right')
    axes[-1].grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'corruption_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_samples(dataset_base, splits, classes, num_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    rows = len(splits)
    cols = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

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

        image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            for col in range(cols):
                axes[row, col].text(0.5, 0.5, f"No images in {split}",
                                    horizontalalignment='center', verticalalignment='center')
                axes[row, col].axis('off')
            continue

        samples = random.sample(image_files, min(num_samples, len(image_files)))

        for col, sample in enumerate(samples):
            img_path = os.path.join(images_path, sample)
            label_path = os.path.join(labels_path, os.path.splitext(sample)[0] + '.txt')

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
            image_class_counts = {cls: 0 for cls in classes}

            if not os.path.exists(label_path):
                cv2.putText(img, "MISSING ANNOTATION", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
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

                                x1 = int((x_center - box_width / 2) * w)
                                y1 = int((y_center - box_height / 2) * h)
                                x2 = int((x_center + box_width / 2) * w)
                                y2 = int((y_center + box_height / 2) * h)

                                colors_bbox = [(255, 0, 0), (0, 255, 255)]
                                color = colors_bbox[class_id % len(colors_bbox)]
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                                cv2.putText(img, class_name, (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception as e:
                    cv2.putText(img, "CORRUPTED ANNOTATION", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            axes[row, col].imshow(img)

            count_str = ", ".join([f"{cls}: {count}" for cls, count in image_class_counts.items() if count > 0])
            if not count_str:
                count_str = "No annotations"
            title = f"{split.capitalize()} #{col + 1}\n{count_str}"
            axes[row, col].set_title(title, fontsize=11, fontweight='bold')
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_distribution_stats(distributions, background_counts, invalid_counts, bbox_per_image_dists, splits, classes,
                            output_dir):
    statistics = {
        'class_distribution': {},
        'bbox_per_image_distribution': {},
        'background_info': {},
        'annotation_quality': {}
    }

    class_stats = {}
    for split in splits:
        total_annotations = sum(distributions[split].values())
        split_stats = {
            'total_annotations': total_annotations,
            'classes': {}
        }

        for cls in classes:
            count = distributions[split][cls]
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            split_stats['classes'][cls] = {
                'count': count,
                'percentage': f"{percentage:.2f}%"
            }

        class_stats[split] = split_stats

    total_annotations = {cls: sum(distributions[split][cls] for split in splits) for cls in classes}
    total_all_annotations = sum(total_annotations.values())

    combined_class_stats = {
        'total_annotations': total_all_annotations,
        'classes': {}
    }

    for cls in classes:
        count = total_annotations[cls]
        percentage = (count / total_all_annotations * 100) if total_all_annotations > 0 else 0
        combined_class_stats['classes'][cls] = {
            'count': count,
            'percentage': f"{percentage:.2f}%"
        }

    statistics['class_distribution'] = {
        'splits': class_stats,
        'combined': combined_class_stats
    }

    bbox_stats = {}
    for split in splits:
        bbox_dist = bbox_per_image_dists[split]
        total_images = sum(bbox_dist.values())
        total_bboxes = sum(k * v for k, v in bbox_dist.items())

        bbox_detail = {}
        for num_bbox, num_images in sorted(bbox_dist.items()):
            percentage = (num_images / total_images * 100) if total_images > 0 else 0
            bbox_detail[f'{num_bbox}_bboxes'] = {
                'image_count': num_images,
                'percentage': f"{percentage:.2f}%",
                'total_bboxes': num_bbox * num_images
            }

        avg_bbox = total_bboxes / total_images if total_images > 0 else 0
        bbox_stats[split] = {
            'total_images_with_annotations': total_images,
            'total_bounding_boxes': total_bboxes,
            'average_bbox_per_image': round(avg_bbox, 2),
            'min_bbox_per_image': min(bbox_dist.keys()) if bbox_dist else 0,
            'max_bbox_per_image': max(bbox_dist.keys()) if bbox_dist else 0,
            'distribution': bbox_detail
        }

    all_bbox_nums = set()
    for bbox_dist in bbox_per_image_dists.values():
        all_bbox_nums.update(bbox_dist.keys())

    combined_bbox_dist = {}
    for num_bbox in all_bbox_nums:
        combined_bbox_dist[num_bbox] = sum(
            bbox_per_image_dists[split].get(num_bbox, 0) for split in splits
        )

    total_images_combined = sum(combined_bbox_dist.values())
    total_bboxes_combined = sum(k * v for k, v in combined_bbox_dist.items())
    avg_bbox_combined = total_bboxes_combined / total_images_combined if total_images_combined > 0 else 0

    combined_bbox_detail = {}
    for num_bbox, num_images in sorted(combined_bbox_dist.items()):
        percentage = (num_images / total_images_combined * 100) if total_images_combined > 0 else 0
        combined_bbox_detail[f'{num_bbox}_bboxes'] = {
            'image_count': num_images,
            'percentage': f"{percentage:.2f}%",
            'total_bboxes': num_bbox * num_images
        }

    statistics['bbox_per_image_distribution'] = {
        'splits': bbox_stats,
        'combined': {
            'total_images_with_annotations': total_images_combined,
            'total_bounding_boxes': total_bboxes_combined,
            'average_bbox_per_image': round(avg_bbox_combined, 2),
            'min_bbox_per_image': min(combined_bbox_dist.keys()) if combined_bbox_dist else 0,
            'max_bbox_per_image': max(combined_bbox_dist.keys()) if combined_bbox_dist else 0,
            'distribution': combined_bbox_detail
        }
    }

    background_info = {}
    for split in splits:
        background_info[split] = {
            'count': background_counts[split],
            'note': 'Images without any annotations (not used in training metrics)'
        }

    statistics['background_info'] = {
        'splits': background_info,
        'combined': {
            'count': sum(background_counts.values()),
            'note': 'Total images without annotations across all splits'
        }
    }

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

    with open(os.path.join(output_dir, 'distribution.yaml'), 'w') as f:
        yaml.dump(statistics, f, sort_keys=False, default_flow_style=False)


def save_corruption_stats(corruption_stats, splits, output_dir):
    statistics = {}

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

    statistics = {
        'splits': split_stats,
        'combined': combined_stats
    }

    with open(os.path.join(output_dir, 'corruption.yaml'), 'w') as f:
        yaml.dump(statistics, f, sort_keys=False, default_flow_style=False)


def analyze_dataset(config):
    print("Starting Dataset Analysis")

    dataset_base = config['dataset_base']
    data_yaml_path = config['data_yaml_path']
    splits = config['splits']
    num_samples = config['num_samples_per_split']
    output_dir = config['output_dir']

    if not check_dataset_splitting(dataset_base):
        print("\nDataset not split yet. Performing splitting with ratio 60:20:20...")
        if not split_dataset(dataset_base):
            print("Error: Failed to split dataset. Please check your dataset structure.")
            return
    else:
        print("\nDataset already split into train/val/test folders")

    os.makedirs(output_dir, exist_ok=True)

    yaml_data = load_yaml(data_yaml_path)
    classes = yaml_data.get('names', [])

    print(f"\nDetected classes: {classes}")
    print(f"Number of classes: {len(classes)}")

    distributions = {}
    background_counts = {}
    invalid_annotations_counts = {}
    corruption_stats = {}
    bbox_per_image_distributions = {}

    for split in splits:
        split_path = os.path.join(dataset_base, split)

        if not os.path.exists(split_path):
            print(f"\nWarning: Split folder '{split}' not found")
            distributions[split] = {cls: 0 for cls in classes}
            background_counts[split] = 0
            invalid_annotations_counts[split] = 0
            bbox_per_image_distributions[split] = {}
            corruption_stats[split] = {
                'corrupted_images': 0,
                'corrupted_annotations': 0,
                'missing_annotations': 0,
                'normal': 0,
                'invalid_annotations': 0
            }
            continue

        print(f"Analyzing {split.upper()} split...")

        dist, bg_count, invalid_annos, bbox_dist = get_class_distribution(split_path, classes)
        distributions[split] = dist
        background_counts[split] = bg_count
        invalid_annotations_counts[split] = invalid_annos
        bbox_per_image_distributions[split] = bbox_dist

        corrupted_images, corrupted_annotations, normal_count, missing_annotations = check_corrupted_files(split_path)
        corruption_stats[split] = {
            'corrupted_images': len(corrupted_images),
            'corrupted_annotations': len(corrupted_annotations) + invalid_annos,
            'missing_annotations': missing_annotations,
            'normal': normal_count,
            'invalid_annotations': invalid_annos
        }

        print(f"\nFile Status:")
        print(f"Normal files: {normal_count}")
        print(f"Corrupted images: {len(corrupted_images)}")
        print(f"Corrupted annotations: {len(corrupted_annotations)}")
        print(f"Invalid annotations: {invalid_annos}")
        print(f"Missing annotations: {missing_annotations}")

        print(f"\nBackground Info:")
        print(f"Images without annotations: {bg_count}")
        print(f"(Note: Not included in class distribution)")

        print(f"\nClass Distribution (Annotated Objects Only):")
        total_annotations = sum(dist.values())
        for cls, count in dist.items():
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"{cls}: {count} ({percentage:.2f}%)")

        print(f"\nBounding Boxes per Image Distribution:")
        if bbox_dist:
            total_images = sum(bbox_dist.values())
            total_bboxes = sum(k * v for k, v in bbox_dist.items())
            avg_bbox = total_bboxes / total_images if total_images > 0 else 0
            print(f"Total images with annotations: {total_images}")
            print(f"Total bounding boxes: {total_bboxes}")
            print(f"Average bboxes per image: {avg_bbox:.2f}")
            print(f"Distribution:")
            for num_bbox in sorted(bbox_dist.keys()):
                num_images = bbox_dist[num_bbox]
                percentage = (num_images / total_images * 100) if total_images > 0 else 0
                print(f"  {num_bbox} bbox(es): {num_images} images ({percentage:.2f}%)")
        else:
            print(f"  No annotated images found")

    print("Generating visualizations...")

    print("\n1. Creating class distribution plots (Fire & Smoke only)...")
    plot_combined_distribution(distributions, splits, classes, output_dir)

    print("2. Creating bounding box per image distribution plots...")
    plot_bbox_per_image_distribution(bbox_per_image_distributions, splits, output_dir)

    print("3. Creating corruption statistics plots...")
    plot_corruption_stats(corruption_stats, splits, output_dir)

    print("4. Creating sample visualizations with bounding boxes...")
    visualize_samples(dataset_base, splits, classes, num_samples, output_dir)

    print("\n5. Saving statistics to YAML files...")
    save_distribution_stats(distributions, background_counts, invalid_annotations_counts,
                            bbox_per_image_distributions, splits, classes, output_dir)
    save_corruption_stats(corruption_stats, splits, output_dir)

    print("Dataset Analysis Complete!")

    print(f"Results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"class_distribution.png (Fire & Smoke distribution)")
    print(f"bbox_per_image_distribution.png (BBoxes per image)")
    print(f"corruption_stats.png")
    print(f"samples_visualization.png")
    print(f"distribution.yaml")
    print(f"corruption.yaml")


def check_dataset_splitting(dataset_base):
    required_folders = ['train', 'val', 'test']
    for folder in required_folders:
        if not os.path.exists(os.path.join(dataset_base, folder)):
            return False
    return True


def split_dataset(dataset_base, split_ratio=(0.6, 0.2, 0.2)):
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
        'dataset_base': 'dataset/HOME-FIRE',
        'data_yaml_path': 'dataset/HOME-FIRE/data.yaml',
        'splits': ['train', 'val', 'test'],
        'target_classes': ['Fire', 'Smoke'],
        'num_samples_per_split': 2,
        'output_dir': 'utils/figures/distribution'
    }

    analyze_dataset(CONFIG)
