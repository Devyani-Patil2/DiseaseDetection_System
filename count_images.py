import os, sys
sys.stdout.reconfigure(encoding='utf-8')

dataset_path = r"d:\2nd Year\Second sem\ML\CP\plantvillage dataset\color"
total = 0
classes = []

for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        total += count
        classes.append((folder, count))

# Write results to a file so we can read them
output_path = r"d:\2nd Year\Second sem\ML\CP\dataset_info.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for name, count in classes:
        f.write(f"{name}: {count} images\n")
    f.write(f"\n{'='*50}\n")
    f.write(f"TOTAL CLASSES: {len(classes)}\n")
    f.write(f"TOTAL IMAGES: {total}\n")
    if classes:
        f.write(f"AVG IMAGES PER CLASS: {total // len(classes)}\n")
        f.write(f"Min class: {min(classes, key=lambda x: x[1])}\n")
        f.write(f"Max class: {max(classes, key=lambda x: x[1])}\n")

    # Check a sample image
    sample_class = classes[0]
    sample_folder = os.path.join(dataset_path, sample_class[0])
    files = [ff for ff in os.listdir(sample_folder) if os.path.isfile(os.path.join(sample_folder, ff))]
    if files:
        sample_path = os.path.join(sample_folder, files[0])
        file_size = os.path.getsize(sample_path) / 1024
        f.write(f"\nSample image: {files[0]}\n")
        f.write(f"File size: {file_size:.1f} KB\n")
        try:
            from PIL import Image
            img = Image.open(sample_path)
            f.write(f"Image dimensions: {img.size}\n")
            f.write(f"Image mode: {img.mode}\n")
        except ImportError:
            f.write("PIL not available\n")

print("Done! Results saved to dataset_info.txt")
