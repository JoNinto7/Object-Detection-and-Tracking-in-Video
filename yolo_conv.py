import os

# === CONFIG ===
label_src = "data/data_tracking_label_2/training/label_02/"
image_base = "data/train/images/"
output_base = "data/train/labels/"
img_width = 1242
img_height = 375

# KITTI class to YOLO class mapping (can be simplified to car/person/etc.)
kitti_classes = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7
}

def convert_box(x1, y1, x2, y2, img_w, img_h):
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    x_center = (x1 + x2) / 2.0 * dw
    y_center = (y1 + y2) / 2.0 * dh
    width = (x2 - x1) * dw
    height = (y2 - y1) * dh
    return x_center, y_center, width, height

for seq in range(21):  # 0000 to 0020
    seq_id = f"{seq:04d}"
    label_file = os.path.join(label_src, f"{seq_id}.txt")

    if not os.path.exists(label_file):
        print(f"Missing: {label_file}")
        continue

    # Prepare output folder
    out_label_dir = os.path.join(output_base, seq_id)
    os.makedirs(out_label_dir, exist_ok=True)

    # Store labels by frame
    frame_dict = {}

    with open(label_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            frame = int(parts[0])
            obj_type = parts[2]
            if obj_type not in kitti_classes:
                continue  # skip "DontCare", etc.

            class_id = kitti_classes[obj_type]
            x1, y1, x2, y2 = map(float, parts[6:10])

            # Convert box to YOLO format
            x, y, w, h = convert_box(x1, y1, x2, y2, img_width, img_height)
            yolo_line = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

            if frame not in frame_dict:
                frame_dict[frame] = []
            frame_dict[frame].append(yolo_line)

    # Write out per-frame YOLO txt files
    for frame_id, lines in frame_dict.items():
        fname = f"{frame_id:06d}.txt"
        out_path = os.path.join(out_label_dir, fname)
        with open(out_path, "w") as out_f:
            out_f.write("\n".join(lines) + "\n")

    print(f"Converted: {label_file} → {out_label_dir}/")
