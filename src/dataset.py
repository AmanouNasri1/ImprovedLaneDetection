import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class BDD100K(Dataset):
    def __init__(self, images_dir, json_file, img_size=(512, 288), draw_width=3, conditions=None):
        """
        BDD100K dataset for lane detection.

        Args:
            images_dir (str): Root directory for training images.
            json_file (str): Path to the label JSON file.
            img_size (tuple): (width, height) to resize images and masks.
            draw_width (int): Thickness of drawn lane lines.
            conditions (list[str] or None): ["A"], ["B"], or ["A","B"] to select subfolders.
                                            If None, use all folders.
        """
        self.images_dir = os.path.normpath(images_dir)
        self.json_file = os.path.normpath(json_file)
        self.img_size = img_size
        self.draw_width = draw_width
        self.conditions = conditions  # e.g., ["A"], ["B"], or ["A","B"]

        # --- Load JSON once ---
        with open(self.json_file, 'r') as f:
            all_data = json.load(f)

        # --- Build label map: image name -> lane polylines ---
        self.label_map = {}
        for entry in all_data:
            lanes = []
            for obj in entry.get("labels", []):
                if obj.get("category") == "lane":
                    for shape in obj.get("poly2d", []):
                        pts = np.array(shape["vertices"], dtype=np.float32)
                        lanes.append(pts)
            if lanes:
                self.label_map[entry["name"]] = lanes

        # --- Recursively collect image paths ---
        all_images = []
        for root, _, files in os.walk(self.images_dir):
            # if using conditions, skip irrelevant folders
            if self.conditions:
                # check if this folder name matches any chosen condition
                if not any(cond.lower() in root.lower() for cond in self.conditions):
                    continue

            for f in files:
                if f.lower().endswith(".jpg"):
                    all_images.append(os.path.join(root, f))

        # --- Keep only images that have lane annotations ---
        self.samples = [
            img_path for img_path in all_images
            if os.path.basename(img_path) in self.label_map
        ]

        print(f"✅ Found {len(self.samples)} labeled images in {self.images_dir}")
        if self.conditions:
            print(f"   → Conditions: {self.conditions}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img_name = os.path.basename(img_path)
        lanes = self.label_map[img_name]

        # --- Load image ---
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img_resized = cv2.resize(img, self.img_size)

        # --- Create binary lane mask ---
        mask = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)
        scale_x = self.img_size[0] / w
        scale_y = self.img_size[1] / h
        for pts in lanes:
            scaled_pts = np.stack([pts[:, 0] * scale_x, pts[:, 1] * scale_y], axis=1)
            scaled_pts = np.round(scaled_pts).astype(np.int32)
            if len(scaled_pts) > 1:
                cv2.polylines(mask, [scaled_pts], False, 1, thickness=self.draw_width)

        # --- Convert to torch tensors ---
        img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {"image": img_tensor, "mask": mask_tensor, "lanes": lanes}
