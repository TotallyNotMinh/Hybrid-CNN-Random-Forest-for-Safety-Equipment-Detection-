import os
import cv2
import torch
from get_pose import PPEPoseCropper

def main():
    test_folder = "testimage"
    output_folder = "cropped_image8"
    os.makedirs(output_folder, exist_ok=True)

    # Find the image
    possible_names = ["image (8).jpg", "image8.jpg", "image (8).png", "image8.png"]
    image_path = None
    for name in possible_names:
        temp_path = os.path.join(test_folder, name)
        if os.path.exists(temp_path):
            image_path = temp_path
            break

    if not image_path:
        print(f"Error: Could not find image8 in {test_folder}/ directory.")
        return

    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Failed to load image.")
        return

    print("Initializing PPEPoseCropper...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    cropper = PPEPoseCropper(encoder_name="resnet18", device=device)

    print("Running pose detection and extracting cropped regions...")
    regions = cropper.extract_regions(img)

    if regions is None:
        print("No pose detected in the image.")
        return

    # Save each crop
    saved_count = 0
    for region_name, crop in regions.items():
        if crop is not None and crop.size > 0:
            save_path = os.path.join(output_folder, f"{region_name}.jpg")
            cv2.imwrite(save_path, crop)
            print(f"Saved '{region_name}' crop to: {save_path}")
            saved_count += 1
        else:
            print(f"Region '{region_name}' not detected/visible.")

    print(f"\nSuccessfully cropped and saved {saved_count} regions to the '{output_folder}/' directory.")

if __name__ == "__main__":
    main()
