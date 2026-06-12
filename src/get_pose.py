import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import urllib.request


class PPEPoseCropper:

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    MODEL_PATH = "pose_landmarker.task"

    def __init__(self, encoder_name="resnet18", device=None, min_visibility=0.5):

        self.min_visibility = min_visibility

        # pick GPU if available, otherwise fall back to CPU
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # download the mediapipe model if we don't have it yet
        if not os.path.exists(self.MODEL_PATH):
            print("Downloading pose landmarker model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)

        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence=0.5,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

        # load the encoder and strip the final classification head
        # so we only get the 512/2048-dim feature vector out
        if encoder_name == "resnet18":
            model = models.resnet18(weights="DEFAULT")
            self.feature_dim = 512
        elif encoder_name == "resnet50":
            model = models.resnet50(weights="DEFAULT")
            self.feature_dim = 2048
        else:
            raise ValueError("Unsupported encoder")

        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.encoder.to(self.device)
        self.encoder.eval()

        # standard imagenet normalisation that resnet expects
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # will hold the raw PoseLandmarker result after each detect call
        # so callers can access landmarks for drawing without re-running detection
        self.last_result = None

    # --- small utilities used by extract_regions ---

    def _get_point(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return [int(lm.x * w), int(lm.y * h)]

    def _crop_region(self, img, points, padding=30):
        points = np.array(points)
        x_min = max(int(points[:, 0].min()) - padding, 0)
        y_min = max(int(points[:, 1].min()) - padding, 0)
        x_max = min(int(points[:, 0].max()) + padding, img.shape[1])
        y_max = min(int(points[:, 1].max()) + padding, img.shape[0])
        return img[y_min:y_max, x_min:x_max]

    # --- run the crop through resnet and return the feature vector ---

    def encode_image(self, image):
        if image is None or image.size == 0:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.encoder(tensor)

        return features.squeeze().cpu().numpy()

    # --- main function: detect pose and cut out the four body regions ---

    def extract_regions(self, image):

        h, w = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        result = self.detector.detect(mp_image)

        # store it so the caller can grab it for landmark drawing
        self.last_result = result

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]  # just use the first person found

        # shoulder width gives us a body-relative scale for all the paddings below
        left_shoulder  = self._get_point(landmarks, 11, w, h)
        right_shoulder = self._get_point(landmarks, 12, w, h)

        shoulder_width = np.linalg.norm(
            np.array(left_shoulder) - np.array(right_shoulder)
        )

        # head region — we use nose + eyes + ears to locate it,
        # then extend upward by ~0.9x shoulder width to catch the helmet
        head_indices = [0, 2, 5, 7, 8]
        head_visibilities = [landmarks[i].visibility for i in head_indices if hasattr(landmarks[i], 'visibility')]
        if head_visibilities and np.mean(head_visibilities) < self.min_visibility:
            head_crop = None
        else:
            head_points = np.array([
                self._get_point(landmarks, 0, w, h),   # nose
                self._get_point(landmarks, 2, w, h),   # left eye
                self._get_point(landmarks, 5, w, h),   # right eye
                self._get_point(landmarks, 7, w, h),   # left ear
                self._get_point(landmarks, 8, w, h),   # right ear
            ])
            head_crop = image[
                max(int(head_points[:, 1].min() - shoulder_width * 0.9), 0)
                : min(int(head_points[:, 1].max() + shoulder_width * 0.15), h),
                max(int(head_points[:, 0].min() - shoulder_width * 0.35), 0)
                : min(int(head_points[:, 0].max() + shoulder_width * 0.35), w),
            ]

        # torso — shoulders to hips, covers the safety vest area
        torso_indices = [11, 12, 23, 24]
        torso_visibilities = [landmarks[i].visibility for i in torso_indices if hasattr(landmarks[i], 'visibility')]
        if torso_visibilities and np.mean(torso_visibilities) < self.min_visibility:
            torso_crop = None
        else:
            torso_crop = self._crop_region(
                image,
                [
                    self._get_point(landmarks, 11, w, h),  # left shoulder
                    self._get_point(landmarks, 12, w, h),  # right shoulder
                    self._get_point(landmarks, 23, w, h),  # left hip
                    self._get_point(landmarks, 24, w, h),  # right hip
                ],
                padding=int(shoulder_width * 0.3)
            )

        # left hand — wrist + knuckle landmarks to get a tight box
        left_hand_indices = [15, 17, 19, 21]
        left_hand_visibilities = [landmarks[i].visibility for i in left_hand_indices if hasattr(landmarks[i], 'visibility')]
        if left_hand_visibilities and np.mean(left_hand_visibilities) < self.min_visibility:
            left_hand_crop = None
        else:
            left_hand_crop = self._crop_region(
                image,
                [
                    self._get_point(landmarks, 15, w, h),  # left wrist
                    self._get_point(landmarks, 17, w, h),  # left pinky
                    self._get_point(landmarks, 19, w, h),  # left index
                    self._get_point(landmarks, 21, w, h),  # left thumb
                ],
                padding=int(shoulder_width * 0.2)
            )

        # right hand — same idea, mirrored
        right_hand_indices = [16, 18, 20, 22]
        right_hand_visibilities = [landmarks[i].visibility for i in right_hand_indices if hasattr(landmarks[i], 'visibility')]
        if right_hand_visibilities and np.mean(right_hand_visibilities) < self.min_visibility:
            right_hand_crop = None
        else:
            right_hand_crop = self._crop_region(
                image,
                [
                    self._get_point(landmarks, 16, w, h),  # right wrist
                    self._get_point(landmarks, 18, w, h),  # right pinky
                    self._get_point(landmarks, 20, w, h),  # right index
                    self._get_point(landmarks, 22, w, h),  # right thumb
                ],
                padding=int(shoulder_width * 0.2)
            )

        return {
            "head":       head_crop,
            "torso":      torso_crop,
            "left_hand":  left_hand_crop,
            "right_hand": right_hand_crop,
        }

    # --- convenience wrapper: extract regions then encode each one ---

    def extract_region_features(self, image):
        regions = self.extract_regions(image)

        if regions is None:
            return None

        return {name: self.encode_image(crop) for name, crop in regions.items()}