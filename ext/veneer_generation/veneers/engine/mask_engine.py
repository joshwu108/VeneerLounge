"""
Mask Engine

Handles ONLY tooth segmentation and mask refinement.
No diffusion, no conditioning, no post-processing.

Mask pipeline:
1. Segment teeth (UNet or HSV fallback)
2. Morphological close to fill holes
3. Contour smoothing to remove jaggedness
4. Gaussian blur + re-threshold for sub-pixel smooth edges
5. Conservative erosion (1-3px) to pull away from gumline
6. Output: strictly binary mask (0 or 255)

Feathering is a separate step, producing a gradient mask for
diffusion and blending (never used for whitening or conditioning).
"""

import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions as MpBaseOptions
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'individual_tooth_segmentation'))
from src.network.model import ResNeSt50_TC as TeethSegmentationNet


class MaskEngine:
    """
    Manages tooth segmentation and mask refinement.
    Single responsibility: generate and refine tooth masks.
    """

    def __init__(self, segmentation_checkpoint=None, device="cuda"):
        """
        Initialize mask engine.

        Args:
            segmentation_checkpoint: Path to tooth segmentation model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = self._resolve_device(device)
        self.seg_model = None

        if segmentation_checkpoint and segmentation_checkpoint != "None":
            try:
                self.seg_model = TeethSegmentationNet(in_ch=3, out_ch=1)
                checkpoint = torch.load(segmentation_checkpoint, map_location=self.device)

                # Handle different checkpoint formats
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "net_state_dict" in checkpoint:
                    state_dict = checkpoint["net_state_dict"]
                else:
                    state_dict = checkpoint

                # Remove 'module.' prefix if present (DataParallel)
                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                self.seg_model.load_state_dict(state_dict)
                self.seg_model.to(self.device)
                self.seg_model.eval()
                print("✓ Tooth segmentation model loaded")
            except Exception as e:
                print(f"Warning: Could not load segmentation model: {e}")
                print("Will use fallback color-based segmentation")
                self.seg_model = None
        else:
            print("Tooth segmentation model not provided - using fallback method")

        # Initialize MediaPipe FaceLandmarker for lip landmark detection
        self.face_landmarker = None
        if _MEDIAPIPE_AVAILABLE:
            try:
                # Locate the face_landmarker.task model file
                model_path = Path(__file__).parent.parent.parent / "models" / "face_landmarker.task"
                if not model_path.exists():
                    print(f"  Warning: MediaPipe model not found at {model_path}")
                    print("  Download from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
                else:
                    options = mp_vision.FaceLandmarkerOptions(
                        base_options=MpBaseOptions(model_asset_path=str(model_path)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                    )
                    self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                    print("  MediaPipe FaceLandmarker initialized")
            except Exception as e:
                print(f"  Warning: Could not initialize MediaPipe FaceLandmarker: {e}")
                self.face_landmarker = None
        else:
            print("  MediaPipe not available - lip-guided masking disabled")

    @staticmethod
    def _resolve_device(requested):
        """Resolve the best available device: cuda > mps > cpu."""
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def segment_raw(self, image, target_size=(512, 512),
                    lip_landmarks=None, crop_box=None, full_image_size=None):
        """
        Get raw segmentation mask before any refinement.

        Useful for debug visualization (comparing raw vs refined mask).

        Args:
            image: PIL Image
            target_size: Target size for segmentation
            lip_landmarks: dict from detect_face_landmarks() (optional)
            crop_box: tuple (x1, y1, x2, y2) (optional)
            full_image_size: tuple (w, h) (optional)

        Returns:
            PIL Image: Raw segmentation mask (before morphological ops)
        """
        if lip_landmarks is not None and crop_box is not None:
            mask = self._mediapipe_segment(
                image, lip_landmarks, crop_box, full_image_size, target_size
            )
            if mask is not None:
                return mask

        return self._segment(image, target_size)

    def generate(self, image, target_size=(512, 512), erosion_px=3,
                 lip_landmarks=None, crop_box=None, full_image_size=None):
        """
        Generate tooth mask with refinement.

        Strategy priority:
        1. MediaPipe lip-guided (if landmarks provided) — most reliable
        2. Neural model (ResNeSt50_TC) — for when MediaPipe unavailable
        3. HSV color-based — fallback if neural fails
        4. Intelligent rectangle — last resort

        Args:
            image: PIL Image
            target_size: Target size for mask generation
            erosion_px: Erosion amount for mask refinement (1-3 recommended)
            lip_landmarks: dict from detect_face_landmarks() (optional)
            crop_box: tuple (x1, y1, x2, y2) in full-image pixels (optional)
            full_image_size: tuple (w, h) of original full image (optional)

        Returns:
            PIL Image: Refined tooth mask (strictly binary 0/255, NOT feathered)
        """
        mask = None
        segmentation_method = "unknown"

        # Strategy 1: MediaPipe lip-guided segmentation (preferred)
        if lip_landmarks is not None and crop_box is not None:
            mask = self._mediapipe_segment(
                image, lip_landmarks, crop_box, full_image_size, target_size
            )
            if mask is not None:
                segmentation_method = "mediapipe"
                print(f"  Segmentation method: MediaPipe lip-guided")

        # Strategy 2+3: Neural model or HSV fallback
        if mask is None:
            segmentation_method = "neural" if self.seg_model else "color_hsv"
            print(f"  Segmentation method: {segmentation_method}")
            mask = self._segment(image, target_size)

        # Sanity check mask coverage
        mask_np = np.array(mask)
        mask_ratio = np.sum(mask_np > 127) / mask_np.size

        coverage_pct = mask_ratio * 100
        print(f"  Mask coverage: {coverage_pct:.1f}% (method: {segmentation_method})")
        if coverage_pct < 5:
            print(f"  WARNING: Mask coverage {coverage_pct:.1f}% < 5% — teeth may not be detected. "
                  "Check segmentation model or input crop region.")
        elif coverage_pct > 40:
            print(f"  WARNING: Mask coverage {coverage_pct:.1f}% > 40% — mask may include lips/gums. "
                  "Consider tighter bounding box or increased erosion.")

        # For non-MediaPipe methods, apply >50% coverage fallback
        # (MediaPipe masks are already geometrically constrained by lip polygon)
        if segmentation_method != "mediapipe" and mask_ratio > 0.50 and self.seg_model is not None:
            print(f"  Warning: segmentation mask covers {mask_ratio:.0%} of crop (> 50%) — "
                  f"likely failed. Falling back to color-based segmentation.")
            mask = self._fallback_segment(image, target_size)
            mask_np = np.array(mask)
            fallback_ratio = np.sum(mask_np > 127) / mask_np.size
            if fallback_ratio < 0.05:
                print(f"  Warning: Fallback segmentation also failed (coverage {fallback_ratio:.1%}). "
                      f"Using intelligent center rectangle based on image analysis.")
                mask = self._create_intelligent_rectangle_mask(image, target_size)

        mask = self._refine(mask, erosion_px)
        return mask

    def feather(self, mask, inner_feather_px=8, outer_feather_px=30):
        """
        Feather a binary mask to create smooth gradients for diffusion/blending.

        Uses distance transform on strictly binary input to create
        a smooth falloff from 255 (interior) to 0 (exterior).

        Args:
            mask: PIL Image (binary mask, must be 0/255 only)
            inner_feather_px: Inner feathering distance (controls gradient width)
            outer_feather_px: Outer feathering distance (unused for now)

        Returns:
            PIL Image: Feathered mask (grayscale gradient)
        """
        return self._feather(mask, inner_feather_px, outer_feather_px)

    def _segment(self, image, target_size=(512, 512)):
        """
        Segment teeth from image.

        Args:
            image: PIL Image
            target_size: Target size for segmentation

        Returns:
            PIL Image: Binary mask
        """
        if self.seg_model is None:
            return self._fallback_segment(image, target_size)

        image_resized = image.resize(target_size, Image.LANCZOS)
        img_array = np.array(image_resized)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            output = self.seg_model(img_tensor)
            mask = torch.sigmoid(output) > 0.5
            mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255

        # CRITICAL FIX: Detect if mask is inverted
        # The neural model appears to output inverted masks for this dataset
        h, w = mask.shape
        total_pixels = h * w
        white_pixels = np.sum(mask > 127)
        coverage_ratio = white_pixels / total_pixels
        
        # Strategy 1: If mask covers >80% of image, it's definitely inverted (teeth never cover that much)
        if coverage_ratio > 0.80:
            print(f"  Neural segmentation mask covers {coverage_ratio:.1%} - definitely inverted. Auto-correcting...")
            mask = 255 - mask
            coverage_ratio = 1.0 - coverage_ratio
        # Strategy 2: Check if edges are brighter than center (for partial inversions)
        elif coverage_ratio > 0.10:  # Only check if there's some content
            center_h_start, center_h_end = int(h * 0.3), int(h * 0.7)
            center_w_start, center_w_end = int(w * 0.3), int(w * 0.7)
            
            # Mean values in center and edges
            center_mean = np.mean(mask[center_h_start:center_h_end, center_w_start:center_w_end])
            edge_mask = mask.copy()
            edge_mask[center_h_start:center_h_end, center_w_start:center_w_end] = 0
            edge_pixels = total_pixels - (center_h_end - center_h_start) * (center_w_end - center_w_start)
            edge_mean = np.sum(edge_mask) / edge_pixels if edge_pixels > 0 else 0
            
            # If edges are significantly brighter than center, mask is inverted
            if edge_mean > center_mean + 30:  # Threshold: 30/255 difference
                print(f"  Neural segmentation mask appears inverted (edges={edge_mean:.1f}, center={center_mean:.1f}). Auto-correcting...")
                mask = 255 - mask

        return Image.fromarray(mask, mode="L")

    def _fallback_segment(self, image, target_size=(512, 512)):
        """
        Fallback color-based segmentation for teeth regions.

        Uses multi-strategy approach:
        1. Try HSV-based bright pixel detection (teeth are whitish)
        2. Combine with horizontal edge detection (teeth have edges near horizontal gumline)
        3. Filter by spatial position (teeth should be in center/upper portion of crop)
        4. Keep largest connected components (avoid noise)

        Args:
            image: PIL Image
            target_size: Target size

        Returns:
            PIL Image: Binary mask
        """
        image_resized = image.resize(target_size, Image.LANCZOS)
        img_array = np.array(image_resized)
        h, w = img_array.shape[:2]

        # Convert to HSV for white/bright detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Strategy 1: Detect bright, low-saturation pixels (teeth are whitish)
        # More permissive thresholds to catch off-white/yellowish teeth
        lower_white = np.array([0, 0, 150])  # Lowered from 170 to catch darker teeth
        upper_white = np.array([180, 50, 255])  # Increased saturation to catch warm tones
        mask_color = cv2.inRange(hsv, lower_white, upper_white)

        # Strategy 2: Edge detection to find tooth boundaries
        edges = cv2.Canny(gray, 30, 100)
        # Dilate edges slightly to make them more continuous
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel_edge, iterations=1)

        # Combine color and edge information
        # Areas that are both bright AND have edges nearby are likely teeth
        mask_combined = cv2.bitwise_and(mask_color, mask_color, mask=edges)
        # Also keep bright areas even without strong edges (middle of teeth)
        mask = cv2.bitwise_or(mask_combined, mask_color)

        # Spatial filter: focus on center horizontal band where teeth are likely
        # Zero out bottom 30% (chin/neck) and top 5% (if any)
        mask[int(h * 0.75):, :] = 0  # Bottom 25%
        mask[:int(h * 0.05), :] = 0  # Top 5%

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # Open to remove small noise blobs (skin highlights, reflections)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Close to fill gaps within tooth regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Keep only the largest 1-2 connected components (teeth region, possibly split)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            # Get areas of all components (excluding background at label 0)
            areas = stats[1:, cv2.CC_STAT_AREA]
            # Sort by area descending
            sorted_indices = np.argsort(areas)[::-1]

            # Keep top 2 components if the second is > 30% of the largest
            mask_new = np.zeros_like(mask)
            if len(sorted_indices) > 0:
                largest_label = sorted_indices[0] + 1
                mask_new[labels == largest_label] = 255

                # If there's a second component that's significant, keep it too
                if len(sorted_indices) > 1:
                    second_area = areas[sorted_indices[1]]
                    largest_area = areas[sorted_indices[0]]
                    if second_area > 0.3 * largest_area:
                        second_label = sorted_indices[1] + 1
                        mask_new[labels == second_label] = 255

            mask = mask_new

        return Image.fromarray(mask, mode="L")

    def _refine(self, mask, erosion_px=3):
        """
        Refine tooth mask with contour smoothing and morphological operations.

        Pipeline:
        1. Morphological close (7x7 ellipse, 2 iterations) to fill holes
        2. Contour approximation to smooth jagged boundaries
        3. Gaussian blur + re-threshold for sub-pixel smooth edges
        4. Conservative erosion (1-3px) to prevent gum/lip leakage

        Args:
            mask: PIL Image (binary mask)
            erosion_px: Erosion kernel size (1-3 recommended)

        Returns:
            PIL Image: Refined strictly binary mask
        """
        mask_np = np.array(mask)

        # 1. Morphological close to fill internal holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        refined = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # 2. Find contours and approximate with smooth polygons
        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smoothed = np.zeros_like(refined)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Drop small noise blobs
                continue
            # Approximate contour to remove jaggedness
            epsilon = 0.008 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(smoothed, [approx], -1, 255, cv2.FILLED)

        # 3. Gaussian blur then re-threshold for sub-pixel smooth edges
        smoothed = cv2.GaussianBlur(smoothed, (9, 9), 3)
        _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

        # 4. Conservative erosion to pull away from gumline/lips
        # 1-3px prevents lip leakage without shrinking below actual tooth area
        if erosion_px > 0:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px, erosion_px))
            smoothed = cv2.erode(smoothed, kernel_erode, iterations=1)

        # 5. Enforce strictly binary output (no soft gradients)
        _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

        return Image.fromarray(smoothed, mode="L")

    def _feather(self, mask, inner_feather_px=8, outer_feather_px=30):
        """
        Distance-based feathering with smooth boundary gradient.

        Strictly enforces binary input before distance transform
        to prevent soft gradient artifacts. Produces a mask that
        is 255 in the interior and smoothly falls to 0 at edges.

        Args:
            mask: PIL Image (binary mask)
            inner_feather_px: Inner feathering distance
            outer_feather_px: Outer feathering distance (unused for now)

        Returns:
            PIL Image: Feathered mask
        """
        # Enforce binary before distance transform (critical: no soft gradients)
        mask_np = np.array(mask)
        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        # Inward distance for soft edge falloff
        dist_in = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        feathered = np.clip(dist_in / inner_feather_px, 0, 1)

        # Smooth the boundary gradient
        feathered = cv2.GaussianBlur(feathered.astype(np.float32), (15, 15), 0)

        return Image.fromarray((feathered * 255).astype(np.uint8), mode="L")

    def _create_intelligent_rectangle_mask(self, image, target_size=(512, 512)):
        """
        Create an intelligent rectangle mask based on brightness analysis.

        Analyzes horizontal brightness profiles to find the brightest region,
        which is likely to be the teeth. More sophisticated than a fixed rectangle.

        Args:
            image: PIL Image
            target_size: Mask dimensions

        Returns:
            PIL Image: Intelligent rectangle mask
        """
        image_resized = image.resize(target_size, Image.LANCZOS)
        img_array = np.array(image_resized)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Compute horizontal brightness profile (average brightness per row)
        row_brightness = np.mean(gray, axis=1)

        # Find the brightest continuous region (likely teeth)
        # Use a sliding window to find peak average brightness
        window_size = int(h * 0.3)  # Look for teeth region ~30% of height
        best_y = 0
        best_avg = 0

        for y in range(h - window_size):
            avg_brightness = np.mean(row_brightness[y:y + window_size])
            if avg_brightness > best_avg:
                best_avg = avg_brightness
                best_y = y

        # Create mask centered on the brightest region
        mask_np = np.zeros((h, w), dtype=np.uint8)

        # Vertical bounds: centered on brightest region, ~35% of height
        h_start = max(0, best_y)
        h_end = min(h, best_y + window_size)

        # Horizontal bounds: center 70% of width (teeth span most of smile)
        w_start = int(w * 0.15)
        w_end = int(w * 0.85)

        # Create rounded rectangle for more natural look
        cv2.rectangle(mask_np, (w_start, h_start), (w_end, h_end), 255, -1)

        # Slight Gaussian blur to soften hard edges
        mask_np = cv2.GaussianBlur(mask_np, (15, 15), 0)
        _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        print(f"  Created intelligent rectangle mask: rows {h_start}-{h_end}, "
              f"cols {w_start}-{w_end} based on brightness analysis")

        return Image.fromarray(mask_np, mode="L")

    def create_fallback_mask(self, size=(768, 768)):
        """
        Create a simple center rectangle mask as fallback.

        Args:
            size: Mask dimensions (width, height)

        Returns:
            PIL Image: Center rectangle mask
        """
        w, h = size
        mask_np = np.zeros((h, w), dtype=np.uint8)
        h_start, h_end = int(h * 0.3), int(h * 0.7)
        w_start, w_end = int(w * 0.15), int(w * 0.85)
        mask_np[h_start:h_end, w_start:w_end] = 255
        return Image.fromarray(mask_np, mode="L")

    # ------------------------------------------------------------------ #
    #  MediaPipe Face Mesh — lip-guided tooth segmentation               #
    # ------------------------------------------------------------------ #

    def detect_face_landmarks(self, full_image):
        """
        Detect face landmarks from the full (uncropped) image using MediaPipe FaceLandmarker.

        Must be called on the FULL image before cropping, because MediaPipe needs
        the complete face structure (eyes, nose, jawline) to detect landmarks reliably.

        Args:
            full_image: PIL Image (full uncropped photo with complete face visible)

        Returns:
            dict or None: Landmark dict with pixel-coordinate lip points,
            or None if face/lips not detected.
        """
        if self.face_landmarker is None:
            return None

        img_np = np.array(full_image)
        h, w = img_np.shape[:2]

        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            print("  MediaPipe: No face detected in full image")
            return None

        face = results.face_landmarks[0]  # list of NormalizedLandmark

        # Inner lip landmark indices (mouth opening where teeth are visible)
        INNER_UPPER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        INNER_LOWER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        # Outer lip landmarks (for gum exclusion reference)
        OUTER_UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        OUTER_LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

        def extract_points(indices):
            points = []
            for idx in indices:
                lm = face[idx]
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append((px, py))
            return points

        landmarks = {
            "inner_upper_lip": extract_points(INNER_UPPER_LIP),
            "inner_lower_lip": extract_points(INNER_LOWER_LIP),
            "outer_upper_lip": extract_points(OUTER_UPPER_LIP),
            "outer_lower_lip": extract_points(OUTER_LOWER_LIP),
            "image_size": (w, h),
        }

        # Sanity check: inner lip polygon should have non-trivial area
        inner_poly = np.array(
            landmarks["inner_upper_lip"] + landmarks["inner_lower_lip"][::-1],
            dtype=np.float32,
        )
        area = cv2.contourArea(inner_poly)
        min_area = w * h * 0.001  # at least 0.1% of image
        if area < min_area:
            print(
                f"  MediaPipe: Lip polygon area too small ({area:.0f} < {min_area:.0f}px) "
                "— mouth likely closed"
            )
            return None

        print(f"  MediaPipe: Detected lip landmarks (polygon area: {area:.0f}px)")
        return landmarks

    def _mediapipe_segment(self, crop_image, lip_landmarks, crop_box,
                           full_image_size, target_size=(512, 512)):
        """
        Segment teeth using MediaPipe lip landmarks as the region-of-interest.

        1. Transform lip landmarks from full-image coords to crop/target coords
        2. Create inner-mouth polygon mask
        3. Within polygon: adaptive LAB thresholding to isolate teeth
        4. Morphological cleanup + connected-component filtering

        Args:
            crop_image: PIL Image (the 512x512 crop for diffusion)
            lip_landmarks: dict from detect_face_landmarks()
            crop_box: tuple (x1, y1, x2, y2) in full-image pixels
            full_image_size: tuple (w, h) of original full image
            target_size: tuple (w, h) for output mask

        Returns:
            PIL Image: Binary mask (0/255), or None on failure
        """
        if lip_landmarks is None:
            return None

        x1, y1, x2, y2 = crop_box
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w <= 0 or crop_h <= 0:
            return None

        tw, th = target_size
        scale_x = tw / crop_w
        scale_y = th / crop_h

        def transform_points(points):
            transformed = []
            for px, py in points:
                tx = int((px - x1) * scale_x)
                ty = int((py - y1) * scale_y)
                transformed.append((tx, ty))
            return transformed

        inner_upper = transform_points(lip_landmarks["inner_upper_lip"])
        inner_lower = transform_points(lip_landmarks["inner_lower_lip"])

        # Closed polygon: upper left→right, then lower right→left
        mouth_polygon = np.array(inner_upper + inner_lower[::-1], dtype=np.int32)

        # Clip polygon to image bounds
        mouth_polygon[:, 0] = np.clip(mouth_polygon[:, 0], 0, tw - 1)
        mouth_polygon[:, 1] = np.clip(mouth_polygon[:, 1], 0, th - 1)

        # Create mouth opening mask from polygon
        mouth_roi = np.zeros((th, tw), dtype=np.uint8)
        cv2.fillPoly(mouth_roi, [mouth_polygon], 255)

        roi_area = np.sum(mouth_roi > 0)
        total_area = tw * th
        if roi_area < total_area * 0.01:
            print(f"  MediaPipe segment: Mouth ROI too small ({roi_area / total_area:.1%})")
            return None

        # --- Within mouth ROI: adaptive LAB thresholding for teeth ---
        crop_np = np.array(crop_image.resize(target_size, Image.LANCZOS))
        lab = cv2.cvtColor(crop_np, cv2.COLOR_RGB2LAB)

        roi_mask_bool = mouth_roi > 0
        l_channel = lab[..., 0].astype(np.float32)
        a_channel = lab[..., 1].astype(np.float32)

        l_roi = l_channel[roi_mask_bool]
        if len(l_roi) < 50:
            print("  MediaPipe segment: Too few pixels in mouth ROI")
            return None

        # Otsu on L channel within mouth ROI — adaptively separates bright teeth
        # from dark mouth interior
        l_roi_uint8 = np.clip(l_roi, 0, 255).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(l_roi_uint8, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        l_threshold = max(float(otsu_thresh), 120.0)

        teeth_mask = np.zeros((th, tw), dtype=np.uint8)
        teeth_condition = (
            roi_mask_bool
            & (l_channel >= l_threshold)   # bright (teeth, not tongue/dark)
            & (a_channel < 148)            # not pink/red (gums)
            & (a_channel > 110)            # not green (artifact)
        )
        teeth_mask[teeth_condition] = 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Keep only connected components > 5% of mouth ROI area
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            teeth_mask, connectivity=8
        )
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            min_component_area = roi_area * 0.05
            teeth_clean = np.zeros_like(teeth_mask)
            for i in range(len(areas)):
                if areas[i] >= min_component_area:
                    teeth_clean[labels == (i + 1)] = 255
            teeth_mask = teeth_clean

        # Coverage check
        teeth_area = np.sum(teeth_mask > 0)
        if teeth_area < roi_area * 0.05:
            print(
                f"  MediaPipe segment: Teeth area too small ({teeth_area / max(roi_area, 1):.1%} of mouth ROI). "
                "Falling back to eroded mouth polygon."
            )
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mouth_roi_eroded = cv2.erode(mouth_roi, kernel_erode, iterations=2)
            return Image.fromarray(mouth_roi_eroded, mode="L")

        teeth_coverage = teeth_area / total_area
        print(
            f"  MediaPipe segment: Teeth coverage {teeth_coverage:.1%} "
            f"(within mouth ROI: {teeth_area / roi_area:.1%})"
        )

        return Image.fromarray(teeth_mask, mode="L")