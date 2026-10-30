"""Spatial domain low-pass, edge-preserving, point transformation, and order-statistic filters.

Academic Overview (Syllabus Units I & II):
    - Grayscale Conversion:
      Luminance projection: Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601).
    - Intensity Histogram & Statistical Moments:
      Discrete probability distribution of 8-bit pixel values across [0, 255],
      computing Mean, Variance, Standard Deviation, Skewness, and Kurtosis.
    - Histogram Equalization:
      Non-linear monotonic mapping using the cumulative distribution function (CDF)
      to linearize the intensity distribution and enhance global dynamic range.
    - Point Transformations:
      - Power-law (Gamma) Transformation: s = c * r^gamma
      - Logarithmic Transformation: s = c * ln(1 + r)
      - Image Negative: s = 255 - r
      - Bit-Plane Slicing: Decomposes 8-bit pixels into 8 binary spatial planes.
      - Gray-Level Quantization: Simulates reduced bit-depth resolution.
    - Contrast Stretching:
      Linear min-max normalization stretching dynamic range across [0, 255].
    - Spatial Smoothing Filters:
      - Gaussian Low-Pass Filter: Isotropic Gaussian kernel G(x, y).
      - Median Filter: Non-linear rank-order filter for salt-and-pepper noise.
      - Bilateral Filter: Edge-preserving filter combining spatial and range kernels.
      - Box / Mean Filter: Uniform spatial averaging kernel.
      - Min / Max / Midpoint Filters: Order-statistic neighborhood operators.
      - Alpha-Trimmed Mean Filter: Trims extreme order statistics before averaging.
    - Sharpening:
      - Unsharp Masking & High-Boost Filtering: I_sharp = I + k * (I - I_smooth).
"""

from typing import Sequence
import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert input image array to 8-bit single-channel grayscale.

    Args:
        image: Input array (2D grayscale, 3-channel BGR, or 4-channel BGRA).

    Returns:
        2D uint8 grayscale numpy array.

    Raises:
        ValueError: If input image is None, empty, or has unsupported channel count.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot convert empty or None image to grayscale.")

    if image.ndim == 2:
        return image.copy().astype(np.uint8)

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return image[:, :, 0].copy().astype(np.uint8)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def compute_histogram(gray_image: np.ndarray) -> np.ndarray:
    """Compute 256-bin 1D intensity histogram for an 8-bit grayscale image.

    Args:
        gray_image: 2D uint8 grayscale image.

    Returns:
        1D float32 numpy array of shape (256,) containing frequency counts.
        The sum of elements equals the total number of pixels.

    Raises:
        ValueError: If input image is empty or not single-channel.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot compute histogram on empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}")

    hist = cv2.calcHist([gray_image.astype(np.uint8)], [0], None, [256], [0, 256])
    return hist.flatten().astype(np.float32)


def compute_histogram_statistics(gray_image: np.ndarray) -> dict[str, float]:
    """Calculate statistical moments of image intensity distribution.

    Formulas:
        Mean: mu = (1 / N) * sum(I_i)
        Variance: sigma^2 = (1 / N) * sum((I_i - mu)^2)
        Standard Deviation: sigma = sqrt(sigma^2)
        Skewness: S = (1 / (N * sigma^3)) * sum((I_i - mu)^3)
        Kurtosis: K = (1 / (N * sigma^4)) * sum((I_i - mu)^4)

    Args:
        gray_image: 2D uint8 grayscale image.

    Returns:
        Dictionary containing 'mean', 'variance', 'std_dev', 'skewness', and 'kurtosis'.
    """
    if gray_image is None or gray_image.size == 0:
        return {"mean": 0.0, "variance": 0.0, "std_dev": 0.0, "skewness": 0.0, "kurtosis": 0.0}

    arr = gray_image.astype(np.float64).ravel()
    n = arr.size
    mean = float(np.mean(arr))
    variance = float(np.var(arr))
    std_dev = float(np.std(arr))

    if std_dev > 1e-6:
        normalized = (arr - mean) / std_dev
        skewness = float(np.mean(normalized ** 3))
        kurtosis = float(np.mean(normalized ** 4))
    else:
        skewness = 0.0
        kurtosis = 0.0

    return {
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def equalize_histogram(gray_image: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization to enhance image dynamic range.

    Transforms intensity values via normalized cumulative distribution function:
        s_k = T(r_k) = (L - 1) * sum_{j=0}^k (n_j / N)

    Args:
        gray_image: 2D uint8 grayscale image.

    Returns:
        Equalized 2D uint8 grayscale image with linearized cumulative histogram.

    Raises:
        ValueError: If input image is empty or not single-channel.
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot equalize empty or None image.")

    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}")

    return cv2.equalizeHist(gray_image.astype(np.uint8))


def apply_gamma_transform(image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """Apply Power-Law (Gamma) radiometric intensity transformation.

    Formula:
        s = c * (r / 255.0)^gamma * 255.0

    Args:
        image: Input 2D grayscale or 3D color uint8 array.
        gamma: Exponent parameter (gamma < 1 expands shadows, gamma > 1 compresses highlights).
        c: Scaling constant (default: 1.0).

    Returns:
        Gamma-transformed uint8 array.
    """
    if image is None or image.size == 0:
        return image

    gamma_safe = max(0.01, float(gamma))
    # Pre-build lookup table for O(1) pixel transformation
    lut = np.array([((i / 255.0) ** gamma_safe) * 255.0 * c for i in range(256)]).clip(0, 255).astype(np.uint8)
    return cv2.LUT(image.astype(np.uint8), lut)


def apply_log_transform(image: np.ndarray, c: float | None = None) -> np.ndarray:
    """Apply Logarithmic dynamic range transformation.

    Formula:
        s = c * ln(1 + r)
        c = 255 / ln(1 + max(r))

    Args:
        image: Input 2D grayscale or 3D color uint8 array.
        c: Optional scaling constant. If None, normalized automatically.

    Returns:
        Log-transformed uint8 array.
    """
    if image is None or image.size == 0:
        return image

    arr = image.astype(np.float32)
    max_val = float(np.max(arr))
    if max_val <= 0:
        return image.copy()

    scale = c if c is not None else (255.0 / np.log(1.0 + max_val))
    transformed = scale * np.log(1.0 + arr)
    return np.clip(transformed, 0, 255).astype(np.uint8)


def apply_image_negative(image: np.ndarray) -> np.ndarray:
    """Compute photographic negative of an image: s = 255 - r."""
    if image is None or image.size == 0:
        return image
    return (255 - image.astype(np.uint8)).astype(np.uint8)


def apply_contrast_stretching(
    gray_image: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """Apply linear min-max contrast stretching across specified percentile boundaries.

    Formula:
        s = ((r - r_min) / (r_max - r_min)) * 255.0

    Args:
        gray_image: 2D uint8 grayscale image.
        low_percentile: Lower intensity percentile cut-off (default: 2.0%).
        high_percentile: Upper intensity percentile cut-off (default: 98.0%).

    Returns:
        Contrast-stretched 2D uint8 grayscale image.
    """
    if gray_image is None or gray_image.size == 0:
        return gray_image

    r_min = float(np.percentile(gray_image, low_percentile))
    r_max = float(np.percentile(gray_image, high_percentile))

    if r_max - r_min < 1e-4:
        return gray_image.copy()

    stretched = ((gray_image.astype(np.float32) - r_min) / (r_max - r_min) * 255.0)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def extract_bit_planes(gray_image: np.ndarray) -> list[np.ndarray]:
    """Decompose an 8-bit grayscale image into its 8 binary bit planes (Bit 0 to Bit 7).

    Args:
        gray_image: 2D uint8 grayscale image.

    Returns:
        List of 8 binary arrays (each with values in {0, 255}) from LSB (0) to MSB (7).
    """
    if gray_image is None or gray_image.size == 0:
        return []

    planes: list[np.ndarray] = []
    arr = gray_image.astype(np.uint8)
    for bit in range(8):
        mask = (arr >> bit) & 1
        plane = (mask * 255).astype(np.uint8)
        planes.append(plane)
    return planes


def simulate_quantization(gray_image: np.ndarray, bits: int = 4) -> np.ndarray:
    """Simulate reduced gray-level quantization resolution (1 to 8 bits).

    Args:
        gray_image: 2D uint8 grayscale image.
        bits: Target bit depth in [1, 8].

    Returns:
        Quantized uint8 image exhibiting discrete gray-level contouring.
    """
    if gray_image is None or gray_image.size == 0:
        return gray_image

    b = max(1, min(8, int(bits)))
    shift = 8 - b
    # Truncate lower bits and map back to full dynamic range
    quantized = ((gray_image >> shift) << shift).astype(np.uint8)
    return quantized


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    """Apply 2D Gaussian low-pass spatial smoothing filter.

    Args:
        image: Input 2D grayscale or 3D color image array.
        kernel_size: Size of square Gaussian kernel (must be positive odd integer).
        sigma: Standard deviation in X/Y (if 0.0, computed from kernel size).

    Returns:
        Filtered image array with same shape and dtype as input.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot apply Gaussian blur to empty or None image.")

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Gaussian kernel_size must be a positive odd integer, got {kernel_size}")

    if sigma < 0.0:
        raise ValueError(f"Gaussian sigma must be non-negative, got {sigma}")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply non-linear median rank-order filter for impulsive noise suppression.

    Replaces each pixel with the statistical median of the kernel aperture.

    Args:
        image: Input 2D grayscale or 3D color image array.
        kernel_size: Aperture linear size (must be positive odd integer > 1).

    Returns:
        Median-filtered image array with same shape and dtype as input.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot apply median filter to empty or None image.")

    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError(f"Median kernel_size must be an odd integer >= 3, got {kernel_size}")

    return cv2.medianBlur(image, kernel_size)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply edge-preserving bilateral smoothing filter.

    Combines spatial closeness Gaussian weights with photometric radiometric similarity weights.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot apply bilateral filter to empty or None image.")

    if sigma_color <= 0.0 or sigma_space <= 0.0:
        raise ValueError("Bilateral filter sigma_color and sigma_space must be positive floats.")

    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_box_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply uniform box / mean averaging filter."""
    if image is None or image.size == 0:
        return image
    k = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.blur(image, (k, k))


def apply_min_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply non-linear min order-statistic filter (morphological erosion)."""
    if image is None or image.size == 0:
        return image
    k = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(image, kernel)


def apply_max_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply non-linear max order-statistic filter (morphological dilation)."""
    if image is None or image.size == 0:
        return image
    k = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(image, kernel)


def apply_midpoint_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply midpoint order-statistic filter: Midpoint = (Min + Max) / 2."""
    if image is None or image.size == 0:
        return image
    min_img = apply_min_filter(image, kernel_size).astype(np.float32)
    max_img = apply_max_filter(image, kernel_size).astype(np.float32)
    midpoint = (min_img + max_img) / 2.0
    return np.clip(midpoint, 0, 255).astype(np.uint8)


def apply_alpha_trimmed_mean(image: np.ndarray, kernel_size: int = 3, alpha: float = 0.2) -> np.ndarray:
    """Apply alpha-trimmed mean filter for heavy-tailed noise attenuation.

    Trims alpha/2 fraction from highest and lowest neighborhood intensities.
    """
    if image is None or image.size == 0:
        return image

    gray = convert_to_grayscale(image) if image.ndim == 3 else image
    h, w = gray.shape
    pad = kernel_size // 2
    padded = np.pad(gray, pad, mode="reflect")
    output = np.zeros_like(gray, dtype=np.uint8)

    k_area = kernel_size * kernel_size
    d = int(np.clip(alpha * k_area, 0, k_area - 1))
    trim_low = d // 2
    trim_high = k_area - (d - trim_low)

    # Neighborhood sliding window extraction
    for y in range(h):
        for x in range(w):
            window = padded[y : y + kernel_size, x : x + kernel_size].flatten()
            window.sort()
            trimmed = window[trim_low:trim_high]
            output[y, x] = int(np.mean(trimmed))

    return output


def apply_unsharp_mask(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    strength: float = 1.5,
) -> np.ndarray:
    """Apply unsharp masking and high-boost sharpening.

    Formula:
        I_sharp = I + strength * (I - I_gaussian)

    Args:
        image: Input 2D grayscale or 3D color uint8 array.
        kernel_size: Gaussian kernel size for low-pass mask.
        sigma: Standard deviation of Gaussian blur.
        strength: High-boost scaling factor k (k=1 is standard unsharp, k>1 is high-boost).
    """
    if image is None or image.size == 0:
        return image

    blurred = apply_gaussian_blur(image, kernel_size=kernel_size, sigma=sigma).astype(np.float32)
    orig = image.astype(np.float32)
    mask = orig - blurred
    sharpened = orig + strength * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Syllabus Unit II: Advanced Local Histogram Processing.
    Divides the image into contextual tiles, applies contrast clipping to limit
    noise amplification, and recombines tiles using bilinear interpolation.

    Args:
        image: Input grayscale (2D) or BGR (3D) uint8 image.
        clip_limit: Threshold for contrast limiting (default: 2.5).
        tile_grid_size: Number of tiles in row and column directions (default: 8x8).

    Returns:
        Enhanced uint8 image with balanced local contrast.
    """
    if image is None or image.size == 0:
        return image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        return clahe.apply(image)

    # For BGR, apply CLAHE to L channel in Lab color space for perceptual preservation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


class SpatialFilterSuite:
    """Class wrapper providing static methods for spatial and radiometric filtering."""

    to_gray = staticmethod(convert_to_grayscale)
    compute_histogram = staticmethod(compute_histogram)
    compute_histogram_statistics = staticmethod(compute_histogram_statistics)
    equalize_histogram = staticmethod(equalize_histogram)
    apply_clahe = staticmethod(apply_clahe)
    apply_gamma = staticmethod(apply_gamma_transform)
    apply_log = staticmethod(apply_log_transform)
    apply_negative = staticmethod(apply_image_negative)
    apply_contrast_stretching = staticmethod(apply_contrast_stretching)
    extract_bit_planes = staticmethod(extract_bit_planes)
    simulate_quantization = staticmethod(simulate_quantization)
    apply_gaussian = staticmethod(apply_gaussian_blur)
    apply_median = staticmethod(apply_median_filter)
    apply_bilateral = staticmethod(apply_bilateral_filter)
    apply_box = staticmethod(apply_box_filter)
    apply_min = staticmethod(apply_min_filter)
    apply_max = staticmethod(apply_max_filter)
    apply_midpoint = staticmethod(apply_midpoint_filter)
    apply_alpha_trimmed_mean = staticmethod(apply_alpha_trimmed_mean)
    apply_unsharp_mask = staticmethod(apply_unsharp_mask)
