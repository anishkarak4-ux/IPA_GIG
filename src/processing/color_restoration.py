"""Color space processing, image restoration, noise modeling, and compression metrics.

Academic Overview (Syllabus Units IV & V):
    - Color Spaces:
      - RGB / BGR: Additive primary trichromatic space.
      - HSV: Perceptual representation separating chromaticity (Hue, Saturation) from luminance (Value).
      - YCbCr: Luminance (Y) and color difference (Cb, Cr) standard for JPEG and video broadcasting.
      - CIE L*a*b*: Device-independent perceptual uniformity separating lightness (L*) from color-opponent dimensions (a*, b*).
    - Color Image Enhancement:
      - Equalizing the V-channel in HSV or L*-channel in Lab improves contrast without chromatic aberration.
    - Pseudo-Coloring:
      - Maps scalar intensity values to false-color spectrums (JET, HOT, VIRIDIS) for thermal and radiometric inspection.
    - Image Degradation & Restoration:
      - Degradation Model: g(x, y) = f(x, y) * h(x, y) + eta(x, y).
      - Noise Models: Additive Gaussian noise N(0, sigma^2) and Salt-and-Pepper impulse noise.
      - Motion Blur Restoration: Wiener deconvolution using PSF synthesized from tracking velocity vector v = (vx, vy).
    - Compression & Information Theory:
      - Shannon Entropy: H = -sum(p_i * log2(p_i)) bits/pixel.
      - Run-Length Encoding (RLE) compression ratio.
"""

from typing import Any, Literal
import cv2
import numpy as np


def decompose_color_spaces(bgr_image: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Decompose BGR color image into HSV, YCbCr, and CIE L*a*b* individual channel planes.

    Args:
        bgr_image: 3-channel uint8 BGR image array.

    Returns:
        Dictionary mapping color space names ('HSV', 'YCbCr', 'LAB') to channel arrays.
    """
    if bgr_image is None or bgr_image.size == 0 or bgr_image.ndim != 3:
        return {}

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

    return {
        "HSV": {"H": hsv[:, :, 0], "S": hsv[:, :, 1], "V": hsv[:, :, 2], "merged": hsv},
        "YCbCr": {"Y": ycbcr[:, :, 0], "Cr": ycbcr[:, :, 1], "Cb": ycbcr[:, :, 2], "merged": ycbcr},
        "LAB": {"L": lab[:, :, 0], "A": lab[:, :, 1], "B": lab[:, :, 2], "merged": lab},
    }


def equalize_color_image(
    bgr_image: np.ndarray,
    method: Literal["hsv", "lab", "naive_bgr"] = "hsv",
) -> np.ndarray:
    """Enhance color image contrast via channel-specific histogram equalization.

    Args:
        bgr_image: 3-channel uint8 BGR image.
        method: 'hsv' (equalizes V channel), 'lab' (equalizes L* channel), or 'naive_bgr' (all channels independently).

    Returns:
        Equalized 3-channel uint8 BGR image.
    """
    if bgr_image is None or bgr_image.size == 0 or bgr_image.ndim != 3:
        return bgr_image

    if method == "naive_bgr":
        b, g, r = cv2.split(bgr_image)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        return cv2.merge([b_eq, g_eq, r_eq])

    if method == "lab":
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    # Default 'hsv': equalizes intensity Value without changing color Hue or Saturation
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v_eq]), cv2.COLOR_HSV2BGR)


def apply_pseudocolor(gray_image: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Apply pseudo-color false thermal mapping to a grayscale image or edge map."""
    if gray_image is None or gray_image.size == 0:
        return gray_image
    gray_8u = gray_image if gray_image.dtype == np.uint8 else cv2.convertScaleAbs(gray_image)
    if gray_8u.ndim == 3:
        gray_8u = cv2.cvtColor(gray_8u, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray_8u, colormap)


def inject_noise(
    image: np.ndarray,
    noise_type: Literal["gaussian", "salt_and_pepper"] = "gaussian",
    sigma: float = 25.0,
    amount: float = 0.05,
) -> np.ndarray:
    """Inject synthetic mathematical noise to evaluate tracker and filter robustness.

    Args:
        image: Input uint8 array.
        noise_type: 'gaussian' or 'salt_and_pepper'.
        sigma: Standard deviation of Gaussian noise.
        amount: Proportion of corrupted pixels for impulse noise.

    Returns:
        Corrupted uint8 image array.
    """
    if image is None or image.size == 0:
        return image

    noisy = image.astype(np.float32)

    if noise_type == "gaussian":
        gauss = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
        noisy = np.clip(noisy + gauss, 0, 255).astype(np.uint8)
        return noisy

    # Salt and pepper
    out = image.copy()
    num_salt = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    out[tuple(coords)] = 255

    num_pepper = int(amount * image.size * 0.5)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    out[tuple(coords)] = 0
    return out


def compute_psnr_mse(clean: np.ndarray, noisy: np.ndarray) -> tuple[float, float]:
    """Calculate Peak Signal-to-Noise Ratio (PSNR in dB) and Mean Squared Error (MSE).

    Formula:
        MSE = (1 / MN) * sum((clean - noisy)^2)
        PSNR = 10 * log10(255^2 / MSE)
    """
    if clean is None or noisy is None or clean.size == 0 or noisy.size == 0:
        return 0.0, 0.0

    c_f = clean.astype(np.float64)
    n_f = noisy.astype(np.float64)
    mse = float(np.mean((c_f - n_f) ** 2))
    if mse < 1e-6:
        return 99.0, 0.0
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse))
    return round(psnr, 2), round(mse, 2)


def add_rayleigh_noise(image: np.ndarray, scale: float = 25.0) -> np.ndarray:
    """Inject Rayleigh distributed asymmetric sensor/radar noise (Syllabus Unit IV).

    Probability density function:
        p(z) = (z / sigma^2) * exp(-z^2 / (2 * sigma^2)) for z >= 0.
    """
    if image is None or image.size == 0:
        return image
    ray = np.random.rayleigh(scale=scale, size=image.shape).astype(np.float32)
    # Center noise around zero so it doesn't just add a bright wash
    ray_centered = ray - np.mean(ray)
    noisy = np.clip(image.astype(np.float32) + ray_centered, 0, 255).astype(np.uint8)
    return noisy


def add_uniform_noise(image: np.ndarray, low: float = -30.0, high: float = 30.0) -> np.ndarray:
    """Inject zero-mean uniform quantization noise (Syllabus Unit IV).

    Probability density function:
        p(z) = 1 / (b - a) for a <= z <= b.
    """
    if image is None or image.size == 0:
        return image
    uni = np.random.uniform(low, high, size=image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + uni, 0, 255).astype(np.uint8)
    return noisy


def add_periodic_noise(
    image: np.ndarray,
    amplitude: float = 30.0,
    frequency_x: float = 8.0,
    frequency_y: float = 4.0,
) -> np.ndarray:
    """Inject 2D sinusoidal periodic stripe interference noise (Syllabus Unit IV).

    Formula:
        eta(x, y) = A * sin(2*pi * fx * x / W + 2*pi * fy * y / H)
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    y_coords, x_coords = np.ogrid[:h, :w]
    sinusoid = amplitude * np.sin(
        2.0 * np.pi * frequency_x * (x_coords / float(w)) +
        2.0 * np.pi * frequency_y * (y_coords / float(h))
    ).astype(np.float32)

    if image.ndim == 3:
        sinusoid = sinusoid[:, :, np.newaxis]

    noisy = np.clip(image.astype(np.float32) + sinusoid, 0, 255).astype(np.uint8)
    return noisy


def generate_all_noise_models(image: np.ndarray) -> dict[str, dict[str, Any]]:
    """Synthesize textbook Unit IV noise distributions and measure resulting degradation.

    Returns:
        Dictionary mapping model keys ('clean', 'gaussian', 'salt_pepper', 'rayleigh', 'uniform', 'periodic')
        to dictionary holding image array, PSNR (dB), MSE, and mathematical descriptor.
    """
    if image is None or image.size == 0:
        return {}

    clean = image.copy()
    psnr_clean, mse_clean = 99.0, 0.0

    # 1. Gaussian Noise
    gauss = inject_noise(clean, noise_type="gaussian", sigma=25.0)
    psnr_g, mse_g = compute_psnr_mse(clean, gauss)

    # 2. Salt and Pepper Noise
    sp = inject_noise(clean, noise_type="salt_and_pepper", amount=0.08)
    psnr_sp, mse_sp = compute_psnr_mse(clean, sp)

    # 3. Rayleigh Noise
    ray = add_rayleigh_noise(clean, scale=28.0)
    psnr_ray, mse_ray = compute_psnr_mse(clean, ray)

    # 4. Uniform Noise
    uni = add_uniform_noise(clean, low=-35.0, high=35.0)
    psnr_uni, mse_uni = compute_psnr_mse(clean, uni)

    # 5. Periodic Sinusoidal Noise
    peri = add_periodic_noise(clean, amplitude=32.0, frequency_x=10.0, frequency_y=5.0)
    psnr_peri, mse_peri = compute_psnr_mse(clean, peri)

    return {
        "clean": {
            "name": "1. Clean Reference ROI",
            "image": clean,
            "psnr": psnr_clean,
            "psnr_db": psnr_clean,
            "mse": mse_clean,
            "desc": "Original f(x, y)",
            "math": "Original f(x, y)",
        },
        "gaussian": {
            "name": "2. Gaussian Noise",
            "image": gauss,
            "psnr": psnr_g,
            "psnr_db": psnr_g,
            "mse": mse_g,
            "desc": "N(mu=0, sigma=25)",
            "math": "N(mu=0, sigma=25)",
        },
        "salt_pepper": {
            "name": "3. Salt & Pepper (Impulse)",
            "image": sp,
            "psnr": psnr_sp,
            "psnr_db": psnr_sp,
            "mse": mse_sp,
            "desc": "p = 0.08 (Salt/Pep)",
            "math": "p = 0.08 (Salt/Pep)",
        },
        "rayleigh": {
            "name": "4. Rayleigh Noise",
            "image": ray,
            "psnr": psnr_ray,
            "psnr_db": psnr_ray,
            "mse": mse_ray,
            "desc": "Asymmetric radar (b=28)",
            "math": "Asymmetric radar (b=28)",
        },
        "uniform": {
            "name": "5. Uniform Noise",
            "image": uni,
            "psnr": psnr_uni,
            "psnr_db": psnr_uni,
            "mse": mse_uni,
            "desc": "Quantization U(-35, 35)",
            "math": "Quantization U(-35, 35)",
        },
        "periodic": {
            "name": "6. Periodic Sinusoidal",
            "image": peri,
            "psnr": psnr_peri,
            "psnr_db": psnr_peri,
            "mse": mse_peri,
            "desc": "A*sin(2*pi*(fx*x + fy*y))",
            "math": "A*sin(2*pi*(fx*x + fy*y))",
        },
    }


def make_motion_psf(length: int = 15, angle_deg: float = 0.0) -> np.ndarray:
    """Construct linear motion blur Point Spread Function (PSF) kernel.

    Args:
        length: Motion blur path length in pixels.
        angle_deg: Motion trajectory orientation angle in degrees.

    Returns:
        2D normalized float64 convolution kernel.
    """
    l = max(1, int(length))
    kernel = np.zeros((l, l), dtype=np.float64)
    kernel[l // 2, :] = 1.0

    # Rotate kernel to align with velocity vector angle
    center = (l / 2.0 - 0.5, l / 2.0 - 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(kernel, rot_mat, (l, l), flags=cv2.INTER_LINEAR)
    sum_val = np.sum(rotated)
    if sum_val > 0:
        rotated /= sum_val
    else:
        rotated[l // 2, l // 2] = 1.0
    return rotated


def restore_motion_blur(
    image: np.ndarray,
    length: int = 11,
    angle_deg: float = 0.0,
    k_regularizer: float = 0.01,
) -> np.ndarray:
    """Restore motion-blurred image using Wiener frequency-domain deconvolution.

    Formula:
        F_est(u, v) = [H*(u, v) / (|H(u, v)|^2 + K)] * G(u, v)

    Args:
        image: Input 2D grayscale or 3D color uint8 array.
        length: Estimated motion blur displacement path length.
        angle_deg: Estimated blur direction angle.
        k_regularizer: Noise-to-signal power ratio regularization parameter K.

    Returns:
        Restored deblurred uint8 array.
    """
    if image is None or image.size == 0:
        return image

    is_color = (image.ndim == 3)
    channels = cv2.split(image) if is_color else [image]
    restored_channels: list[np.ndarray] = []

    h, w = image.shape[:2]
    psf = make_motion_psf(length=length, angle_deg=angle_deg)
    # Pad PSF to full image size
    psf_padded = np.zeros((h, w), dtype=np.float64)
    ph, pw = psf.shape
    psf_padded[:ph, :pw] = psf
    psf_padded = np.roll(psf_padded, -ph // 2, axis=0)
    psf_padded = np.roll(psf_padded, -pw // 2, axis=1)

    h_dft = np.fft.fft2(psf_padded)
    h_conj = np.conj(h_dft)
    h_mag_sq = np.abs(h_dft) ** 2
    wiener_filter = h_conj / (h_mag_sq + max(1e-5, k_regularizer))

    for ch in channels:
        ch_dft = np.fft.fft2(ch.astype(np.float64))
        deconv_dft = ch_dft * wiener_filter
        restored = np.real(np.fft.ifft2(deconv_dft))
        restored_channels.append(np.clip(restored, 0, 255).astype(np.uint8))

    if is_color:
        return cv2.merge(restored_channels)
    return restored_channels[0]


def compute_shannon_entropy(gray_image: np.ndarray) -> float:
    """Calculate Shannon information entropy: H = -sum(p_i * log2(p_i)) bits/pixel."""
    if gray_image is None or gray_image.size == 0:
        return 0.0

    counts = np.bincount(gray_image.astype(np.uint8).ravel(), minlength=256)
    probs = counts / float(gray_image.size)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_rle_compression_ratio(binary_mask: np.ndarray) -> tuple[float, int]:
    """Compute Run-Length Encoding (RLE) compressed size and compression ratio.

    Returns:
        Tuple of (compression_ratio, rle_run_count).
    """
    if binary_mask is None or binary_mask.size == 0:
        return 1.0, 0

    flat = (binary_mask.ravel() > 0).astype(np.uint8)
    runs = np.diff(np.concatenate(([flat[0] ^ 1], flat, [flat[-1] ^ 1]))).nonzero()[0]
    run_lengths = np.diff(runs)
    run_count = len(run_lengths)

    # Each run typically requires 2 bytes (count + value), compared to original 1 byte/pixel
    compressed_bytes = max(1, run_count * 2)
    original_bytes = flat.size
    ratio = float(original_bytes) / float(compressed_bytes)

    return ratio, run_count
