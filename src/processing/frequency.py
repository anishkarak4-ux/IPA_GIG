"""Frequency domain processing and Fourier analysis suite.

Academic Overview (Syllabus Unit III):
    - 2D Discrete Fourier Transform (2D DFT):
      F(u, v) = sum_{x=0}^{M-1} sum_{y=0}^{N-1} f(x, y) * exp(-j * 2*pi * (u*x/M + v*y/N))
    - Spectrum Centering (fftshift):
      Relocates the zero-frequency (DC) component to the center of spectrum (M/2, N/2).
    - Log-Magnitude Spectrum:
      S(u, v) = ln(1 + |F_shifted(u, v)|), scaled to [0, 255] uint8 for display.
    - Frequency Domain Filtering:
      G(u, v) = H(u, v) * F(u, v)
      1. Ideal Low-Pass / High-Pass Filter (ILPF / IHPF): Sharp cutoff radius D0.
         Exhibits prominent Gibbs ringing phenomenon in spatial domain.
      2. Butterworth Filter (BLPF / BHPF): Smooth transition with order n (default n=2).
      3. Gaussian Filter (GLPF / GHPF): Completely suppresses ringing artifacts.
    - High-Frequency Emphasis & Homomorphic Filtering:
      Separates multiplicative illumination (low frequency) and reflectance (high frequency).
"""

from typing import Literal
import cv2
import numpy as np


def compute_fft2_spectrum(gray_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D Discrete Fourier Transform and centered log-magnitude spectrum.

    Args:
        gray_image: 2D uint8 grayscale image.

    Returns:
        Tuple of (dft_shifted_complex, log_magnitude_uint8):
            - dft_shifted_complex: 2D complex128 array centered with DC at (M/2, N/2).
            - log_magnitude_uint8: 2D uint8 image scaled across [0, 255] for visual inspection.
    """
    if gray_image is None or gray_image.size == 0:
        empty = np.zeros((0, 0), dtype=np.uint8)
        return np.zeros((0, 0), dtype=np.complex128), empty

    img_float = gray_image.astype(np.float64)
    dft = np.fft.fft2(img_float)
    dft_shifted = np.fft.fftshift(dft)

    magnitude = np.abs(dft_shifted)
    log_mag = np.log1p(magnitude)

    max_val = float(np.max(log_mag))
    if max_val > 1e-6:
        log_mag_uint8 = (log_mag / max_val * 255.0).clip(0, 255).astype(np.uint8)
    else:
        log_mag_uint8 = np.zeros_like(gray_image, dtype=np.uint8)

    return dft_shifted, log_mag_uint8


def make_frequency_filter_mask(
    shape: tuple[int, int],
    cutoff_radius: float = 30.0,
    filter_type: Literal["ideal", "butterworth", "gaussian"] = "gaussian",
    pass_type: Literal["low", "high"] = "low",
    order: int = 2,
) -> np.ndarray:
    """Generate 2D frequency domain transfer function H(u, v) centered at (M/2, N/2).

    Args:
        shape: Dimensions (height, width) of target spectrum.
        cutoff_radius: Distance cutoff D0 in frequency coordinates.
        filter_type: 'ideal', 'butterworth', or 'gaussian'.
        pass_type: 'low' for LPF, 'high' for HPF.
        order: Exponent order n for Butterworth filter.

    Returns:
        2D float64 array of shape (height, width) with values in [0, 1].
    """
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    d0 = max(1.0, float(cutoff_radius))

    if filter_type == "ideal":
        mask = (d <= d0).astype(np.float64)
    elif filter_type == "butterworth":
        n = max(1, order)
        mask = 1.0 / (1.0 + (d / d0) ** (2 * n))
    elif filter_type == "gaussian":
        mask = np.exp(-(d ** 2) / (2.0 * (d0 ** 2)))
    else:
        mask = np.ones((h, w), dtype=np.float64)

    if pass_type == "high":
        mask = 1.0 - mask

    return mask


def apply_frequency_filter(
    gray_image: np.ndarray,
    cutoff_radius: float = 30.0,
    filter_type: Literal["ideal", "butterworth", "gaussian"] = "gaussian",
    pass_type: Literal["low", "high"] = "low",
    order: int = 2,
) -> np.ndarray:
    """Filter image in the 2D Fourier frequency domain and reconstruct in spatial domain.

    Transforms image to frequency space via FFT, applies transfer function mask H(u, v),
    and reconstructs via inverse FFT.
    """
    if gray_image is None or gray_image.size == 0:
        return gray_image

    dft_shifted, _ = compute_fft2_spectrum(gray_image)
    mask = make_frequency_filter_mask(
        gray_image.shape,
        cutoff_radius=cutoff_radius,
        filter_type=filter_type,
        pass_type=pass_type,
        order=order,
    )

    filtered_dft_shifted = dft_shifted * mask
    filtered_dft = np.fft.ifftshift(filtered_dft_shifted)
    idft = np.fft.ifft2(filtered_dft)
    spatial_reconstructed = np.real(idft)

    return np.clip(spatial_reconstructed, 0, 255).astype(np.uint8)


def apply_homomorphic_filter(
    gray_image: np.ndarray,
    cutoff_radius: float = 30.0,
    gamma_low: float = 0.5,
    gamma_high: float = 1.5,
    c: float = 1.0,
) -> np.ndarray:
    """Apply homomorphic filter to normalize illumination and enhance reflectance details.

    Mathematical formulation:
        ln(f) -> FFT -> multiply by H_homo(u, v) -> IFFT -> exp(output)
        H_homo(u, v) = (gamma_H - gamma_L) * [1 - exp(-c * (D^2 / D0^2))] + gamma_L
    """
    if gray_image is None or gray_image.size == 0:
        return gray_image

    img_float = gray_image.astype(np.float64) + 1.0
    img_log = np.log(img_float)

    h, w = gray_image.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    d0_2 = max(1.0, float(cutoff_radius) ** 2)

    h_homo = (gamma_high - gamma_low) * (1.0 - np.exp(-c * (d2 / d0_2))) + gamma_low

    dft = np.fft.fft2(img_log)
    dft_shifted = np.fft.fftshift(dft)
    filtered = dft_shifted * h_homo
    reconstructed_log = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))

    reconstructed = np.exp(reconstructed_log) - 1.0
    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def compute_phase_spectrum(gray_image: np.ndarray) -> np.ndarray:
    """Compute the 2D Fourier Phase Spectrum angle(F(u, v)).

    Syllabus Unit III: Fourier Properties & Phase Representation.
    Extracts phase angles in [-pi, +pi] and normalizes to [0, 255] uint8.
    The phase spectrum contains essential edge location and structural information.
    """
    if gray_image is None or gray_image.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    dft = np.fft.fft2(gray_image.astype(np.float64))
    dft_shifted = np.fft.fftshift(dft)
    phase = np.angle(dft_shifted)  # [-pi, +pi]

    phase_uint8 = ((phase + np.pi) / (2.0 * np.pi) * 255.0).clip(0, 255).astype(np.uint8)
    return phase_uint8


# Standard ISO/IEC JPEG Luminance Quantization Table (Q50)
JPEG_LUMA_Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)


def compute_block_dct(
    gray_image: np.ndarray,
    block_size: int = 8,
    quality: int = 50,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute 2D 8x8 Block Discrete Cosine Transform (DCT) & JPEG Quantization simulation.

    Syllabus Unit V: Transform Coding, Energy Compaction, & JPEG Compression.

    Args:
        gray_image: 2D uint8 grayscale image.
        block_size: DCT block size (standard 8x8).
        quality: JPEG quality scale factor 1-100 (default: 50).

    Returns:
        Tuple of (reconstructed_image, metrics_dict) where metrics_dict contains:
            - 'energy_compaction': percentage of AC energy in top-left low frequencies.
            - 'zero_coefficient_pct': percentage of zeroed quantized coefficients.
            - 'psnr_db': Peak Signal-to-Noise Ratio of reconstructed image.
            - 'mse': Mean Squared Error.
    """
    if gray_image is None or gray_image.size == 0:
        return gray_image, {"energy_compaction": 0.0, "zero_coefficient_pct": 0.0, "psnr_db": 0.0, "mse": 0.0}

    h, w = gray_image.shape[:2]
    # Pad image to integer multiples of block_size
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    if pad_h > 0 or pad_w > 0:
        padded = cv2.copyMakeBorder(gray_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    else:
        padded = gray_image.copy()

    ph, pw = padded.shape
    reconstructed = np.zeros((ph, pw), dtype=np.float32)

    # Scale quantization matrix by quality factor
    q = max(1, min(100, quality))
    scale = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
    q_table = np.clip(np.round((JPEG_LUMA_Q50 * scale + 50.0) / 100.0), 1.0, 255.0).astype(np.float32)

    total_ac_energy = 0.0
    low_freq_energy = 0.0
    total_coeffs = 0
    zero_coeffs = 0

    img_float = padded.astype(np.float32) - 128.0

    for r in range(0, ph, block_size):
        for c in range(0, pw, block_size):
            blk = img_float[r : r + block_size, c : c + block_size]
            dct_blk = cv2.dct(blk)

            # Energy compaction (AC coefficients)
            ac_blk = dct_blk.copy()
            ac_blk[0, 0] = 0.0  # Zero out DC
            blk_energy = float(np.sum(ac_blk ** 2))
            total_ac_energy += blk_energy

            # Low frequency triangle (top-left 3x3 AC)
            low_ac = ac_blk[:3, :3]
            low_freq_energy += float(np.sum(low_ac ** 2))

            # Quantization
            quant_blk = np.round(dct_blk / q_table)
            zero_coeffs += int(np.sum(quant_blk == 0))
            total_coeffs += block_size * block_size

            # Dequantization & Inverse DCT
            dequant_blk = quant_blk * q_table
            idct_blk = cv2.idct(dequant_blk)
            reconstructed[r : r + block_size, c : c + block_size] = idct_blk

    # Crop padding back to original dimensions
    rec_cropped = np.clip(reconstructed[:h, :w] + 128.0, 0, 255).astype(np.uint8)

    # Compute PSNR & MSE
    orig_f = gray_image.astype(np.float64)
    rec_f = rec_cropped.astype(np.float64)
    mse = float(np.mean((orig_f - rec_f) ** 2))
    psnr = 99.0 if mse < 1e-6 else float(10.0 * np.log10((255.0 ** 2) / mse))

    energy_compaction = (low_freq_energy / total_ac_energy * 100.0) if total_ac_energy > 0 else 100.0
    zero_pct = (zero_coeffs / total_coeffs * 100.0) if total_coeffs > 0 else 0.0

    metrics = {
        "energy_compaction": round(energy_compaction, 2),
        "energy_compaction_pct": round(energy_compaction, 2),
        "zero_coefficient_pct": round(zero_pct, 2),
        "psnr_db": round(psnr, 2),
        "mse": round(mse, 2),
    }

    return rec_cropped, metrics
