from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, find_peaks, resample, sosfiltfilt, welch

import neurokit2 as nk

from core.schema import SensorSample

from .basic_feature_description import BaseFeatureDescriptionInput


class WESADFeatureDescriptionInput(BaseFeatureDescriptionInput):
    dataset_name = "WESAD"
    title = "Input feature description for WESAD paper-style features:"

    def extract_features(self, sample: SensorSample) -> dict[str, float | None]:
        return extract_wesad_paper_features(sample.signals)

    def format_features(self, features: dict[str, float | None]) -> str:
        return format_wesad_paper_features(features)


def extract_wesad_paper_features(signals: dict) -> dict[str, float | None]:
    fs_chest = 700
    fs_wrist_bvp = 64
    fs_wrist_acc = 32
    fs_wrist_eda = 4
    fs_wrist_temp = 4

    features: dict[str, float | None] = {}

    if _has_signal(signals, "chest_acc"):
        features.update(_add_prefix(extract_acc_features(_center_crop(signals["chest_acc"], fs_chest, 5.0), fs_chest), "chest"))
    if _has_signal(signals, "chest_ecg"):
        features.update(_add_prefix(extract_ecg_features(signals["chest_ecg"], fs_chest), "chest"))
    if _has_signal(signals, "chest_eda"):
        features.update(_add_prefix(extract_eda_features(signals["chest_eda"], fs_chest), "chest"))
    if _has_signal(signals, "chest_emg"):
        chest_emg = _arr(signals["chest_emg"])
        features.update(_add_prefix(extract_emg_features(_center_crop(chest_emg, fs_chest, 5.0), chest_emg, fs_chest), "chest"))
    if _has_signal(signals, "chest_resp"):
        features.update(_add_prefix(extract_resp_features(signals["chest_resp"], fs_chest), "chest"))
    if _has_signal(signals, "chest_temp"):
        features.update(_add_prefix(extract_temp_features(signals["chest_temp"], fs_chest), "chest"))

    if _has_signal(signals, "wrist_acc"):
        features.update(_add_prefix(extract_acc_features(_center_crop(signals["wrist_acc"], fs_wrist_acc, 5.0), fs_wrist_acc), "wrist"))
    if _has_signal(signals, "wrist_bvp"):
        features.update(_add_prefix(extract_bvp_features(signals["wrist_bvp"], fs_wrist_bvp), "wrist"))
    if _has_signal(signals, "wrist_eda"):
        features.update(_add_prefix(extract_eda_features(signals["wrist_eda"], fs_wrist_eda), "wrist"))
    if _has_signal(signals, "wrist_temp"):
        features.update(_add_prefix(extract_temp_features(signals["wrist_temp"], fs_wrist_temp), "wrist"))

    return {name: _finite(value) for name, value in features.items()}


def format_wesad_paper_features(features: dict[str, float | None]) -> str:
    sections = {
        "Chest ACC": "chest_acc_",
        "Chest ECG/HRV": "chest_hr",
        "Chest EDA/SCR": ("chest_eda_", "chest_scl_", "chest_scr_"),
        "Chest EMG": "chest_emg_",
        "Chest RESP": "chest_resp_",
        "Chest TEMP": "chest_temp_",
        "Wrist ACC": "wrist_acc_",
        "Wrist BVP/HRV": "wrist_hr",
        "Wrist EDA/SCR": ("wrist_eda_", "wrist_scl_", "wrist_scr_"),
        "Wrist TEMP": "wrist_temp_",
    }

    lines = ["Input feature description for WESAD paper-style features:"]
    if not features:
        return "\n".join([*lines, "- no WESAD signal features available"])

    emitted: set[str] = set()
    for section, prefixes in sections.items():
        names = [name for name in features if _has_prefix(name, prefixes)]
        if not names:
            continue
        lines.append("")
        lines.append(f"{section}:")
        lines.extend(_compact_feature_lines(features, names))
        emitted.update(names)

    remaining = [name for name in features if name not in emitted]
    if remaining:
        lines.append("")
        lines.append("Other:")
        lines.extend(_compact_feature_lines(features, remaining))
    return "\n".join(lines)


def extract_acc_features(acc_segment, fs: int) -> dict[str, float | None]:
    feats = _unavailable_features([
        "acc_x_mean", "acc_x_std", "acc_x_abs_int", "acc_x_peak_freq",
        "acc_y_mean", "acc_y_std", "acc_y_abs_int", "acc_y_peak_freq",
        "acc_z_mean", "acc_z_std", "acc_z_abs_int", "acc_z_peak_freq",
        "acc_3d_mean", "acc_3d_std", "acc_3d_abs_int",
    ])
    acc_segment = _arr(acc_segment)
    if acc_segment.ndim != 2 or acc_segment.shape[0] < 10 or acc_segment.shape[1] < 3:
        return feats

    nperseg_val = min(256, len(acc_segment))
    for idx, axis in enumerate(["x", "y", "z"]):
        sig = acc_segment[:, idx]
        feats[f"acc_{axis}_mean"] = float(np.mean(sig))
        feats[f"acc_{axis}_std"] = float(np.std(sig))
        feats[f"acc_{axis}_abs_int"] = _absolute_integral(sig, fs)
        f, pxx = welch(sig, fs=fs, nperseg=nperseg_val)
        feats[f"acc_{axis}_peak_freq"] = float(f[np.argmax(pxx)]) if len(f) else None

    acc_3d = np.linalg.norm(acc_segment[:, :3], axis=1)
    feats["acc_3d_mean"] = float(np.mean(acc_3d))
    feats["acc_3d_std"] = float(np.std(acc_3d))
    feats["acc_3d_abs_int"] = _absolute_integral(acc_3d, fs)
    return feats


def extract_ecg_features(ecg_segment, fs: int) -> dict[str, float | None]:
    feats = _empty_cardiac_features()
    ecg_segment = _arr(ecg_segment).flatten()
    if len(ecg_segment) < fs * 10:
        return feats
    try:
        cleaned = nk.ecg_clean(ecg_segment, sampling_rate=fs)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        feats.update(_compute_cardiac_features_from_peaks(info.get("ECG_R_Peaks", []), fs))
    except Exception:
        pass
    return feats


def extract_bvp_features(bvp_segment, fs: int) -> dict[str, float | None]:
    feats = _empty_cardiac_features()
    bvp_segment = _arr(bvp_segment).flatten()
    if len(bvp_segment) < fs * 10:
        return feats
    try:
        cleaned = nk.ppg_clean(bvp_segment, sampling_rate=fs)
        info = nk.ppg_findpeaks(cleaned, sampling_rate=fs)
        feats.update(_compute_cardiac_features_from_peaks(info.get("PPG_Peaks", []), fs))
    except Exception:
        pass
    return feats


def extract_eda_features(eda_seg, fs: int) -> dict[str, float | None]:
    feats = _unavailable_features([
        "eda_mean", "eda_std", "eda_min", "eda_max", "eda_slope", "eda_range",
        "scl_mean", "scl_std", "scr_std", "scl_time_corr",
        "scr_num", "scr_amp_sum", "scr_dur_sum", "scr_area",
    ])
    eda_seg = _arr(eda_seg).flatten()
    if len(eda_seg) < max(10, int(fs * 5)):
        return feats

    try:
        process_fs = fs
        process_seg = eda_seg
        if fs < 50:
            process_fs = 64
            process_seg = resample(eda_seg, int(len(eda_seg) * (process_fs / fs)))

        nyq = process_fs / 2.0
        cutoff = min(5.0, 0.9 * nyq)
        sos = butter(4, cutoff / nyq, btype="low", output="sos")
        eda_f = sosfiltfilt(sos, process_seg)
        decomp = nk.eda_phasic(eda_f, sampling_rate=process_fs, method="neurokit")
        scl = np.asarray(decomp["EDA_Tonic"].values, dtype=float)
        scr = np.asarray(decomp["EDA_Phasic"].values, dtype=float)

        feats["eda_mean"] = float(np.mean(eda_f))
        feats["eda_std"] = float(np.std(eda_f))
        feats["eda_min"] = float(np.min(eda_f))
        feats["eda_max"] = float(np.max(eda_f))
        feats["eda_slope"] = _slope_feature(eda_f, process_fs)
        feats["eda_range"] = _dynamic_range(eda_f)
        feats["scl_mean"] = float(np.mean(scl))
        feats["scl_std"] = float(np.std(scl))
        feats["scr_std"] = float(np.std(scr))
        feats["scl_time_corr"] = _safe_corr_with_time(scl)

        info = nk.eda_findpeaks(scr, sampling_rate=process_fs)
        peaks = np.asarray(info.get("SCR_Peaks", []), dtype=float)
        peaks = peaks[~np.isnan(peaks)].astype(int)
        peaks = peaks[(peaks >= 0) & (peaks < len(scr))]
        amps = np.asarray(info.get("SCR_Amplitude", []), dtype=float)
        amps = amps[~np.isnan(amps)] if len(amps) else np.array([])
        feats["scr_num"] = float(len(peaks))
        feats["scr_amp_sum"] = float(np.sum(amps)) if len(amps) else 0.0

        dur_sum = 0.0
        area_sum = 0.0
        for start, end in _estimate_scr_regions_from_peaks(scr, peaks):
            dur_sum += (end - start) / process_fs
            area_sum += np.trapezoid(np.maximum(scr[start:end], 0), dx=1.0 / process_fs)
        feats["scr_dur_sum"] = float(dur_sum)
        feats["scr_area"] = float(area_sum)
    except Exception:
        pass
    return feats


def extract_emg_features(emg_5s, emg_60s, fs: int) -> dict[str, float | None]:
    feats = _unavailable_features([
        "emg_mean", "emg_std", "emg_range", "emg_abs_int",
        "emg_median", "emg_p10", "emg_p90",
        "emg_freq_mean", "emg_freq_median", "emg_peak_freq",
        "emg_psd_band_1", "emg_psd_band_2", "emg_psd_band_3",
        "emg_psd_band_4", "emg_psd_band_5", "emg_psd_band_6",
        "emg_psd_band_7",
        "emg_peaks_num", "emg_peak_amp_mean", "emg_peak_amp_std",
        "emg_peak_amp_sum", "emg_peak_amp_sum_norm",
    ])
    emg_5s = _arr(emg_5s).flatten()
    emg_60s = _arr(emg_60s).flatten()
    if len(emg_5s) < max(10, fs) or len(emg_60s) < max(10, fs):
        return feats

    try:
        nyq = 0.5 * fs
        e5 = sosfiltfilt(butter(4, 1.0 / nyq, btype="high", output="sos"), emg_5s)
        feats["emg_mean"] = float(np.mean(e5))
        feats["emg_std"] = float(np.std(e5))
        feats["emg_range"] = _dynamic_range(e5)
        feats["emg_abs_int"] = _absolute_integral(e5, fs)
        feats["emg_median"] = float(np.median(e5))
        feats["emg_p10"] = float(np.percentile(e5, 10))
        feats["emg_p90"] = float(np.percentile(e5, 90))

        f, pxx = welch(e5, fs=fs, nperseg=min(1024, len(e5)))
        if len(f) and np.sum(pxx) > 0:
            feats["emg_freq_mean"] = float(np.sum(f * pxx) / np.sum(pxx))
            cdf = np.cumsum(pxx) / np.sum(pxx)
            feats["emg_freq_median"] = float(f[np.searchsorted(cdf, 0.5)])
            feats["emg_peak_freq"] = float(f[np.argmax(pxx)])

        band_edges = np.linspace(0, 350, 8)
        for idx in range(7):
            lo, hi = band_edges[idx], band_edges[idx + 1]
            mask = (f >= lo) & (f < hi)
            feats[f"emg_psd_band_{idx + 1}"] = float(np.trapezoid(pxx[mask], f[mask])) if np.any(mask) else None

        e60 = sosfiltfilt(butter(4, 50.0 / nyq, btype="low", output="sos"), emg_60s)
        peaks, props = find_peaks(e60, height=np.mean(e60) + np.std(e60))
        amps = props.get("peak_heights", np.array([]))
        feats["emg_peaks_num"] = float(len(peaks))
        if len(amps):
            feats["emg_peak_amp_mean"] = float(np.mean(amps))
            feats["emg_peak_amp_std"] = float(np.std(amps))
            feats["emg_peak_amp_sum"] = float(np.sum(amps))
            feats["emg_peak_amp_sum_norm"] = float(np.sum(amps) / (np.sum(np.abs(e60)) + 1e-8))
    except Exception:
        pass
    return feats


def extract_resp_features(resp_seg, fs: int) -> dict[str, float | None]:
    feats = _unavailable_features([
        "resp_inhale_mean", "resp_inhale_std",
        "resp_exhale_mean", "resp_exhale_std",
        "resp_ie_ratio", "resp_range",
        "resp_insp_vol", "resp_rate", "resp_duration",
    ])
    resp_seg = _arr(resp_seg).flatten()
    if len(resp_seg) < max(10, int(fs * 10)):
        return feats

    try:
        valid = np.isfinite(resp_seg)
        if not np.any(valid):
            return feats
        if not np.all(valid):
            idx = np.arange(len(resp_seg))
            resp_seg[~valid] = np.interp(idx[~valid], idx[valid], resp_seg[valid])

        resp_seg = resp_seg - np.mean(resp_seg)
        nyq = 0.5 * fs
        rf = sosfiltfilt(butter(4, [0.1 / nyq, 0.35 / nyq], btype="band", output="sos"), resp_seg)
        win = max(5, int(fs * 0.2))
        win = win + 1 if win % 2 == 0 else win
        rf = np.convolve(rf, np.ones(win) / win, mode="same")

        min_dist = int(fs * 1.5)
        peaks, _ = find_peaks(rf, distance=min_dist)
        troughs, _ = find_peaks(-rf, distance=min_dist)
        if len(peaks) >= 2:
            feats["resp_rate"] = float(len(peaks) / (len(resp_seg) / fs / 60.0))
        feats["resp_range"] = float(np.ptp(rf))

        inhale, exhale, duration, insp_vol = [], [], [], []
        peaks = np.sort(np.asarray(peaks, dtype=int))
        troughs = np.sort(np.asarray(troughs, dtype=int))
        for idx in range(len(troughs) - 1):
            start, end = troughs[idx], troughs[idx + 1]
            mid_peaks = peaks[(peaks > start) & (peaks < end)]
            if len(mid_peaks) == 0:
                continue
            peak = mid_peaks[0]
            inh = (peak - start) / fs
            exh = (end - peak) / fs
            dur = (end - start) / fs
            if inh <= 0 or exh <= 0 or dur <= 0:
                continue
            inhale.append(inh)
            exhale.append(exh)
            duration.append(dur)
            insp_vol.append((rf[peak] - min(rf[start], rf[end])) * inh)

        if inhale:
            feats["resp_inhale_mean"] = float(np.mean(inhale))
            feats["resp_inhale_std"] = float(np.std(inhale))
            feats["resp_exhale_mean"] = float(np.mean(exhale))
            feats["resp_exhale_std"] = float(np.std(exhale))
            feats["resp_ie_ratio"] = _safe_div(float(np.mean(inhale)), float(np.mean(exhale)))
            feats["resp_insp_vol"] = float(np.mean(insp_vol))
            feats["resp_duration"] = float(np.mean(duration))
            feats["resp_rate"] = _safe_div(60.0, float(np.mean(duration)))
    except Exception:
        pass
    return feats


def extract_temp_features(temp_seg, fs: int) -> dict[str, float | None]:
    feats = _unavailable_features([
        "temp_mean", "temp_std", "temp_min", "temp_max", "temp_range", "temp_slope",
    ])
    temp_seg = _arr(temp_seg).flatten()
    if len(temp_seg) == 0:
        return feats
    feats["temp_mean"] = float(np.mean(temp_seg))
    feats["temp_std"] = float(np.std(temp_seg))
    feats["temp_min"] = float(np.min(temp_seg))
    feats["temp_max"] = float(np.max(temp_seg))
    feats["temp_range"] = _dynamic_range(temp_seg)
    feats["temp_slope"] = _slope_feature(temp_seg, fs)
    return feats


def _empty_cardiac_features() -> dict[str, float | None]:
    return _unavailable_features([
        "hr_mean", "hr_std", "hrv_mean", "hrv_std", "hrv_nn50", "hrv_pnn50",
        "hrv_tinn", "hrv_rmssd", "hrv_abs_ulf", "hrv_abs_lf", "hrv_abs_hf",
        "hrv_abs_uhf", "hrv_lf_hf_ratio", "hrv_total_power", "hrv_rel_ulf",
        "hrv_rel_lf", "hrv_rel_hf", "hrv_rel_uhf", "hrv_lf_norm", "hrv_hf_norm",
    ])


def _compute_cardiac_features_from_peaks(peaks, fs: int) -> dict[str, float | None]:
    feats = _empty_cardiac_features()
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 4:
        return feats
    rr = np.diff(peaks) / fs
    if len(rr) < 3:
        return feats

    hr = 60.0 / rr
    feats["hr_mean"] = float(np.mean(hr))
    feats["hr_std"] = float(np.std(hr))
    feats["hrv_mean"] = float(np.mean(rr))
    feats["hrv_std"] = float(np.std(rr))
    diff_rr_ms = np.abs(np.diff(rr)) * 1000.0
    nn50 = np.sum(diff_rr_ms > 50.0)
    feats["hrv_nn50"] = float(nn50)
    feats["hrv_pnn50"] = float(nn50 / len(diff_rr_ms)) if len(diff_rr_ms) else None
    feats["hrv_tinn"] = _compute_tinn(rr)
    feats["hrv_rmssd"] = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else None
    feats.update(_compute_hrv_spectral_features(rr))
    return feats


def _compute_hrv_spectral_features(intervals_sec) -> dict[str, float | None]:
    feats = _unavailable_features([
        "hrv_abs_ulf", "hrv_abs_lf", "hrv_abs_hf", "hrv_abs_uhf",
        "hrv_lf_hf_ratio", "hrv_total_power",
        "hrv_rel_ulf", "hrv_rel_lf", "hrv_rel_hf", "hrv_rel_uhf",
        "hrv_lf_norm", "hrv_hf_norm",
    ])
    intervals_sec = _arr(intervals_sec).flatten()
    if len(intervals_sec) < 4:
        return feats
    try:
        t = np.cumsum(intervals_sec)
        if len(t) < 3 or (t[-1] - t[0]) < 4.0:
            return feats
        interp_f = interp1d(t, intervals_sec, kind="linear", fill_value="extrapolate")
        t_uniform = np.arange(t[0], t[-1], 1 / 4.0)
        if len(t_uniform) < 16:
            return feats
        x = interp_f(t_uniform)
        f, pxx = welch(x, fs=4.0, nperseg=min(256, len(x)), nfft=1024)
        bands = {"ulf": (0.01, 0.04), "lf": (0.04, 0.15), "hf": (0.15, 0.40), "uhf": (0.40, 1.00)}
        powers = {}
        for name, (lo, hi) in bands.items():
            mask = (f >= lo) & (f < hi)
            powers[name] = float(np.trapezoid(pxx[mask], f[mask])) if np.any(mask) else None
            feats[f"hrv_abs_{name}"] = powers[name]
        total = sum(value for value in powers.values() if value is not None)
        feats["hrv_total_power"] = total
        feats["hrv_lf_hf_ratio"] = _safe_div(powers["lf"], powers["hf"])
        if total > 0:
            for name in bands:
                feats[f"hrv_rel_{name}"] = powers[name] / total if powers[name] is not None else None
        denom = None
        if powers["lf"] is not None and powers["hf"] is not None:
            denom = powers["lf"] + powers["hf"]
        if denom is not None and denom > 0:
            feats["hrv_lf_norm"] = powers["lf"] / denom
            feats["hrv_hf_norm"] = powers["hf"] / denom
    except Exception:
        pass
    return feats


def _compute_tinn(intervals_sec, bins: int = 64) -> float | None:
    intervals_sec = _arr(intervals_sec).flatten()
    if len(intervals_sec) < 5:
        return None
    try:
        hist, edges = np.histogram(intervals_sec * 1000.0, bins=bins)
        nonzero = np.where(hist > 0)[0]
        if len(nonzero) < 2:
            return None
        return float(edges[nonzero[-1] + 1] - edges[nonzero[0]])
    except Exception:
        return None


def _estimate_scr_regions_from_peaks(scr_sig, peaks) -> list[tuple[int, int]]:
    scr_sig = _arr(scr_sig).flatten()
    peaks = np.asarray(peaks, dtype=int)
    if len(scr_sig) == 0 or len(peaks) == 0:
        return []
    regions = []
    for peak in peaks:
        start = peak
        end = peak
        while start > 0 and scr_sig[start] > 0:
            start -= 1
        while end < len(scr_sig) and scr_sig[end] > 0:
            end += 1
        if end > start:
            regions.append((start, end))
    return regions


def _center_crop(sig, fs: int, seconds: float):
    arr = _arr(sig)
    target_len = int(fs * seconds)
    if len(arr) <= target_len:
        return arr
    start = max(0, (len(arr) - target_len) // 2)
    return arr[start : start + target_len]


def _compact_feature_lines(features: dict[str, float | None], names: list[str], per_line: int = 4) -> list[str]:
    lines = []
    for start in range(0, len(names), per_line):
        chunk = names[start : start + per_line]
        lines.append("- " + ", ".join(f"{name}={_format_feature_value(features[name])}" for name in chunk))
    return lines


def _has_prefix(name: str, prefixes) -> bool:
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    return any(name.startswith(prefix) for prefix in prefixes)


def _add_prefix(feature_dict: dict[str, float | None], prefix: str) -> dict[str, float | None]:
    return {f"{prefix}_{key}": value for key, value in feature_dict.items()}


def _has_signal(signals: dict, name: str) -> bool:
    if name not in signals:
        return False
    return len(_arr(signals[name])) > 0


def _arr(value) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float)


def _absolute_integral(x, fs: int) -> float | None:
    x = _arr(x).flatten()
    if len(x) == 0:
        return None
    return float(np.trapezoid(np.abs(x), dx=1.0 / fs))


def _slope_feature(x, fs: int) -> float | None:
    x = _arr(x).flatten()
    if len(x) < 2:
        return None
    t = np.arange(len(x)) / fs
    return float(np.polyfit(t, x, 1)[0])


def _dynamic_range(x) -> float | None:
    x = _arr(x).flatten()
    if len(x) == 0:
        return None
    return float(np.nanmax(x) - np.nanmin(x))


def _safe_corr_with_time(x) -> float | None:
    x = _arr(x).flatten()
    if len(x) < 2 or np.nanstd(x) == 0:
        return None
    return float(np.corrcoef(x, np.arange(len(x), dtype=float))[0, 1])


def _safe_div(a, b) -> float | None:
    if a is None or b is None or b in [0, 0.0]:
        return None
    return float(a) / float(b)


def _finite(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _unavailable_features(names: list[str]) -> dict[str, float | None]:
    return {name: None for name in names}


def _format_feature_value(value: float | None) -> str:
    if value is None:
        return "unavailable due to insufficient signal"
    return f"{float(value):.4g}"


__all__ = [
    "WESADFeatureDescriptionInput",
    "extract_wesad_paper_features",
    "format_wesad_paper_features",
]


def _demo() -> None:
    sample = SensorSample(
        dataset="WESAD",
        subject="mock",
        label=2,
        signals={
            "chest_acc": np.zeros((20, 3)),
            "chest_temp": np.linspace(32.0, 33.0, 20),
        },
    )
    print(WESADFeatureDescriptionInput().build_input(sample))


if __name__ == "__main__":
    _demo()
