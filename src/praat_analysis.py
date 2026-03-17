# import math
# import parselmouth
# from parselmouth.praat import call

# from dotenv import load_dotenv
# load_dotenv()


# def _safe_formant_value(formant, formant_number: int, time_sec: float):
#     try:
#         v = call(formant, "Get value at time", formant_number, time_sec, "Hertz", "Linear")
#         if v is None:
#             return None
#         if isinstance(v, float) and math.isnan(v):
#             return None
#         return v
#     except Exception:
#         return None


# def _build_formant_object(snd: parselmouth.Sound):
#     return call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)


# def analyze_at_time(wav: str, time_sec: float):
#     snd = parselmouth.Sound(wav)
#     formant = _build_formant_object(snd)

#     # 範囲外を防ぐ
#     time_sec = max(0.0, min(time_sec, snd.duration))

#     return {
#         "f1": _safe_formant_value(formant, 1, time_sec),
#         "f2": _safe_formant_value(formant, 2, time_sec),
#         "f3": _safe_formant_value(formant, 3, time_sec),
#     }


# def analyze_segment(wav: str, start_sec: float, duration_sec: float):
#     snd = parselmouth.Sound(wav)
#     formant = _build_formant_object(snd)

#     if duration_sec is None or duration_sec <= 0:
#         center = start_sec
#     else:
#         center = start_sec + duration_sec / 2.0

#     center = max(0.0, min(center, snd.duration))

#     return {
#         "f1": _safe_formant_value(formant, 1, center),
#         "f2": _safe_formant_value(formant, 2, center),
#         "f3": _safe_formant_value(formant, 3, center),
#     }


# def analyze(wav: str):
#     snd = parselmouth.Sound(wav)
#     center = snd.duration / 2.0
#     formant = _build_formant_object(snd)

#     return {
#         "f1": _safe_formant_value(formant, 1, center),
#         "f2": _safe_formant_value(formant, 2, center),
#         "f3": _safe_formant_value(formant, 3, center),
#     }
import math
import parselmouth
from parselmouth.praat import call

from dotenv import load_dotenv
load_dotenv()


DEFAULT_NUM_SAMPLES = 5


def _safe_formant_value(formant, formant_number: int, time_sec: float):
    try:
        v = call(formant, "Get value at time", formant_number, time_sec, "Hertz", "Linear")
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    except Exception:
        return None


def _build_formant_object(snd: parselmouth.Sound):
    return call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)


def _clamp_time(time_sec: float, duration_sec: float) -> float:
    if time_sec is None:
        return 0.0
    return max(0.0, min(float(time_sec), float(duration_sec)))


def _mean(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _min(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return min(valid)


def _max(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return max(valid)


def _sample_times_in_segment(start_sec: float, duration_sec: float, total_duration: float, num_samples: int):
    start_sec = _clamp_time(start_sec, total_duration)

    if duration_sec is None or duration_sec <= 0:
        return [start_sec]

    end_sec = _clamp_time(start_sec + duration_sec, total_duration)

    if end_sec <= start_sec:
        return [start_sec]

    if num_samples <= 1:
        return [(start_sec + end_sec) / 2.0]

    step = (end_sec - start_sec) / (num_samples - 1)
    return [start_sec + i * step for i in range(num_samples)]


def _extract_formants_at_times(formant, times):
    samples = []
    for t in times:
        f1 = _safe_formant_value(formant, 1, t)
        f2 = _safe_formant_value(formant, 2, t)
        f3 = _safe_formant_value(formant, 3, t)

        gap_f3_f2 = None
        if f2 is not None and f3 is not None:
            gap_f3_f2 = f3 - f2

        samples.append({
            "time": t,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f3_f2_gap": gap_f3_f2,
        })
    return samples


def _summarize_samples(samples):
    f1s = [s["f1"] for s in samples]
    f2s = [s["f2"] for s in samples]
    f3s = [s["f3"] for s in samples]
    gaps = [s["f3_f2_gap"] for s in samples]

    mid_idx = len(samples) // 2
    center_sample = samples[mid_idx] if samples else {"f1": None, "f2": None, "f3": None}

    return {
        # 後方互換用: 代表値として中央サンプルを残す
        "f1": center_sample.get("f1"),
        "f2": center_sample.get("f2"),
        "f3": center_sample.get("f3"),

        # 要約値
        "f1_mean": _mean(f1s),
        "f2_mean": _mean(f2s),
        "f3_mean": _mean(f3s),

        "f1_min": _min(f1s),
        "f2_min": _min(f2s),
        "f3_min": _min(f3s),

        "f1_max": _max(f1s),
        "f2_max": _max(f2s),
        "f3_max": _max(f3s),

        "f3_f2_gap_mean": _mean(gaps),
        "f3_f2_gap_min": _min(gaps),
        "f3_f2_gap_max": _max(gaps),
    }


def analyze_at_time(wav: str, time_sec: float):
    snd = parselmouth.Sound(wav)
    formant = _build_formant_object(snd)

    time_sec = _clamp_time(time_sec, snd.duration)

    return {
        "f1": _safe_formant_value(formant, 1, time_sec),
        "f2": _safe_formant_value(formant, 2, time_sec),
        "f3": _safe_formant_value(formant, 3, time_sec),
    }


def analyze_segment(wav: str, start_sec: float, duration_sec: float, num_samples: int = DEFAULT_NUM_SAMPLES):
    snd = parselmouth.Sound(wav)
    formant = _build_formant_object(snd)

    times = _sample_times_in_segment(
        start_sec=start_sec,
        duration_sec=duration_sec,
        total_duration=snd.duration,
        num_samples=num_samples,
    )
    samples = _extract_formants_at_times(formant, times)
    summary = _summarize_samples(samples)

    return {
        "start_sec": _clamp_time(start_sec, snd.duration),
        "duration_sec": duration_sec,
        "num_samples": len(samples),
        "samples": samples,
        **summary,
    }


def analyze(wav: str):
    snd = parselmouth.Sound(wav)
    center = snd.duration / 2.0
    formant = _build_formant_object(snd)

    return {
        "f1": _safe_formant_value(formant, 1, center),
        "f2": _safe_formant_value(formant, 2, center),
        "f3": _safe_formant_value(formant, 3, center),
    }


def analyze_formant_track(
    wav: str,
    time_step_sec: float = 0.01,
    start_sec: float | None = None,
    end_sec: float | None = None,
):
    snd = parselmouth.Sound(wav)
    formant = _build_formant_object(snd)

    start = 0.0 if start_sec is None else _clamp_time(start_sec, snd.duration)
    end = snd.duration if end_sec is None else _clamp_time(end_sec, snd.duration)

    if end < start:
        start, end = end, start

    if time_step_sec <= 0:
        time_step_sec = 0.01

    times = []
    t = start
    while t <= end:
        times.append(t)
        t += time_step_sec

    if not times or times[-1] < end:
        times.append(end)

    samples = _extract_formants_at_times(formant, times)

    return {
        "start_sec": start,
        "end_sec": end,
        "time_step_sec": time_step_sec,
        "samples": samples,
    }


def analyze_segment_track(
    wav: str,
    start_sec: float,
    duration_sec: float,
    time_step_sec: float = 0.005,
):
    if duration_sec is None or duration_sec <= 0:
        return {
            "start_sec": start_sec,
            "end_sec": start_sec,
            "time_step_sec": time_step_sec,
            "samples": [],
        }

    end_sec = start_sec + duration_sec
    return analyze_formant_track(
        wav=wav,
        time_step_sec=time_step_sec,
        start_sec=start_sec,
        end_sec=end_sec,
    )