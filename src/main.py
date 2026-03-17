import json
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

from record import RecordingCancelled, record_audio_with_gui
from azure_eval import evaluate, extract_phonemes
from praat_analysis import analyze, analyze_segment

WORDS_FILE = "words.txt"
RECORD_DIR = "recordings"
RESULT_DIR = "results"

WORD_HISTORY_FILE = os.path.join(RESULT_DIR, "history.csv")
PHONEME_HISTORY_FILE = os.path.join(RESULT_DIR, "phoneme_history.csv")


def append_csv(df: pd.DataFrame, path: str):
    if df.empty:
        return

    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def ticks_to_seconds(x):
    if x is None:
        return None
    try:
        # Azure Pronunciation Assessment の offset / duration は 100ns 単位
        return float(x) / 10_000_000.0
    except Exception:
        return None


def to_json_str(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return None


def run():
    os.makedirs(RECORD_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(WORDS_FILE, encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]

    word_rows = []
    phoneme_rows = []

    for word in words:
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(RECORD_DIR, f"{word}_{session_id}.wav")

        try:
            record_audio_with_gui(filename, prompt_text=word)
        except RecordingCancelled:
            print("\nRecording session cancelled by user.")
            break

        az = evaluate(filename, word)
        pr = analyze(filename)

        nbest = az.get("NBest", [])
        if not nbest:
            print(f"[WARN] No NBest returned for word: {word}")
            continue

        nbest0 = nbest[0]
        pa = nbest0.get("PronunciationAssessment", {})

        # 単語単位の履歴
        word_rows.append({
            "session_id": session_id,
            "time": now.isoformat(),
            "word": word,
            "wav_path": filename,
            "recognized_text": nbest0.get("Display", ""),
            "pron": pa.get("PronScore"),
            "accuracy": pa.get("AccuracyScore"),
            "fluency": pa.get("FluencyScore"),
            "completeness": pa.get("CompletenessScore"),
            "f1": pr.get("f1"),
            "f2": pr.get("f2"),
            "f3": pr.get("f3"),
        })

        # 音素単位の履歴
        extracted = extract_phonemes(az)
        for row in extracted:
            offset_raw = row.get("offset")
            duration_raw = row.get("duration")

            offset_sec = ticks_to_seconds(offset_raw)
            duration_sec = ticks_to_seconds(duration_raw)

            ph_formant = analyze_segment(
                filename,
                start_sec=offset_sec if offset_sec is not None else 0.0,
                duration_sec=duration_sec if duration_sec is not None else 0.0,
            )

            phoneme_rows.append({
                "session_id": session_id,
                "time": now.isoformat(),
                "target_word": word,
                "wav_path": filename,
                "recognized_text": nbest0.get("Display", ""),
                "word_in_result": row.get("word_in_result"),
                "word_index": row.get("word_index"),
                "syllable_index": row.get("syllable_index"),
                "phoneme_index": row.get("phoneme_index"),
                "phoneme": row.get("phoneme"),
                "offset": offset_raw,
                "duration": duration_raw,
                "offset_sec": offset_sec,
                "duration_sec": duration_sec,
                "accuracy": row.get("accuracy"),

                # 後方互換用: 中央サンプル値
                "f1": ph_formant.get("f1"),
                "f2": ph_formant.get("f2"),
                "f3": ph_formant.get("f3"),

                # 5点サンプリング情報
                "num_samples": ph_formant.get("num_samples"),
                "formant_samples_json": to_json_str(ph_formant.get("samples", [])),

                # 要約値
                "f1_mean": ph_formant.get("f1_mean"),
                "f2_mean": ph_formant.get("f2_mean"),
                "f3_mean": ph_formant.get("f3_mean"),

                "f1_min": ph_formant.get("f1_min"),
                "f2_min": ph_formant.get("f2_min"),
                "f3_min": ph_formant.get("f3_min"),

                "f1_max": ph_formant.get("f1_max"),
                "f2_max": ph_formant.get("f2_max"),
                "f3_max": ph_formant.get("f3_max"),

                "f3_f2_gap_mean": ph_formant.get("f3_f2_gap_mean"),
                "f3_f2_gap_min": ph_formant.get("f3_f2_gap_min"),
                "f3_f2_gap_max": ph_formant.get("f3_f2_gap_max"),
            })

        print(
            f"[OK] {word} | "
            f"pron={pa.get('PronScore')} "
            f"accuracy={pa.get('AccuracyScore')}"
        )

    word_df = pd.DataFrame(word_rows)
    phoneme_df = pd.DataFrame(phoneme_rows)

    append_csv(word_df, WORD_HISTORY_FILE)
    append_csv(phoneme_df, PHONEME_HISTORY_FILE)

    print(f"\nSaved word history -> {WORD_HISTORY_FILE}")
    print(f"Saved phoneme history -> {PHONEME_HISTORY_FILE}")


if __name__ == "__main__":
    run()
