from dotenv import load_dotenv
import os
import json
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

def evaluate(wav_path, reference_text):
    speech_key = os.environ.get("SPEECH_KEY")
    speech_region = os.environ.get("SPEECH_REGION")

    if not speech_key or not speech_region:
        raise ValueError("SPEECH_KEY または SPEECH_REGION が未設定です")

    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=speech_region
    )
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.AudioConfig(filename=wav_path)

    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        json_string=(
            '{"referenceText":"%s",'
            '"gradingSystem":"HundredMark",'
            '"granularity":"Phoneme",'
            '"phonemeAlphabet":"IPA",'
            '"nBestPhonemeCount":5}' % reference_text
        )
    )
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )
    pronunciation_config.apply_to(recognizer)

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        raw_json = result.properties.get(
            speechsdk.PropertyId.SpeechServiceResponse_JsonResult
        )
        return json.loads(raw_json)

    elif result.reason == speechsdk.ResultReason.NoMatch:
        raise RuntimeError("音声は取得できたが、認識結果がありませんでした")

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = speechsdk.CancellationDetails(result)

        # Python SDKでは error_code がない環境があるため、安全に取る
        reason = getattr(cancellation, "reason", None)
        error_details = getattr(cancellation, "error_details", None)

        # 参考までに、利用可能な属性を確認したいとき用
        available_attrs = [
            name for name in dir(cancellation)
            if not name.startswith("_")
        ]

        raise RuntimeError(
            "Azure Speech がキャンセルしました\n"
            f"reason: {reason}\n"
            f"error_details: {error_details}\n"
            f"available_attrs: {available_attrs}"
        )

    else:
        raise RuntimeError(f"想定外の結果です: {result.reason}")

def extract_phonemes(result_json):
    phoneme_rows = []

    nbest = result_json.get("NBest", [])
    if not nbest:
        return phoneme_rows

    words = nbest[0].get("Words", [])

    for word_index, word_data in enumerate(words):
        word_text = word_data.get("Word", "")

        # ここが重要: Phonemes は word_data の直下
        phonemes = word_data.get("Phonemes", [])

        for phoneme_index, phoneme in enumerate(phonemes):
            pa = phoneme.get("PronunciationAssessment", {})
            nbest_phonemes = pa.get("NBestPhonemes", [])
            misrecognition_candidates = [
                {
                    "phoneme": candidate.get("Phoneme"),
                    "score": candidate.get("Score"),
                }
                for candidate in nbest_phonemes
            ]

            phoneme_rows.append({
                "word_in_result": word_text,
                "word_index": word_index,
                "phoneme_index": phoneme_index,
                "phoneme": phoneme.get("Phoneme"),
                "offset": phoneme.get("Offset"),
                "duration": phoneme.get("Duration"),
                "accuracy": pa.get("AccuracyScore"),
                # 誤認候補も一緒に保存すると便利
                "misrecognition_candidates": misrecognition_candidates,
            })

    return phoneme_rows
