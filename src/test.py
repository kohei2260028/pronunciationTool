from dotenv import load_dotenv
import os
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

speech_config = speechsdk.SpeechConfig(
    subscription=os.environ["SPEECH_KEY"],
    region=os.environ["SPEECH_REGION"]
)

print("接続成功")
# from dotenv import load_dotenv
# import os
# import azure.cognitiveservices.speech as speechsdk

# load_dotenv()

# speech_config = speechsdk.SpeechConfig(
#     subscription=os.environ["SPEECH_KEY"],
#     region=os.environ["SPEECH_REGION"]
# )
# speech_config.speech_recognition_language = "en-US"

# audio_config = speechsdk.AudioConfig(filename="recordings/right_1773714873.wav")
# recognizer = speechsdk.SpeechRecognizer(
#     speech_config=speech_config,
#     audio_config=audio_config
# )

# result = recognizer.recognize_once()

# print("result.reason =", result.reason)

# if result.reason == speechsdk.ResultReason.RecognizedSpeech:
#     print("text =", result.text)
# elif result.reason == speechsdk.ResultReason.Canceled:
#     cancellation = speechsdk.CancellationDetails(result)
#     print("cancellation.reason =", cancellation.reason)
#     print("error_code =", cancellation.error_code)
#     print("error_details =", cancellation.error_details)
# elif result.reason == speechsdk.ResultReason.NoMatch:
#     print("NoMatch")