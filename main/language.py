from deep_translator import GoogleTranslator
import speech_recognition as sr
import pyttsx3


r = sr.Recognizer()
voiceID1="HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_trTR_Tolga"#türkçe dil
voiceID2="HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"#ingilizce dil


def speaker(sample, voice_id):

    engine = pyttsx3.init()
    engine.setProperty('voice', voice_id)
    engine.setProperty("rate", 160)
    engine.say(sample)
    engine.runAndWait()


def speak_main(prediction):
    while True:
        try:
            with sr.Microphone() as source:

                print("Bir dil seçin / Choose a language")
                speaker("bir dil secin", voiceID1)
                speaker("choose a language", voiceID2)

                r.adjust_for_ambient_noise(source, duration=0.1)
                audio = r.listen(source)

                text = r.recognize_google(audio, language='tr')
                text = text.lower()

                print("Did you say: " + text)

                if text == 'i̇ngilizce' or text == 'english':
                    print("İngilizce seçildi.")
                    for i in prediction:
                        speaker(i, voiceID2)
                        print(i)

                elif text == 'türkçe' or text == "turkish":
                    print("Türkçe seçildi.")
                    for j in prediction:
                        result = GoogleTranslator(source='en', target='tr').translate(j)
                        speaker(result, voiceID1)
                        print(result)

                elif text == 'bitir' or text == 'finish':
                    break

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            error1 = "Hatalı giriş, lütfen tekrar deneyin."
            error2 = "Wrong language, please try again."
            print(error1 + " " + error2)
            speaker(error1, voiceID1)
            speaker(error2, voiceID2)
