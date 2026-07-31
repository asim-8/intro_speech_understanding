import gtts
import speech_recognition as sr
import librosa
import soundfile as sf

def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech, then write it to filename.
    
    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it
    filename (str) - the filename in which it should be saved
    '''
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)


def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files should be created as MP3, then converted to WAV, then recognized.

    @param:
    texts - a list of the texts you want to synthesize
    languages - a list of their languages
    filenames - a list of their root filenames, without the ".mp3" ending

    @return:
    recognized_texts - list of the strings that were recognized from each file
    '''
    recognizer = sr.Recognizer()
    recognized_texts = []

    for text, lang, root_fn in zip(texts, languages, filenames):
        mp3_filename = root_fn + ".mp3"
        wav_filename = root_fn + ".wav"

        # 1. Synthesize text to MP3
        synthesize(text, lang, mp3_filename)

        # 2. Convert MP3 to WAV using librosa and soundfile
        audio_data, sr_rate = librosa.load(mp3_filename, sr=None)
        sf.write(wav_filename, audio_data, sr_rate)

        # 3. Recognize text from WAV file using SpeechRecognition
        with sr.AudioFile(wav_filename) as source:
            audio = recognizer.record(source)

        try:
            # Use Google Speech Recognition with the target language code
            rec_text = recognizer.recognize_google(audio, language=lang)
        except (sr.UnknownValueError, sr.RequestError):
            rec_text = ""

        recognized_texts.append(rec_text)

    return recognized_texts