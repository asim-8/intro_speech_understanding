import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Use sr.Recognizer.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text.
    
    @params:
    filename (str) - the filename from which to read the audio
    language (str) - the language of the audio
    
    @returns:
    text (str) - the recognized speech
    '''
    # Initialize the speech recognition manager engine instance
    recognizer = sr.Recognizer()
    
    # Safely load the target WAV sound file into context memory
    with sr.AudioFile(filename) as source:
        # Extract audio records stream from the entire file duration
        audio_data = recognizer.record(source)
        
    try:
        # Process the digital recording stream with Google Web Speech API
        text = recognizer.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        # Raised if the acoustic model cannot interpret or resolve the words
        return ""
    except sr.RequestError as e:
        # Raised if the cloud server API endpoint is inaccessible
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""