from flask import Flask, request, jsonify
import speech_recognition as sr
import os

app = Flask(__name__)
recognizer = sr.Recognizer()

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        # Check if audio file is in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        temp_path = 'temp_audio.wav'
        audio_file.save(temp_path)

        # Process audio with speech recognition
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
            try:
                # Use Google's Speech-to-Text API
                text = recognizer.recognize_google(audio)
                os.remove(temp_path)  # Clean up
                return jsonify({'text': text})
            except sr.UnknownValueError:
                os.remove(temp_path)
                return jsonify({'error': 'Could not understand audio'}), 400
            except sr.RequestError as e:
                os.remove(temp_path)
                return jsonify({'error': f'Speech recognition error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
