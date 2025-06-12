from flask import Flask, request, jsonify, send_file
import whisper
from TTS.api import TTS
import os
import uuid

app = Flask(__name__)
# Load Whisper model for transcription
whisper_model = whisper.load_model("tiny")  # Use 'base' or 'small' for better accuracy
# Load Coqui TTS model for realistic speech (VITS model for English)
tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    temp_audio_path = None
    temp_tts_path = None
    try:
        # Check if audio file is in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        temp_audio_path = f'temp_audio_{uuid.uuid4()}.wav'
        audio_file.save(temp_audio_path)

        # Transcribe audio using Whisper
        result = whisper_model.transcribe(temp_audio_path)
        transcribed_text = result['text'].strip()

        if not transcribed_text:
            return jsonify({'error': 'No speech detected in audio'}), 400

        # Generate realistic TTS audio using Coqui TTS
        temp_tts_path = f'temp_tts_{uuid.uuid4()}.wav'
        tts_model.tts_to_file(text=transcribed_text, file_path=temp_tts_path)

        # Clean up input audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        # Serve TTS audio file
        tts_url = f'/tts/{os.path.basename(temp_tts_path)}'
        return jsonify({'text': transcribed_text, 'audio_url': tts_url})

    except Exception as e:
        # Clean up files in case of error
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if temp_tts_path and os.path.exists(temp_tts_path):
            os.remove(temp_tts_path)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/tts/<filename>', methods=['GET'])
def serve_tts_audio(filename):
    tts_path = filename
    try:
        return send_file(tts_path, mimetype='audio/wav', as_attachment=False)
    except Exception as e:
        return jsonify({'error': f'Error serving audio: {str(e)}'}), 500
    finally:
        # Clean up TTS file after serving
        if os.path.exists(tts_path):
            os.remove(tts_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
