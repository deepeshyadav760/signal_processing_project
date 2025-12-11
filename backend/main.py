"""
AI Speech Intelligence Platform - Main API
Integrates DSP, Whisper ASR, Gender Classification, and Voice Commands
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import base64
from datetime import datetime
import logging
from typing import Optional
import soundfile as sf
import io
from pydub import AudioSegment

# Import custom modules
from dsp_processor import DSPProcessor
from gender_classifier import GenderClassifier
from whisper_service import WhisperService
from command_detector import CommandDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Speech Intelligence Platform",
    description="Real-time DSP + Whisper + ML Voice Processing System",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
logger.info("Initializing services...")

dsp_processor = DSPProcessor(sample_rate=16000)
gender_classifier = GenderClassifier(sample_rate=16000)
command_detector = CommandDetector(wake_word="hey dsp")

# Initialize Whisper with REAL model (transformers installed!)
try:
    whisper_service = WhisperService(
        model_name="openai/whisper-base",  # Using base - better accuracy than tiny
        use_local=True  # REAL Whisper enabled!
    )
    logger.info("Whisper service initialized with REAL model (base, English)")
except Exception as e:
    logger.warning(f"Failed to initialize Whisper, falling back to mock: {e}")
    whisper_service = WhisperService(use_local=False)

# In-memory storage
transcripts_db = []
stats_db = {
    "total_recordings": 0,
    "total_duration": 0,
    "genders": {"male": 0, "female": 0, "unknown": 0},
    "dsp_usage": {"enabled": 0, "disabled": 0},
    "commands_detected": 0
}

# ==================== API MODELS ====================

class TranscriptRequest(BaseModel):
    audio_base64: str
    enable_dsp: bool = True
    predict_gender_flag: bool = True

class CommandRequest(BaseModel):
    text: str
    require_wake_word: bool = False

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "AI Speech Intelligence Platform API",
        "version": "2.0.0",
        "status": "running",
        "services": {
            "dsp": "active",
            "whisper": "active" if whisper_service.is_available() else "mock",
            "gender_classifier": "active",
            "command_detector": "active"
        },
        "endpoints": [
            "/api/transcribe",
            "/api/detect-command",
            "/api/history",
            "/api/stats",
            "/api/commands",
            "/api/status"
        ]
    }

@app.post("/api/transcribe")
async def transcribe_audio(request: TranscriptRequest):
    """
    Main transcription endpoint

    Process flow:
    1. Decode base64 audio
    2. Apply DSP pipeline (if enabled)
    3. Transcribe with Whisper
    4. Predict gender (if enabled)
    5. Detect voice commands
    6. Store results and update stats
    """
    try:
        # Decode audio
        logger.info(f"Received base64 audio: {len(request.audio_base64)} chars")
        audio_bytes = base64.b64decode(request.audio_base64)
        logger.info(f"Decoded to bytes: {len(audio_bytes)} bytes")
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        logger.info(f"Converted to audio: {len(audio_data)} samples ({len(audio_data)/16000:.2f}s)")
        logger.info(f"Audio energy: {np.mean(np.abs(audio_data)):.6f}, max: {np.max(np.abs(audio_data)):.6f}")

        # Step 1: Apply DSP processing
        dsp_result = dsp_processor.process_audio(audio_data, enable_dsp=request.enable_dsp)
        audio_clean = dsp_result["audio"]

        logger.info(f"DSP applied: {dsp_result['dsp_applied']}, Voice ratio: {dsp_result['voice_ratio']:.2f}")

        # Step 2: Transcribe with Whisper
        transcription_result = whisper_service.transcribe(audio_clean, sample_rate=16000)
        transcript = transcription_result.get("text", "")

        if not transcription_result.get("success", False):
            logger.warning(f"Transcription failed: {transcription_result.get('error', 'Unknown error')}")

        logger.info(f"Transcript: {transcript}")

        # Step 3: Predict gender
        gender_result = {"gender": "unknown", "confidence": 0.0, "features": {}}

        if request.predict_gender_flag and len(audio_clean) > 8000:
            gender_result = gender_classifier.predict_gender(audio_clean)

        # Step 4: Detect voice commands
        command_result = command_detector.extract_command(transcript, require_wake_word=False)

        if command_result.get("detected", False):
            logger.info(f"Command detected: {command_result['command']}")

        # Step 5: Store record
        record = {
            "id": len(transcripts_db) + 1,
            "timestamp": datetime.now().isoformat(),
            "transcript": transcript,
            "gender": gender_result["gender"],
            "gender_confidence": gender_result.get("confidence", 0.0),
            "duration": len(audio_data) / 16000,
            "dsp_enabled": request.enable_dsp,
            "dsp_stats": {
                "voice_ratio": dsp_result.get("voice_ratio", 0),
                "energy": dsp_result.get("energy", 0)
            },
            "command": command_result,
            "features": gender_result.get("features", {})
        }

        transcripts_db.append(record)

        # Step 6: Update statistics
        stats_db["total_recordings"] += 1
        stats_db["total_duration"] += record["duration"]

        gender = gender_result["gender"]
        if gender in ["male", "female", "unknown"]:
            stats_db["genders"][gender] += 1

        if request.enable_dsp:
            stats_db["dsp_usage"]["enabled"] += 1
        else:
            stats_db["dsp_usage"]["disabled"] += 1

        if command_result.get("detected", False):
            stats_db["commands_detected"] += 1

        # Store audio for download
        audio_original = dsp_result.get("audio_original", audio_data)
        record["audio_original"] = audio_original
        record["audio_processed"] = audio_clean
        record["sample_rate"] = 16000

        # Convert audio to base64 for waveform visualization
        audio_original_b64 = base64.b64encode(audio_original.astype(np.float32).tobytes()).decode('utf-8')
        audio_processed_b64 = base64.b64encode(audio_clean.astype(np.float32).tobytes()).decode('utf-8')

        # Return response with detailed DSP and gender info
        return {
            "success": True,
            "transcript": transcript,
            "gender": gender_result["gender"],
            "gender_confidence": gender_result.get("confidence", 0.0),
            "gender_features": {
                "pitch": gender_result.get("features", {}).get("pitch", 0),
                "formant_f1": gender_result.get("features", {}).get("formant_f1", 0),
                "spectral_centroid": gender_result.get("features", {}).get("spectral_centroid", 0)
            },
            "command": command_result,
            "duration": record["duration"],
            "record_id": record["id"],
            "dsp_stats": {
                "enabled": request.enable_dsp,
                "voice_ratio": dsp_result.get("voice_ratio", 0),
                "energy": dsp_result.get("energy", 0),
                "noise_level": dsp_result.get("noise_level", 0) if dsp_result.get("dsp_applied") else 0,
                "noise_reduction_db": dsp_result.get("noise_reduction_db", 0) if dsp_result.get("dsp_applied") else 0,
                "psnr": dsp_result.get("psnr", 0) if dsp_result.get("dsp_applied") else 0,
                "snr": dsp_result.get("snr", 0) if dsp_result.get("dsp_applied") else 0,
                "applied": dsp_result.get("dsp_applied", False)
            },
            "waveform": {
                "original": audio_original_b64,
                "processed": audio_processed_b64,
                "sample_rate": 16000,
                "length": len(audio_original)
            },
            "whisper_method": transcription_result.get("method", "unknown")
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    enable_dsp: bool = True,
    predict_gender: bool = True
):
    """
    Upload audio file (.wav or .mp3) and process it
    Returns transcription, gender prediction, DSP analysis, and waveform data
    """
    try:
        logger.info(f"📁 Received file: {file.filename}, type: {file.content_type}")

        # Read file content
        content = await file.read()
        logger.info(f"📦 File size: {len(content)} bytes")

        # Convert to audio array based on file type
        if file.filename.lower().endswith('.wav'):
            # Handle WAV files
            audio_data, sample_rate = sf.read(io.BytesIO(content))
            logger.info(f"🎵 WAV file loaded: {len(audio_data)} samples, {sample_rate}Hz")
        elif file.filename.lower().endswith('.mp3'):
            # Handle MP3 files using pydub
            audio_segment = AudioSegment.from_mp3(io.BytesIO(content))
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            # If stereo, convert to mono
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)
            # Normalize to float32 [-1, 1]
            audio_data = samples.astype(np.float32) / (2**15)
            sample_rate = audio_segment.frame_rate
            logger.info(f"🎵 MP3 file loaded: {len(audio_data)} samples, {sample_rate}Hz")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use .wav or .mp3")

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
            logger.info(f"🔄 Resampled to 16kHz: {len(audio_data)} samples")

        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        duration = len(audio_data) / sample_rate
        logger.info(f"⏱️ Audio duration: {duration:.2f}s")

        # DSP Processing
        dsp_result = dsp_processor.process_audio(audio_data, enable_dsp=enable_dsp)
        audio_clean = dsp_result["audio"]
        audio_original = dsp_result.get("audio_original", audio_data)

        # Whisper Transcription
        transcription_result = whisper_service.transcribe(audio_clean, sample_rate=16000)
        transcript = transcription_result.get("text", "")

        # Gender Prediction
        gender_result = gender_classifier.predict_gender(audio_clean) if predict_gender else {
            "gender": "unknown", "confidence": 0, "features": {}
        }

        # Command Detection
        command_result = command_detector.extract_command(transcript)

        # Convert audio to base64 for waveform visualization
        audio_original_b64 = base64.b64encode(audio_original.astype(np.float32).tobytes()).decode('utf-8')
        audio_processed_b64 = base64.b64encode(audio_clean.astype(np.float32).tobytes()).decode('utf-8')

        # Store processed audio in memory for download
        record_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        transcripts_db.append({
            "id": record_id,
            "transcript": transcript,
            "gender": gender_result["gender"],
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "audio_original": audio_original,
            "audio_processed": audio_clean,
            "sample_rate": 16000
        })

        logger.info(f"✅ File processed successfully: {transcript[:50]}...")

        return {
            "success": True,
            "transcript": transcript,
            "gender": gender_result["gender"],
            "gender_confidence": gender_result["confidence"],
            "gender_features": gender_result["features"],
            "command": command_result,
            "duration": duration,
            "record_id": record_id,
            "dsp_stats": {
                "enabled": enable_dsp,
                "voice_ratio": dsp_result.get("voice_ratio", 0),
                "energy": dsp_result.get("energy", 0),
                "noise_level": dsp_result.get("noise_level", 0),
                "noise_reduction_db": dsp_result.get("noise_reduction_db", 0),
                "psnr": dsp_result.get("psnr", 0),
                "snr": dsp_result.get("snr", 0),
                "applied": dsp_result.get("dsp_applied", False)
            },
            "waveform": {
                "original": audio_original_b64,
                "processed": audio_processed_b64,
                "sample_rate": 16000,
                "length": len(audio_original)
            },
            "whisper_method": transcription_result.get("method", "unknown")
        }

    except Exception as e:
        logger.error(f"File upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-audio/{record_id}")
async def download_audio(record_id: str, format: str = "wav"):
    """
    Download processed audio file in WAV or MP3 format
    """
    try:
        # Find record
        record = None
        for r in transcripts_db:
            if r["id"] == record_id:
                record = r
                break

        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        audio_data = record.get("audio_processed")
        sample_rate = record.get("sample_rate", 16000)

        if audio_data is None:
            raise HTTPException(status_code=404, detail="Processed audio not available")

        # Generate audio file
        if format.lower() == "wav":
            # Create WAV file in memory
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            buffer.seek(0)

            return Response(
                content=buffer.getvalue(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=processed_{record_id}.wav"
                }
            )
        elif format.lower() == "mp3":
            # Create WAV first, then convert to MP3
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format='WAV')
            wav_buffer.seek(0)

            # Convert to MP3 using pydub
            audio_segment = AudioSegment.from_wav(wav_buffer)
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3", bitrate="192k")
            mp3_buffer.seek(0)

            return Response(
                content=mp3_buffer.getvalue(),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"attachment; filename=processed_{record_id}.mp3"
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Format must be 'wav' or 'mp3'")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect-command")
async def detect_command(request: CommandRequest):
    """Detect command from text"""
    try:
        result = command_detector.extract_command(
            request.text,
            require_wake_word=request.require_wake_word
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        logger.error(f"Command detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(limit: int = 20, gender: Optional[str] = None):
    """
    Get transcript history with optional filtering

    Args:
        limit: Maximum number of records to return
        gender: Filter by gender (male/female/unknown)
    """
    try:
        records = transcripts_db

        # Filter by gender if specified
        if gender:
            records = [r for r in records if r.get("gender") == gender]

        # Get last N records
        records = records[-limit:]

        # Remove numpy arrays from records (audio data) for JSON serialization
        serializable_records = []
        for record in records:
            clean_record = {
                "id": record.get("id"),
                "transcript": record.get("transcript"),
                "gender": record.get("gender"),
                "timestamp": record.get("timestamp"),
                "duration": record.get("duration")
            }
            serializable_records.append(clean_record)

        return {
            "success": True,
            "records": serializable_records,
            "total": len(transcripts_db),
            "filtered": len(serializable_records)
        }

    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Calculate additional stats
        avg_duration = (
            stats_db["total_duration"] / stats_db["total_recordings"]
            if stats_db["total_recordings"] > 0
            else 0
        )

        total_genders = sum(stats_db["genders"].values())
        gender_percentages = {
            gender: (count / total_genders * 100 if total_genders > 0 else 0)
            for gender, count in stats_db["genders"].items()
        }

        return {
            "success": True,
            "stats": {
                **stats_db,
                "avg_duration": avg_duration,
                "gender_percentages": gender_percentages
            }
        }

    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/history")
async def clear_history():
    """Clear all history"""
    try:
        transcripts_db.clear()

        stats_db["total_recordings"] = 0
        stats_db["total_duration"] = 0
        stats_db["genders"] = {"male": 0, "female": 0, "unknown": 0}
        stats_db["dsp_usage"] = {"enabled": 0, "disabled": 0}
        stats_db["commands_detected"] = 0

        return {
            "success": True,
            "message": "History cleared successfully"
        }

    except Exception as e:
        logger.error(f"History clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/commands")
async def get_commands(category: Optional[str] = None):
    """Get available voice commands"""
    try:
        commands = command_detector.get_available_commands(category)
        categories = command_detector.get_command_categories()

        return {
            "success": True,
            "wake_word": command_detector.wake_word,
            "commands": commands,
            "categories": categories,
            "total": len(commands)
        }

    except Exception as e:
        logger.error(f"Commands retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Get system status and health check"""
    try:
        whisper_info = whisper_service.get_model_info()

        return {
            "success": True,
            "status": "operational",
            "services": {
                "dsp": {
                    "status": "active",
                    "sample_rate": dsp_processor.sample_rate
                },
                "whisper": {
                    "status": "active" if whisper_info["available"] else "mock",
                    **whisper_info
                },
                "gender_classifier": {
                    "status": "active",
                    "sample_rate": gender_classifier.sample_rate
                },
                "command_detector": {
                    "status": "active",
                    "wake_word": command_detector.wake_word,
                    "total_commands": len(command_detector.commands)
                }
            },
            "database": {
                "total_records": len(transcripts_db),
                "storage_type": "in-memory"
            }
        }

    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/help")
async def get_help():
    """Get help information"""
    try:
        help_text = command_detector.get_command_help()

        return {
            "success": True,
            "help_text": help_text,
            "documentation": "AI Speech Intelligence Platform v2.0",
            "features": [
                "Real-time DSP noise reduction",
                "Whisper-based speech recognition",
                "Gender classification using ML",
                "Voice command detection",
                "Wake word support ('hey dsp')",
                "Session history and statistics"
            ]
        }

    except Exception as e:
        logger.error(f"Help retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=" * 60)
    logger.info("AI Speech Intelligence Platform - Starting")
    logger.info("=" * 60)
    logger.info(f"DSP Processor: Initialized (sample_rate={dsp_processor.sample_rate})")
    logger.info(f"Gender Classifier: Initialized")
    logger.info(f"Command Detector: Initialized (wake_word='{command_detector.wake_word}')")
    logger.info(f"Whisper Service: {whisper_service.get_model_info()}")
    logger.info("=" * 60)
    logger.info("All services ready!")
    logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
