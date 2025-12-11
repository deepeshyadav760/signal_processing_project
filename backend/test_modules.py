"""
Test script to verify all modules work correctly
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dsp_processor():
    """Test DSP processor module"""
    print("\n" + "="*60)
    print("Testing DSP Processor")
    print("="*60)

    try:
        from dsp_processor import DSPProcessor

        dsp = DSPProcessor(sample_rate=16000)

        # Create test audio (1 second of noise + tone)
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Generate test signal: 440 Hz tone + noise
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        noise = 0.1 * np.random.randn(len(t))
        audio = signal + noise

        # Process audio
        result = dsp.process_audio(audio, enable_dsp=True)

        print(f"✓ DSP processing successful")
        print(f"  - Voice ratio: {result['voice_ratio']:.2f}")
        print(f"  - Energy: {result['energy']:.4f}")
        print(f"  - DSP applied: {result['dsp_applied']}")

        return True

    except Exception as e:
        print(f"✗ DSP processor test failed: {e}")
        return False


def test_gender_classifier():
    """Test gender classifier module"""
    print("\n" + "="*60)
    print("Testing Gender Classifier")
    print("="*60)

    try:
        from gender_classifier import GenderClassifier

        classifier = GenderClassifier(sample_rate=16000)

        # Create test audio with male-like pitch (120 Hz)
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        signal = 0.5 * np.sin(2 * np.pi * 120 * t)
        audio = signal + 0.05 * np.random.randn(len(t))

        # Predict gender
        result = classifier.predict_gender(audio)

        print(f"✓ Gender classification successful")
        print(f"  - Gender: {result['gender']}")
        print(f"  - Confidence: {result['confidence']:.2f}")
        print(f"  - Pitch: {result['features'].get('pitch', 0):.1f} Hz")

        return True

    except Exception as e:
        print(f"✗ Gender classifier test failed: {e}")
        return False


def test_command_detector():
    """Test command detector module"""
    print("\n" + "="*60)
    print("Testing Command Detector")
    print("="*60)

    try:
        from command_detector import CommandDetector

        detector = CommandDetector(wake_word="hey dsp")

        # Test wake word detection
        test_texts = [
            "Hey DSP, start recording",
            "enable noise reduction",
            "show me the history",
            "random text without commands"
        ]

        print("Testing command detection:")
        for text in test_texts:
            result = detector.extract_command(text)
            detected = "✓" if result['detected'] else "✗"
            print(f"  {detected} '{text}' -> {result.get('command', 'none')}")

        # Get available commands
        commands = detector.get_available_commands()
        print(f"\n✓ Command detector successful")
        print(f"  - Total commands: {len(commands)}")
        print(f"  - Wake word: '{detector.wake_word}'")

        return True

    except Exception as e:
        print(f"✗ Command detector test failed: {e}")
        return False


def test_whisper_service():
    """Test Whisper service module"""
    print("\n" + "="*60)
    print("Testing Whisper Service")
    print("="*60)

    try:
        from whisper_service import WhisperService

        # Initialize with mock mode for testing
        whisper = WhisperService(use_local=False)

        # Create test audio
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        signal = 0.3 * np.sin(2 * np.pi * 440 * t)
        audio = signal + 0.05 * np.random.randn(len(t))

        # Transcribe
        result = whisper.transcribe(audio, sample_rate=16000)

        print(f"✓ Whisper service successful (mock mode)")
        print(f"  - Method: {result.get('method', 'unknown')}")
        print(f"  - Success: {result.get('success', False)}")
        print(f"  - Text: '{result.get('text', '')[:50]}...'")

        # Try loading real model (may fail without transformers)
        print("\nAttempting to load real Whisper model...")
        try:
            whisper_real = WhisperService(
                model_name="openai/whisper-tiny",
                use_local=True
            )
            info = whisper_real.get_model_info()
            print(f"  - Model available: {info['available']}")
            print(f"  - Model name: {info['model_name']}")
        except Exception as e:
            print(f"  - Real model not available: {e}")
            print(f"  - This is OK - system will use mock mode")

        return True

    except Exception as e:
        print(f"✗ Whisper service test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("AI SPEECH INTELLIGENCE PLATFORM")
    print("Module Testing Suite")
    print("="*60)

    results = {
        "DSP Processor": test_dsp_processor(),
        "Gender Classifier": test_gender_classifier(),
        "Command Detector": test_command_detector(),
        "Whisper Service": test_whisper_service()
    }

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)

    for module, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {module}")

    all_passed = all(results.values())

    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYour system is ready to use!")
        print("Start the server with: python main.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease check the errors above.")
        print("You may need to install additional dependencies.")

    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
