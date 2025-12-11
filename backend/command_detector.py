"""
Voice Command Detection Module
Detects wake words and commands from transcribed text
"""

import re
import logging
from typing import Optional, List, Dict
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class CommandDetector:
    """
    Voice command detection and processing
    Supports wake word detection and various system commands
    """

    def __init__(self, wake_word: str = "hey dsp"):
        self.wake_word = wake_word.lower()
        self.commands = self._initialize_commands()
        self.last_command = None

    def _initialize_commands(self) -> Dict[str, dict]:
        """Initialize available voice commands"""
        return {
            # Recording commands
            "start recording": {
                "action": "start_recording",
                "category": "recording",
                "description": "Start audio recording",
                "aliases": ["begin recording", "start record", "record now"]
            },
            "stop recording": {
                "action": "stop_recording",
                "category": "recording",
                "description": "Stop audio recording",
                "aliases": ["end recording", "stop record", "finish recording"]
            },

            # DSP commands
            "enable dsp": {
                "action": "enable_dsp",
                "category": "dsp",
                "description": "Enable DSP noise reduction",
                "aliases": ["turn on dsp", "activate dsp", "dsp on"]
            },
            "disable dsp": {
                "action": "disable_dsp",
                "category": "dsp",
                "description": "Disable DSP noise reduction",
                "aliases": ["turn off dsp", "deactivate dsp", "dsp off"]
            },
            "enable noise reduction": {
                "action": "enable_dsp",
                "category": "dsp",
                "description": "Enable noise reduction",
                "aliases": ["noise reduction on", "reduce noise"]
            },

            # Gender prediction commands
            "predict gender": {
                "action": "predict_gender",
                "category": "ml",
                "description": "Predict speaker gender",
                "aliases": ["detect gender", "what gender", "analyze gender"]
            },
            "enable gender prediction": {
                "action": "enable_gender_prediction",
                "category": "ml",
                "description": "Enable automatic gender prediction",
                "aliases": ["gender prediction on"]
            },
            "disable gender prediction": {
                "action": "disable_gender_prediction",
                "category": "ml",
                "description": "Disable automatic gender prediction",
                "aliases": ["gender prediction off"]
            },

            # Data management commands
            "save transcript": {
                "action": "save_transcript",
                "category": "data",
                "description": "Save current transcript",
                "aliases": ["save this", "save recording"]
            },
            "show history": {
                "action": "show_history",
                "category": "data",
                "description": "Display recording history",
                "aliases": ["view history", "display history", "show recordings"]
            },
            "clear history": {
                "action": "clear_history",
                "category": "data",
                "description": "Clear all recording history",
                "aliases": ["delete history", "erase history", "remove history"]
            },

            # UI commands
            "clear screen": {
                "action": "clear_screen",
                "category": "ui",
                "description": "Clear the screen",
                "aliases": ["clear display", "clean screen"]
            },
            "show stats": {
                "action": "show_stats",
                "category": "ui",
                "description": "Show statistics",
                "aliases": ["display stats", "view statistics"]
            },

            # System commands
            "help": {
                "action": "show_help",
                "category": "system",
                "description": "Show available commands",
                "aliases": ["what can you do", "show commands", "list commands"]
            },
            "status": {
                "action": "show_status",
                "category": "system",
                "description": "Show system status",
                "aliases": ["system status", "check status"]
            }
        }

    def detect_wake_word(self, text: str) -> bool:
        """
        Detect wake word in transcribed text
        Uses fuzzy matching to handle variations
        """
        if not text:
            return False

        text_lower = text.lower()

        # Direct match
        if self.wake_word in text_lower:
            logger.info(f"Wake word detected: {self.wake_word}")
            return True

        # Fuzzy match with similarity threshold
        words = text_lower.split()
        wake_words = self.wake_word.split()

        # Check for consecutive word matches
        for i in range(len(words) - len(wake_words) + 1):
            phrase = " ".join(words[i:i+len(wake_words)])
            similarity = SequenceMatcher(None, phrase, self.wake_word).ratio()

            if similarity > 0.8:  # 80% similarity threshold
                logger.info(f"Wake word detected (fuzzy): {phrase} -> {self.wake_word}")
                return True

        return False

    def extract_command(self, text: str, require_wake_word: bool = False) -> Optional[dict]:
        """
        Extract command from transcribed text

        Args:
            text: Transcribed text
            require_wake_word: If True, only detect commands after wake word

        Returns:
            Dictionary with command details or None
        """
        if not text:
            return {
                "detected": False,
                "command": None,
                "text": None,
                "had_wake_word": False
            }

        text_lower = text.lower()

        # Check for wake word if required
        has_wake_word = self.detect_wake_word(text)

        if require_wake_word and not has_wake_word:
            return None

        # Remove wake word from text for cleaner command detection
        if has_wake_word:
            text_lower = text_lower.replace(self.wake_word, "").strip()

        # Try to match commands
        best_match = None
        best_score = 0

        for cmd_text, cmd_info in self.commands.items():
            # Check exact match
            if cmd_text in text_lower:
                return {
                    "detected": True,
                    "command": cmd_info["action"],
                    "text": cmd_text,
                    "category": cmd_info["category"],
                    "description": cmd_info["description"],
                    "had_wake_word": has_wake_word,
                    "match_type": "exact"
                }

            # Check aliases
            for alias in cmd_info.get("aliases", []):
                if alias in text_lower:
                    return {
                        "detected": True,
                        "command": cmd_info["action"],
                        "text": alias,
                        "category": cmd_info["category"],
                        "description": cmd_info["description"],
                        "had_wake_word": has_wake_word,
                        "match_type": "alias"
                    }

            # Fuzzy matching
            similarity = SequenceMatcher(None, text_lower, cmd_text).ratio()
            if similarity > best_score and similarity > 0.7:
                best_score = similarity
                best_match = {
                    "detected": True,
                    "command": cmd_info["action"],
                    "text": cmd_text,
                    "category": cmd_info["category"],
                    "description": cmd_info["description"],
                    "had_wake_word": has_wake_word,
                    "match_type": "fuzzy",
                    "confidence": similarity
                }

        if best_match:
            logger.info(f"Command detected: {best_match['command']} (confidence: {best_match.get('confidence', 1.0):.2f})")
            return best_match

        # No command detected
        return {
            "detected": False,
            "command": None,
            "text": None,
            "had_wake_word": has_wake_word
        }

    def get_available_commands(self, category: Optional[str] = None) -> List[dict]:
        """
        Get list of available commands

        Args:
            category: Filter by category (recording, dsp, ml, data, ui, system)

        Returns:
            List of command dictionaries
        """
        commands_list = []

        for cmd_text, cmd_info in self.commands.items():
            if category is None or cmd_info["category"] == category:
                commands_list.append({
                    "text": cmd_text,
                    "action": cmd_info["action"],
                    "category": cmd_info["category"],
                    "description": cmd_info["description"],
                    "aliases": cmd_info.get("aliases", [])
                })

        return commands_list

    def get_command_categories(self) -> List[str]:
        """Get all command categories"""
        categories = set()
        for cmd_info in self.commands.values():
            categories.add(cmd_info["category"])
        return sorted(list(categories))

    def validate_command(self, action: str) -> bool:
        """Check if a command action is valid"""
        for cmd_info in self.commands.values():
            if cmd_info["action"] == action:
                return True
        return False

    def get_command_help(self) -> str:
        """Generate help text for all commands"""
        help_text = f"Wake Word: '{self.wake_word}'\n\n"
        help_text += "Available Commands:\n\n"

        categories = self.get_command_categories()

        for category in categories:
            help_text += f"{category.upper()}:\n"
            commands = self.get_available_commands(category)

            for cmd in commands:
                help_text += f"  • {cmd['text']}: {cmd['description']}\n"
                if cmd['aliases']:
                    help_text += f"    Aliases: {', '.join(cmd['aliases'])}\n"

            help_text += "\n"

        return help_text
