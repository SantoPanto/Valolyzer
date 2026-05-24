"""
Debug utilities for Valolyzer.
Saves problematic raw data for inspection and debugging.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from utils.logging import get_logger

logger = get_logger(__name__)


class DebugManager:
    """Manage debug output and payload inspection."""

    DEBUG_DIR = Path("data/debug")

    @classmethod
    def ensure_debug_dir(cls):
        """Ensure debug directory exists."""
        cls.DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def save_failed_map(cls, map_id: str, raw_map: Dict[str, Any], reason: str):
        """
        Save a failed map payload for inspection.

        Args:
            map_id: Map identifier
            raw_map: Raw map dictionary
            reason: Reason for failure
        """
        cls.ensure_debug_dir()
        
        filename = f"failed_map_{map_id}.json"
        filepath = cls.DEBUG_DIR / filename
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "map_id": map_id,
            "reason": reason,
            "data": raw_map,
        }
        
        try:
            with open(filepath, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.debug(f"Saved failed map to {filepath}")
        except Exception as e:
            logger.error(f"Error saving failed map: {e}")

    @classmethod
    def save_failed_composition(cls, comp_id: str, raw_comp: Dict[str, Any], reason: str):
        """Save a failed composition payload."""
        cls.ensure_debug_dir()
        
        filename = f"failed_composition_{comp_id}.json"
        filepath = cls.DEBUG_DIR / filename
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "comp_id": comp_id,
            "reason": reason,
            "data": raw_comp,
        }
        
        try:
            with open(filepath, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.debug(f"Saved failed composition to {filepath}")
        except Exception as e:
            logger.error(f"Error saving failed composition: {e}")

    @classmethod
    def save_failed_player_stats(cls, stats_id: str, raw_stats: Dict[str, Any], reason: str):
        """Save a failed player stats payload."""
        cls.ensure_debug_dir()
        
        filename = f"failed_player_stats_{stats_id}.json"
        filepath = cls.DEBUG_DIR / filename
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "stats_id": stats_id,
            "reason": reason,
            "data": raw_stats,
        }
        
        try:
            with open(filepath, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.debug(f"Saved failed player stats to {filepath}")
        except Exception as e:
            logger.error(f"Error saving failed player stats: {e}")

    @classmethod
    def save_raw_data_summary(cls, raw_data: Dict[str, List[Dict[str, Any]]]):
        """Save summary of raw data for inspection."""
        cls.ensure_debug_dir()
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "matches": len(raw_data.get("matches", [])),
            "maps": len(raw_data.get("maps", [])),
            "compositions": len(raw_data.get("compositions", [])),
            "player_stats": len(raw_data.get("player_stats", [])),
            "rounds": len(raw_data.get("rounds", [])),
            "sample_match": raw_data.get("matches", [{}])[0] if raw_data.get("matches") else None,
            "sample_map": raw_data.get("maps", [{}])[0] if raw_data.get("maps") else None,
        }
        
        filepath = cls.DEBUG_DIR / "raw_data_summary.json"
        
        try:
            with open(filepath, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Saved raw data summary to {filepath}")
        except Exception as e:
            logger.error(f"Error saving raw data summary: {e}")

    @classmethod
    def save_parser_errors(cls, errors: List[Dict[str, Any]]):
        """Save parser errors for inspection."""
        cls.ensure_debug_dir()
        
        filepath = cls.DEBUG_DIR / "parser_errors.json"
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "total_errors": len(errors),
            "errors": errors[:100],  # Save first 100 errors
        }
        
        try:
            with open(filepath, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info(f"Saved {len(errors)} parser errors to {filepath}")
        except Exception as e:
            logger.error(f"Error saving parser errors: {e}")

    @classmethod
    def cleanup_old_debug_files(cls, max_files: int = 100):
        """Remove old debug files if directory gets too large."""
        cls.ensure_debug_dir()
        
        try:
            files = sorted(cls.DEBUG_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if len(files) > max_files:
                for file in files[:-max_files]:
                    file.unlink()
                    logger.debug(f"Removed old debug file: {file}")
        except Exception as e:
            logger.error(f"Error cleaning up debug files: {e}")
