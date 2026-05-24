"""
Data validation utilities for Valolyzer.
Provides lightweight schema validation for matches, maps, compositions, and player stats.
"""

from typing import Dict, List, Any, Optional, Tuple
from utils.logging import get_logger

logger = get_logger(__name__)


class DataValidators:
    """Lightweight validators for data objects."""

    @staticmethod
    def validate_match(match: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate match object.

        Args:
            match: Match dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["match_id", "team_a", "team_b", "source"]
        
        for field in required_fields:
            if not match.get(field):
                return False, f"Missing required field: {field}"
        
        if match.get("match_id") is None or match.get("match_id") == "":
            return False, "match_id cannot be empty"
        
        if match.get("team_a") == match.get("team_b"):
            return False, "team_a and team_b must be different"
        
        return True, ""

    @staticmethod
    def validate_map(map_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate map object.

        Args:
            map_data: Map dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["map_id", "match_id", "map_name", "map_order", "source"]
        
        for field in required_fields:
            if map_data.get(field) is None:
                return False, f"Missing required field: {field}"
        
        if not isinstance(map_data.get("map_order"), int):
            return False, "map_order must be an integer"
        
        if map_data.get("map_order", -1) < 0:
            return False, "map_order must be non-negative"
        
        return True, ""

    @staticmethod
    def validate_composition(comp: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate composition object.

        Args:
            comp: Composition dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["map_id", "team", "source"]
        
        for field in required_fields:
            if not comp.get(field):
                return False, f"Missing required field: {field}"
        
        agent_fields = [f"agent_{i}" for i in range(1, 6)]
        agents = [comp.get(field) for field in agent_fields]
        agent_count = len([a for a in agents if a])
        
        if agent_count < 5:
            return False, f"Must have 5 agents, found {agent_count}"
        
        return True, ""

    @staticmethod
    def validate_player_stats(stats: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate player stats object.

        Args:
            stats: Player stats dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["map_id", "player", "agent", "source"]
        
        for field in required_fields:
            if not stats.get(field):
                return False, f"Missing required field: {field}"
        
        # Validate numeric fields
        numeric_fields = ["kills", "deaths", "assists", "acs", "adr"]
        for field in numeric_fields:
            value = stats.get(field, 0)
            if not isinstance(value, (int, float)):
                return False, f"{field} must be numeric"
            if value < 0:
                return False, f"{field} cannot be negative"
        
        return True, ""

    @staticmethod
    def validate_round(round_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate round object.

        Args:
            round_data: Round dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["round_id", "map_id", "round_number", "winner", "source"]
        
        for field in required_fields:
            if not round_data.get(field):
                return False, f"Missing required field: {field}"
        
        if not isinstance(round_data.get("round_number"), int):
            return False, "round_number must be an integer"
        
        return True, ""


class BulkValidator:
    """Validate collections of objects."""

    @staticmethod
    def validate_matches(matches: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """
        Validate list of matches.

        Returns:
            Tuple of (valid_count, error_list)
        """
        valid_count = 0
        errors = []
        
        for idx, match in enumerate(matches):
            is_valid, error = DataValidators.validate_match(match)
            if is_valid:
                valid_count += 1
            else:
                errors.append(f"Match {idx}: {error}")
        
        return valid_count, errors

    @staticmethod
    def validate_maps(maps: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """Validate list of maps."""
        valid_count = 0
        errors = []
        
        for idx, map_data in enumerate(maps):
            is_valid, error = DataValidators.validate_map(map_data)
            if is_valid:
                valid_count += 1
            else:
                errors.append(f"Map {idx}: {error}")
        
        return valid_count, errors

    @staticmethod
    def validate_all(raw_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Validate all data types in a raw_data dict.

        Returns:
            Summary of validation results
        """
        results = {}
        
        matches = raw_data.get("matches", [])
        valid_matches, match_errors = BulkValidator.validate_matches(matches)
        results["matches"] = {
            "total": len(matches),
            "valid": valid_matches,
            "invalid": len(matches) - valid_matches,
            "errors": match_errors[:5],  # First 5 errors
        }
        
        maps = raw_data.get("maps", [])
        valid_maps, map_errors = BulkValidator.validate_maps(maps)
        results["maps"] = {
            "total": len(maps),
            "valid": valid_maps,
            "invalid": len(maps) - valid_maps,
            "errors": map_errors[:5],
        }
        
        return results
