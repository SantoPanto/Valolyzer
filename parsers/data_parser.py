"""
Data parsers for normalizing and transforming raw data.
"""

from typing import Dict, List, Any, Optional
from utils.normalizers import Normalizers
from utils.logging import get_logger

logger = get_logger(__name__)


class MatchParser:
    """Parse and normalize match-level data."""

    @staticmethod
    def parse(raw_match: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize raw match data.

        Args:
            raw_match: Raw match dictionary from scraper

        Returns:
            Normalized match data
        """
        return {
            "match_id": raw_match.get("match_id"),
            "event": raw_match.get("event"),
            "date": raw_match.get("date"),
            "patch": Normalizers.normalize_patch(raw_match.get("patch", "")),
            "bo_type": Normalizers.normalize_bo_type(raw_match.get("bo_type", "")),
            "team_a": Normalizers.normalize_team_name(raw_match.get("team_a")),
            "team_b": Normalizers.normalize_team_name(raw_match.get("team_b")),
            "winner": Normalizers.normalize_team_name(raw_match.get("winner")),
            "score_a": raw_match.get("score_a"),
            "score_b": raw_match.get("score_b"),
            "maps_played": raw_match.get("maps_played"),
            "source": raw_match.get("source", "unknown"),
        }

    @staticmethod
    def validate(match: Dict[str, Any]) -> bool:
        """Validate required fields."""
        required = ["match_id", "team_a", "team_b"]
        return all(match.get(field) for field in required)


class MapParser:
    """Parse and normalize map-level data."""

    @staticmethod
    def parse(raw_map: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw map data."""
        raw_map_name = raw_map.get("map_name")
        normalized_map_name = Normalizers.normalize_map_name(raw_map_name)
        
        # Fallback: if normalization fails but we have a map_name, keep the original
        # (defensive parsing - better to preserve data than lose it)
        if normalized_map_name is None and raw_map_name:
            logger.debug(f"Map normalization failed for '{raw_map_name}', using raw value")
            normalized_map_name = raw_map_name.strip()
        
        return {
            "map_id": raw_map.get("map_id"),
            "match_id": raw_map.get("match_id"),
            "map_name": normalized_map_name,
            "map_order": raw_map.get("map_order"),
            "team_a_score": raw_map.get("team_a_score", 0),
            "team_b_score": raw_map.get("team_b_score", 0),
            "attacker_start": raw_map.get("attacker_start"),
            "duration_seconds": raw_map.get("duration_seconds"),
            "source": raw_map.get("source", "unknown"),
        }

    @staticmethod
    def validate(map_data: Dict[str, Any]) -> bool:
        """Validate required fields."""
        required = ["map_id", "match_id", "map_name", "map_order"]
        return all(map_data.get(field) is not None for field in required)


class CompositionParser:
    """Parse and normalize agent composition data."""

    @staticmethod
    def parse(raw_comp: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw composition data."""
        agents = raw_comp.get("agents", [])

        # Normalize each agent
        normalized_agents = [
            Normalizers.normalize_agent_name(a) for a in agents
        ]
        normalized_agents = [a for a in normalized_agents if a]  # Remove None

        return {
            "map_id": raw_comp.get("map_id"),
            "team": Normalizers.normalize_team_name(raw_comp.get("team")),
            "agent_1": normalized_agents[0] if len(normalized_agents) > 0 else None,
            "agent_2": normalized_agents[1] if len(normalized_agents) > 1 else None,
            "agent_3": normalized_agents[2] if len(normalized_agents) > 2 else None,
            "agent_4": normalized_agents[3] if len(normalized_agents) > 3 else None,
            "agent_5": normalized_agents[4] if len(normalized_agents) > 4 else None,
            "source": raw_comp.get("source", "unknown"),
        }

    @staticmethod
    def validate(comp: Dict[str, Any]) -> bool:
        """Validate required fields - must have map_id, team, and at least 5 agent slots."""
        agents = [
            comp.get(f"agent_{i}")
            for i in range(1, 6)
        ]
        # Check if we have map_id and team, and if all 5 agent slots are filled
        return (comp.get("map_id") and 
                comp.get("team") and 
                len([a for a in agents if a]) == 5)


class PlayerStatsParser:
    """Parse and normalize player statistics."""

    @staticmethod
    def parse(raw_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw player stats."""
        kills = raw_stats.get("kills", 0)
        deaths = raw_stats.get("deaths", 0)

        kd_ratio = kills / deaths if deaths > 0 else kills

        return {
            "map_id": raw_stats.get("map_id"),
            "player": raw_stats.get("player"),
            "team": Normalizers.normalize_team_name(raw_stats.get("team")),
            "agent": Normalizers.normalize_agent_name(raw_stats.get("agent")),
            "kills": kills,
            "deaths": deaths,
            "assists": raw_stats.get("assists", 0),
            "acs": float(raw_stats.get("acs", 0)),
            "adr": float(raw_stats.get("adr", 0)),
            "hs_percent": float(raw_stats.get("hs_percent", 0)),
            "kd_ratio": kd_ratio,
            "source": raw_stats.get("source", "unknown"),
        }

    @staticmethod
    def validate(stats: Dict[str, Any]) -> bool:
        """Validate required fields."""
        required = ["map_id", "player", "agent"]
        return all(stats.get(field) for field in required)


class RoundParser:
    """Parse and normalize round-level data."""

    @staticmethod
    def parse(raw_round: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw round data."""
        return {
            "round_id": raw_round.get("round_id"),
            "map_id": raw_round.get("map_id"),
            "round_number": raw_round.get("round_number"),
            "winner": raw_round.get("winner"),
            "win_type": raw_round.get("win_type", "elimination"),
            "spike_planted": bool(raw_round.get("spike_planted", False)),
            "econ_a": raw_round.get("econ_a", 0),
            "econ_b": raw_round.get("econ_b", 0),
            "duration_seconds": raw_round.get("duration_seconds"),
            "source": raw_round.get("source", "unknown"),
        }

    @staticmethod
    def validate(round_data: Dict[str, Any]) -> bool:
        """Validate required fields."""
        required = ["round_id", "map_id", "round_number", "winner"]
        return all(round_data.get(field) for field in required)


class DataPipeline:
    """Process raw data through parsers."""

    @staticmethod
    def process_matches(raw_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of matches."""
        processed = []

        for raw in raw_matches:
            try:
                parsed = MatchParser.parse(raw)
                if MatchParser.validate(parsed):
                    processed.append(parsed)
            except Exception as e:
                logger.error(f"Error processing match: {e}")

        logger.info(f"Processed {len(processed)}/{len(raw_matches)} matches")
        return processed

    @staticmethod
    def process_maps(raw_maps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of maps."""
        processed = []
        failed = []

        for idx, raw in enumerate(raw_maps):
            try:
                parsed = MapParser.parse(raw)
                if MapParser.validate(parsed):
                    processed.append(parsed)
                else:
                    failed.append((idx, raw, "validation failed"))
                    logger.debug(f"Map validation failed: {raw}")
            except Exception as e:
                failed.append((idx, raw, str(e)))
                logger.error(f"Error processing map {idx}: {e}")

        if failed:
            logger.warning(f"Failed to process {len(failed)} maps out of {len(raw_maps)}")
        
        logger.info(f"Processed {len(processed)}/{len(raw_maps)} maps")
        return processed

    @staticmethod
    def process_compositions(raw_comps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of compositions."""
        processed = []
        failed = []

        for idx, raw in enumerate(raw_comps):
            try:
                parsed = CompositionParser.parse(raw)
                if CompositionParser.validate(parsed):
                    processed.append(parsed)
                else:
                    failed.append((idx, raw, "validation failed - incomplete agents"))
                    logger.debug(f"Composition validation failed for map {raw.get('map_id')}: {raw}")
            except Exception as e:
                failed.append((idx, raw, str(e)))
                logger.error(f"Error processing composition: {e}")

        if failed:
            logger.warning(f"Failed to process {len(failed)} compositions out of {len(raw_comps)}")
        
        logger.info(f"Processed {len(processed)}/{len(raw_comps)} compositions")
        return processed

    @staticmethod
    def process_player_stats(raw_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of player stats."""
        processed = []
        failed = []

        for idx, raw in enumerate(raw_stats):
            try:
                parsed = PlayerStatsParser.parse(raw)
                if PlayerStatsParser.validate(parsed):
                    processed.append(parsed)
                else:
                    failed.append((idx, raw, "validation failed"))
                    logger.debug(f"Player stats validation failed: {raw}")
            except Exception as e:
                failed.append((idx, raw, str(e)))
                logger.error(f"Error processing player stats: {e}")

        if failed:
            logger.warning(f"Failed to process {len(failed)} player stats out of {len(raw_stats)}")
        
        logger.info(f"Processed {len(processed)}/{len(raw_stats)} player stats")
        return processed

    @staticmethod
    def process_rounds(raw_rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of rounds."""
        processed = []
        failed = []

        for idx, raw in enumerate(raw_rounds):
            try:
                parsed = RoundParser.parse(raw)
                if RoundParser.validate(parsed):
                    processed.append(parsed)
                else:
                    failed.append((idx, raw, "validation failed"))
                    logger.debug(f"Round validation failed: {raw}")
            except Exception as e:
                failed.append((idx, raw, str(e)))
                logger.error(f"Error processing round: {e}")

        if failed:
            logger.warning(f"Failed to process {len(failed)} rounds out of {len(raw_rounds)}")
        
        logger.info(f"Processed {len(processed)}/{len(raw_rounds)} rounds")
        return processed

    @staticmethod
    def process_all(raw_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Process all data types."""
        return {
            "matches": DataPipeline.process_matches(raw_data.get("matches", [])),
            "maps": DataPipeline.process_maps(raw_data.get("maps", [])),
            "compositions": DataPipeline.process_compositions(
                raw_data.get("compositions", [])
            ),
            "player_stats": DataPipeline.process_player_stats(
                raw_data.get("player_stats", [])
            ),
            "rounds": DataPipeline.process_rounds(raw_data.get("rounds", [])),
        }


if __name__ == "__main__":
    # Test parsers
    sample_match = {
        "match_id": "1",
        "event": "Champions",
        "date": "2025-01-01",
        "patch": "11.05",
        "bo_type": "Bo3",
        "team_a": "prx",
        "team_b": "geng",
    }

    parsed = MatchParser.parse(sample_match)
    print(parsed)
