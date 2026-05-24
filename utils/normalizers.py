"""
Data normalization utilities for Valorant analytics.
Standardizes team names, agent names, map names, and patch versions.
"""

from typing import Dict, Optional
import re


class Normalizers:
    """Centralized normalization for Valorant data."""

    # Official agent names (canonical)
    AGENTS = {
        "jett", "raze", "breach", "omen", "brimstone", "viper", "killjoy",
        "cypher", "sova", "sage", "phoenix", "reyna", "neon", "fade",
        "astra", "kay/o", "chamber", "skye", "yoru", "harbor", "gekko",
        "deadlock", "iso", "clove", "vyse"
    }

    # Official map names
    MAPS = {
        "bind", "haven", "split", "ascent", "icebox", "fracture",
        "sunset", "abyss", "lotus", "pearl"
    }

    # Common patch format: X.XX
    PATCH_PATTERN = re.compile(r"(\d+\.\d+)")

    # Team name aliases and corrections
    TEAM_ALIASES: Dict[str, str] = {
        # International teams
        "paper rex": "Paper Rex",
        "prx": "Paper Rex",
        "sen": "Sentinels",
        "sentinels": "Sentinels",
        "fnatic": "Fnatic",
        "fnc": "Fnatic",
        "loud": "LOUD",
        "lg": "LOUD",
        "nrg": "NRG",
        "nrg esports": "NRG",
        "optic": "OpTic",
        "optc": "OpTic",
        "optic gaming": "OpTic",
        "acend": "Acend",
        "acnd": "Acend",
        "gambit": "Gambit",
        "gmb": "Gambit",
        "version1": "Version1",
        "v1": "Version1",
        "faze": "FaZe",
        "faze clan": "FaZe",
        "cloud9": "Cloud9",
        "c9": "Cloud9",
        "g2": "G2",
        "g2 esports": "G2",
        "kru": "KRÜ",
        "kru esports": "KRÜ",
        "leviatan": "Leviatán",
        "leviatán": "Leviatán",
        ""%k": "%K",
        "misfits": "Misfits",
        "misfits gaming": "Misfits",
        "guild": "GUILD",
        "guild esports": "GUILD",
        "equinox": "Equinox",
        "eqx": "Equinox",
        "trace esports": "Trace",
        "trace": "Trace",
        "xs": "Xtreme Scrims",
        "xtreme scrims": "Xtreme Scrims",
        # Asian teams
        "xavier esports": "Xavier",
        "xavier": "Xavier",
        "zeta division": "Zeta Division",
        "zeta": "Zeta Division",
        "t1": "T1",
        "drx": "DRX",
        "gen.g": "Gen.G",
        "geng": "Gen.G",
        "f4q": "F4Q",
        "f4q esports": "F4Q",
        "bleed esports": "Bleed",
        "bleed": "Bleed",
        "prx": "Paper Rex",
        "rrq": "RRQ Akatsuki",
        "rrq akatsuki": "RRQ Akatsuki",
        # Pacific teams
        "talon": "Talon",
        "talon esports": "Talon",
        "fnatic": "Fnatic",
    }

    @classmethod
    def normalize_team_name(cls, team_name: str) -> Optional[str]:
        """
        Normalize team names to standard format.

        Args:
            team_name: Raw team name from source

        Returns:
            Standardized team name or None if invalid
        """
        if not team_name:
            return None

        # Clean and lowercase
        cleaned = team_name.strip().lower()

        # Check aliases
        if cleaned in cls.TEAM_ALIASES:
            return cls.TEAM_ALIASES[cleaned]

        # Try to find partial matches
        for alias, standard in cls.TEAM_ALIASES.items():
            if alias in cleaned or cleaned in alias:
                return standard

        # Return original with title case if no match
        return team_name.strip()

    @classmethod
    def normalize_agent_name(cls, agent_name: str) -> Optional[str]:
        """
        Normalize agent names to standard format.

        Args:
            agent_name: Raw agent name from source

        Returns:
            Standardized agent name or None if invalid
        """
        if not agent_name:
            return None

        normalized = agent_name.strip().lower()

        # Handle KAY/O special case
        if normalized in ("kay/o", "kayo", "kay o"):
            return "KAY/O"

        # Check against official agents
        if normalized in cls.AGENTS:
            return normalized.title()

        return None

    @classmethod
    def normalize_map_name(cls, map_name: str) -> Optional[str]:
        """
        Normalize map names to standard format.

        Args:
            map_name: Raw map name from source

        Returns:
            Standardized map name or None if invalid
        """
        if not map_name:
            return None

        normalized = map_name.strip().lower()

        if normalized in cls.MAPS:
            return normalized.title()

        return None

    @classmethod
    def normalize_patch(cls, patch_str: str) -> Optional[str]:
        """
        Normalize patch version format (e.g., "11.05" -> "11.05").

        Args:
            patch_str: Raw patch string from source

        Returns:
            Standardized patch version or None if invalid
        """
        if not patch_str:
            return None

        match = cls.PATCH_PATTERN.search(patch_str.strip())
        if match:
            return match.group(1)

        return None

    @classmethod
    def normalize_bo_type(cls, bo_str: str) -> Optional[str]:
        """
        Normalize Best-of format (e.g., "Bo3" -> "bo3").

        Args:
            bo_str: Raw BO format string

        Returns:
            Standardized format (bo1, bo3, bo5, etc.) or None
        """
        if not bo_str:
            return None

        cleaned = bo_str.strip().lower().replace(" ", "")

        # Match BO format
        match = re.search(r"bo(\d)", cleaned)
        if match:
            return f"bo{match.group(1)}"

        return None

    @classmethod
    def extract_team_names(cls, match_title: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract team names from match title (e.g., "Paper Rex vs Gen.G").

        Args:
            match_title: Match title string

        Returns:
            Tuple of (team_a, team_b) normalized names
        """
        if not match_title:
            return None, None

        # Split by common separators
        separators = [" vs ", " v ", " vs. ", " v. "]
        teams = None

        for sep in separators:
            if sep.lower() in match_title.lower():
                parts = match_title.split(sep)
                if len(parts) == 2:
                    teams = parts
                    break

        if not teams:
            return None, None

        team_a = cls.normalize_team_name(teams[0])
        team_b = cls.normalize_team_name(teams[1])

        return team_a, team_b


class EconomyNormalizer:
    """Normalize economy round values."""

    ECONOMY_BRACKETS = {
        "buy": (2000, 3800),
        "half_buy": (1400, 2000),
        "eco": (0, 1400),
        "full_buy": (3800, 16000),
    }

    @classmethod
    def classify_economy(cls, credit_spent: int) -> str:
        """
        Classify economy state based on credits spent.

        Args:
            credit_spent: Total credits spent in round

        Returns:
            Economy classification: 'eco', 'half_buy', 'buy', 'full_buy'
        """
        for eco_type, (min_val, max_val) in cls.ECONOMY_BRACKETS.items():
            if min_val <= credit_spent <= max_val:
                return eco_type

        return "eco" if credit_spent < 1400 else "full_buy"


if __name__ == "__main__":
    # Test normalizers
    print("Agent:", Normalizers.normalize_agent_name("jett"))
    print("Map:", Normalizers.normalize_map_name("bind"))
    print("Patch:", Normalizers.normalize_patch("Episode 8, Act 1"))
    print("Team:", Normalizers.normalize_team_name("prx"))
    print("Teams:", Normalizers.extract_team_names("Paper Rex vs Gen.G"))
