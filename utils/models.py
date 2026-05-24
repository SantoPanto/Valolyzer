"""
Pydantic models for Valorant data structures.
Provides validated, type-safe data containers for scraping and storage.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class BestOfType(str, Enum):
    """Best-of match formats."""
    BO1 = "bo1"
    BO3 = "bo3"
    BO5 = "bo5"


class WinType(str, Enum):
    """Round win types."""
    ELIMINATION = "elimination"
    SPIKE_DEFUSED = "spike_defused"
    SPIKE_DETONATED = "spike_detonated"
    TIMEOUT = "timeout"


class Match(BaseModel):
    """Match-level data model."""
    match_id: str
    event: str
    date: datetime
    patch: Optional[str] = None
    bo_type: Optional[BestOfType] = None
    team_a: str
    team_b: str
    winner: Optional[str] = None
    score_a: Optional[int] = None
    score_b: Optional[int] = None
    maps_played: Optional[int] = None
    source: str = "vlr"

    class Config:
        json_schema_extra = {
            "example": {
                "match_id": "542195",
                "event": "Valorant Champions 2025",
                "date": "2025-09-12T09:00:00",
                "patch": "11.05",
                "bo_type": "bo3",
                "team_a": "Paper Rex",
                "team_b": "Gen.G",
                "winner": "Paper Rex",
                "score_a": 2,
                "score_b": 0,
                "maps_played": 2,
                "source": "vlr"
            }
        }


class Map(BaseModel):
    """Map-level data model."""
    map_id: str
    match_id: str
    map_name: str
    map_order: int
    team_a_score: int
    team_b_score: int
    attacker_start: Optional[str] = None  # "team_a" or "team_b"
    duration_seconds: Optional[int] = None
    mvp: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "map_id": "542195_1",
                "match_id": "542195",
                "map_name": "Bind",
                "map_order": 1,
                "team_a_score": 13,
                "team_b_score": 5,
                "attacker_start": "team_a",
                "duration_seconds": 1850,
                "mvp": "jinggg"
            }
        }


class Composition(BaseModel):
    """Agent composition for a map and team."""
    map_id: str
    team: str
    agents: List[str] = Field(min_length=5, max_length=5)
    
    @field_validator('agents')
    @classmethod
    def validate_agents(cls, v):
        """Ensure exactly 5 unique agents."""
        if len(set(v)) != 5:
            raise ValueError('Must have exactly 5 unique agents')
        return v

    @property
    def agent_1(self) -> str:
        return self.agents[0] if len(self.agents) > 0 else None

    @property
    def agent_2(self) -> str:
        return self.agents[1] if len(self.agents) > 1 else None

    @property
    def agent_3(self) -> str:
        return self.agents[2] if len(self.agents) > 2 else None

    @property
    def agent_4(self) -> str:
        return self.agents[3] if len(self.agents) > 3 else None

    @property
    def agent_5(self) -> str:
        return self.agents[4] if len(self.agents) > 4 else None

    class Config:
        json_schema_extra = {
            "example": {
                "map_id": "542195_1",
                "team": "Paper Rex",
                "agents": ["Jett", "Omen", "Sage", "Sova", "Reyna"]
            }
        }


class PlayerStats(BaseModel):
    """Player-level statistics for a map."""
    map_id: str
    player: str
    team: str
    agent: str
    kills: int
    deaths: int
    assists: int
    acs: float  # Average Combat Score
    adr: float  # Average Damage per Round
    hs_percent: float  # Headshot percentage
    kd_ratio: Optional[float] = None

    @field_validator('hs_percent')
    @classmethod
    def validate_headshot_percent(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Headshot percentage must be between 0-100')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "map_id": "542195_1",
                "player": "jinggg",
                "team": "Paper Rex",
                "agent": "Jett",
                "kills": 21,
                "deaths": 5,
                "assists": 3,
                "acs": 314.5,
                "adr": 189.2,
                "hs_percent": 45.2,
                "kd_ratio": 4.2
            }
        }


class Round(BaseModel):
    """Round-level data."""
    round_id: str
    map_id: str
    round_number: int
    winner: str
    win_type: WinType
    spike_planted: bool = False
    econ_a: int  # Economy spent
    econ_b: int
    duration_seconds: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "round_id": "542195_1_1",
                "map_id": "542195_1",
                "round_number": 1,
                "winner": "team_a",
                "win_type": "elimination",
                "spike_planted": False,
                "econ_a": 3500,
                "econ_b": 3600,
                "duration_seconds": 95
            }
        }


class PickBan(BaseModel):
    """Pick/Ban information from a match."""
    match_id: str
    team: str
    action: str  # "pick" or "ban"
    map_name: Optional[str] = None
    order: int

    class Config:
        json_schema_extra = {
            "example": {
                "match_id": "542195",
                "team": "Paper Rex",
                "action": "ban",
                "map_name": "Abyss",
                "order": 1
            }
        }


class ScraperState(BaseModel):
    """Track scraper progress for resumable scraping."""
    scraper_name: str
    last_scraped_at: datetime
    last_match_id: Optional[str] = None
    total_matches: int = 0
    total_errors: int = 0
    status: str = "active"  # active, paused, error

    class Config:
        json_schema_extra = {
            "example": {
                "scraper_name": "vlr_event_scraper",
                "last_scraped_at": "2025-09-15T10:30:00",
                "last_match_id": "542195",
                "total_matches": 150,
                "total_errors": 2,
                "status": "active"
            }
        }


if __name__ == "__main__":
    # Test model creation
    match = Match(
        match_id="542195",
        event="Valorant Champions 2025",
        date=datetime.now(),
        patch="11.05",
        bo_type=BestOfType.BO3,
        team_a="Paper Rex",
        team_b="Gen.G",
        winner="Paper Rex",
        score_a=2,
        score_b=0
    )
    print(match.model_dump_json(indent=2))
