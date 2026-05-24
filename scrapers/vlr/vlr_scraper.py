"""
VLR.gg scraper - Primary source for professional Valorant match data.
Scrapes events, matches, maps, and player statistics.
"""

import re
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from scrapers.base import BaseScraper
from utils.normalizers import Normalizers
from utils.logging import get_logger

logger = get_logger(__name__)


class VLRScraper(BaseScraper):
    """Scraper for vlr.gg - Primary Valorant esports data source."""

    BASE_URL = "https://www.vlr.gg"
    EVENTS_URL = f"{BASE_URL}/events"

    def __init__(self, *args, **kwargs):
        """Initialize VLR scraper."""
        super().__init__("vlr_scraper", *args, **kwargs)
        self.events: List[Dict[str, Any]] = []

    async def scrape(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Main scraping method.

        Returns:
            Dictionary with all scraped data
        """
        try:
            # Scrape events first
            await self._scrape_events()
            logger.info(f"Found {len(self.events)} events")

            # For each event, scrape matches
            for event in self.events[:5]:  # Limit to recent 5 events for testing
                await self._scrape_event_matches(event)

            return {
                "matches": self.matches,
                "maps": self.maps,
                "compositions": self.compositions,
                "player_stats": self.player_stats,
                "rounds": self.rounds,
            }

        except Exception as e:
            logger.error(f"Error in VLR scraper: {e}", exc_info=True)
            raise

    async def _scrape_events(self):
        """Scrape events from vlr.gg/events."""
        try:
            client = await self.http_client
            html = await client.get(self.EVENTS_URL)

            if not html:
                logger.warning("Failed to fetch events page")
                return

            soup = BeautifulSoup(html, "html.parser")
            event_items = soup.find_all("a", class_="event-item")

            for item in event_items:
                event_url = item.get("href")
                event_name = item.find("div", class_="event-item-title")
                event_date = item.find("div", class_="event-item-date")

                if event_url and event_name:
                    self.events.append({
                        "name": event_name.get_text(strip=True),
                        "url": urljoin(self.BASE_URL, event_url),
                        "date": event_date.get_text(strip=True) if event_date else None,
                        "source_id": event_url.split("/")[-1],
                    })

            logger.info(f"Scraped {len(self.events)} events")

        except Exception as e:
            logger.error(f"Error scraping events: {e}")

    async def _scrape_event_matches(self, event: Dict[str, Any]):
        """Scrape all matches for a specific event."""
        try:
            client = await self.http_client
            html = await client.get(event["url"])

            if not html:
                logger.warning(f"Failed to fetch event: {event['name']}")
                return

            soup = BeautifulSoup(html, "html.parser")
            match_items = soup.find_all("a", class_="match-item")

            for match_item in match_items:
                match_url = match_item.get("href")
                if match_url:
                    await self._scrape_match_detail(
                        urljoin(self.BASE_URL, match_url),
                        event
                    )

            logger.info(f"Scraped matches for event: {event['name']}")

        except Exception as e:
            logger.error(f"Error scraping event {event['name']}: {e}")

    async def _scrape_match_detail(self, match_url: str, event: Dict[str, Any]):
        """Scrape detailed match information."""
        try:
            client = await self.http_client
            html = await client.get(match_url)

            if not html:
                return

            soup = BeautifulSoup(html, "html.parser")

            # Extract match information
            match_data = await self.parse_match_page(soup, event)
            if match_data:
                self.matches.append(match_data)

            # Extract map information
            map_items = soup.find_all("div", class_="map-results")
            for idx, map_item in enumerate(map_items):
                map_data = await self.parse_map_page(map_item, match_data, idx + 1)
                if map_data:
                    self.maps.append(map_data)

                # Extract player stats for map
                player_stats = await self.parse_player_stats(map_item, map_data)
                self.player_stats.extend(player_stats)

                # Extract compositions for map
                comps = await self.parse_compositions(map_item, map_data)
                self.compositions.extend(comps)

        except Exception as e:
            logger.error(f"Error scraping match detail: {e}")

    async def parse_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw match data."""
        # This is called when processing stored data
        return match_data

    async def parse_map(self, map_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw map data."""
        return map_data

    async def parse_match_page(self, soup: BeautifulSoup, 
                               event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse match page to extract match-level data."""
        try:
            # Extract teams
            teams_section = soup.find("div", class_="match-header")
            if not teams_section:
                return None

            team_elements = teams_section.find_all("div", class_="team-name")
            if len(team_elements) < 2:
                return None

            team_a = team_elements[0].get_text(strip=True)
            team_b = team_elements[1].get_text(strip=True)

            # Normalize team names
            team_a = Normalizers.normalize_team_name(team_a)
            team_b = Normalizers.normalize_team_name(team_b)

            # Extract score
            score_section = soup.find("div", class_="match-score")
            score_a, score_b = 0, 0
            if score_section:
                scores = score_section.get_text(strip=True).split("-")
                if len(scores) == 2:
                    score_a = int(scores[0].strip())
                    score_b = int(scores[1].strip())

            # Extract match ID from URL
            match_id = re.search(r"/match/(\d+)", soup.find("html").prettify())
            match_id = match_id.group(1) if match_id else None

            # Extract date and patch
            date_str = soup.find("div", class_="match-date")
            patch = event.get("name", "").split()[-1] if event.get("name") else None

            return {
                "match_id": match_id or f"{event['source_id']}_{team_a}_{team_b}",
                "event": event["name"],
                "date": datetime.now(),  # Would parse from page
                "patch": Normalizers.normalize_patch(patch) if patch else None,
                "bo_type": "bo3",  # Default, would detect from page
                "team_a": team_a,
                "team_b": team_b,
                "winner": team_a if score_a > score_b else (team_b if score_b > score_a else None),
                "score_a": score_a,
                "score_b": score_b,
                "maps_played": score_a + score_b,
                "source": "vlr",
            }

        except Exception as e:
            logger.error(f"Error parsing match page: {e}")
            return None

    async def parse_map_page(self, map_section: BeautifulSoup, 
                            match_data: Dict[str, Any],
                            map_order: int) -> Optional[Dict[str, Any]]:
        """Parse map section to extract map-level data."""
        try:
            map_name_el = map_section.find("div", class_="map-name")
            if not map_name_el:
                return None

            map_name = map_name_el.get_text(strip=True)
            map_name = Normalizers.normalize_map_name(map_name)

            # Extract scores
            scores = map_section.find_all("div", class_="score")
            score_a, score_b = 0, 0
            if len(scores) >= 2:
                score_a = int(scores[0].get_text(strip=True))
                score_b = int(scores[1].get_text(strip=True))

            return {
                "map_id": f"{match_data['match_id']}_{map_order}",
                "match_id": match_data["match_id"],
                "map_name": map_name,
                "map_order": map_order,
                "team_a_score": score_a,
                "team_b_score": score_b,
                "attacker_start": "team_a",  # Would detect from page
                "source": "vlr",
            }

        except Exception as e:
            logger.error(f"Error parsing map page: {e}")
            return None

    async def parse_player_stats(self, map_section: BeautifulSoup,
                                map_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse player statistics from map section."""
        player_stats = []

        try:
            stat_rows = map_section.find_all("tr", class_="player-stat-row")

            for row in stat_rows:
                cells = row.find_all("td")
                if len(cells) < 7:
                    continue

                player_name = cells[0].get_text(strip=True)
                agent = cells[1].get_text(strip=True)
                agent = Normalizers.normalize_agent_name(agent)

                if not agent:
                    continue

                try:
                    kills = int(cells[2].get_text(strip=True))
                    deaths = int(cells[3].get_text(strip=True))
                    assists = int(cells[4].get_text(strip=True))
                    acs = float(cells[5].get_text(strip=True))
                    adr = float(cells[6].get_text(strip=True))

                    hs_percent = 0.0
                    if len(cells) > 7:
                        hs_str = cells[7].get_text(strip=True).rstrip("%")
                        hs_percent = float(hs_str) if hs_str else 0.0

                    player_stats.append({
                        "map_id": map_data["map_id"],
                        "player": player_name,
                        "team": None,  # Would determine from page context
                        "agent": agent,
                        "kills": kills,
                        "deaths": deaths,
                        "assists": assists,
                        "acs": acs,
                        "adr": adr,
                        "hs_percent": hs_percent,
                        "source": "vlr",
                    })

                except (ValueError, IndexError):
                    continue

        except Exception as e:
            logger.error(f"Error parsing player stats: {e}")

        return player_stats

    async def parse_compositions(self, map_section: BeautifulSoup,
                                map_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse agent compositions from map section."""
        compositions = []

        try:
            # Find team sections
            team_sections = map_section.find_all("div", class_="team-comp")

            for team_idx, team_section in enumerate(team_sections):
                agents = []
                agent_els = team_section.find_all("div", class_="agent-icon")

                for agent_el in agent_els:
                    agent_name = agent_el.get("title", "").strip()
                    agent_name = Normalizers.normalize_agent_name(agent_name)

                    if agent_name:
                        agents.append(agent_name)

                if len(agents) == 5:
                    team = map_data["team_a"] if team_idx == 0 else map_data["team_b"]
                    compositions.append({
                        "map_id": map_data["map_id"],
                        "team": team,
                        "agents": agents,
                        "source": "vlr",
                    })

        except Exception as e:
            logger.error(f"Error parsing compositions: {e}")

        return compositions


async def main():
    """Test VLR scraper."""
    scraper = VLRScraper(rate_limit=1.0)

    try:
        stats = await scraper.run()
        print(f"Scraping stats: {stats}")

        # Export data
        export_stats = scraper.export_to_csv()
        print(f"Export stats: {export_stats}")

    finally:
        await scraper.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
