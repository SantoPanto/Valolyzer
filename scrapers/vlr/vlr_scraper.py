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
        """Main scraping method."""
        try:
            await self._scrape_events()
            logger.info(f"Found {len(self.events)} events")

            # Bütün etkinlikleri çeker (Dikkat: Çok uzun sürebilir)
            for event in self.events[:100]: 
                await self._scrape_event_matches(event)
                await asyncio.sleep(2)

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
        """Scrape events from vlr.gg/events across multiple pages."""
        try:
            client = await self.http_client
            
            # 1'den 5'e kadar olan etkinlik sayfalarını gezer (Daha fazla veri için 10 yapabilirsiniz)
            for page in range(1, 6):
                page_url = f"{self.EVENTS_URL}/?page={page}"
                logger.info(f"Etkinlikler çekiliyor: Sayfa {page}...")
                
                html = await client.get(page_url)
                if not html:
                    continue

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
                        
                # Siteyi yormamak ve banlanmamak için sayfalar arası çok ufak bir bekleme
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error scraping events: {e}")

    async def _scrape_event_matches(self, event: Dict[str, Any]):
        """Scrape all matches for a specific event."""
        try:
            client = await self.http_client
            html = await client.get(event["url"])
            if not html:
                #logger.warning(f"Failed to fetch event: {event['name']}")
                return

            soup = BeautifulSoup(html, "html.parser")
            # Güncel URL Regex Yakalayıcısı
            match_items = soup.find_all("a", href=re.compile(r"^\/\d+\/"))

            for match_item in match_items:
                match_url = match_item.get("href")
                if match_url:
                    await self._scrape_match_detail(urljoin(self.BASE_URL, match_url), event)
                    # Her maç sayfasına girişte 0.5 saniye bekle (Spam engelleme)
                    await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error scraping event {event['name']}: {e}")

    async def _scrape_match_detail(self, match_url: str, event: Dict[str, Any]):
        """Scrape detailed match information."""
        try:
            client = await self.http_client
            html = await client.get(match_url)
            if not html: return

            soup = BeautifulSoup(html, "html.parser")

            match_data = await self.parse_match_page(soup, event, match_url=match_url)
            if not match_data:
                return # Skip if match data couldn't be parsed

            self.matches.append(match_data)
            map_items = soup.find_all("div", class_="vm-stats-game")

            #if not map_items:
            #    logger.warning(f"Dikkat: {match_url} sayfasında 'vm-stats-game' div'i hiç bulunamadı!")
            #else:
            #    tables = map_items[0].find_all("table", class_="wf-table-batched")
            #    if not tables:
            #        logger.warning(f"Dikkat: 'vm-stats-game' bulundu ama içinde 'wf-table-batched' tablosu yok!")
            
            for idx, map_item in enumerate(map_items):
                game_id = map_item.get("data-game-id")
                if game_id == "all": 
                    continue

                # Extract map name from header
                map_name = self._extract_map_name(map_item) or f"unknown_map_{idx}"
                
                # Extract team scores for this map
                team_a_score, team_b_score = self._extract_map_scores(map_item)

                map_data = {
                    "map_id": f"{match_data['match_id']}_{idx}", 
                    "match_id": match_data["match_id"],
                    "map_name": map_name,
                    "map_order": idx,
                    "team_a_score": team_a_score,
                    "team_b_score": team_b_score,
                    "source": "vlr"
                }
                self.maps.append(map_data)
                logger.debug(f"Extracted map: {map_name} ({team_a_score}-{team_b_score})")

                # Oyuncu istatistiklerini çek
                player_stats = await self.parse_player_stats(map_item, map_data)
                self.player_stats.extend(player_stats)
                
                # Ajan kompozisyonlarını kurtar (assume tables are ordered: team_a, then team_b)
                if match_data and player_stats and len(player_stats) >= 10:
                    # Split player stats by assuming first 5 are team_a, next 5+ are team_b
                    # This is a fallback when team info isn't reliably extracted
                    team_a_stats = [p for p in player_stats if p.get("team") == "team_a"]
                    team_b_stats = [p for p in player_stats if p.get("team") == "team_b"]
                    
                    # If team filtering didn't work, fall back to positional split
                    if not team_a_stats and not team_b_stats:
                        team_a_stats = player_stats[:5]
                        team_b_stats = player_stats[5:10]
                    
                    # Extract agents from team_a
                    if len(team_a_stats) >= 5:
                        agents_a = [p.get("agent") for p in team_a_stats[:5] if p.get("agent")]
                        if len(agents_a) == 5:
                            self.compositions.append({
                                "map_id": map_data["map_id"], 
                                "team": match_data["team_a"], 
                                "agents": agents_a,
                                "agent_1": agents_a[0], 
                                "agent_2": agents_a[1], 
                                "agent_3": agents_a[2], 
                                "agent_4": agents_a[3], 
                                "agent_5": agents_a[4], 
                                "source": "vlr"
                            })
                            logger.debug(f"Extracted composition for {match_data['team_a']}: {agents_a}")
                    
                    # Extract agents from team_b
                    if len(team_b_stats) >= 5:
                        agents_b = [p.get("agent") for p in team_b_stats[:5] if p.get("agent")]
                        if len(agents_b) == 5:
                            self.compositions.append({
                                "map_id": map_data["map_id"], 
                                "team": match_data["team_b"], 
                                "agents": agents_b,
                                "agent_1": agents_b[0], 
                                "agent_2": agents_b[1], 
                                "agent_3": agents_b[2], 
                                "agent_4": agents_b[3], 
                                "agent_5": agents_b[4], 
                                "source": "vlr"
                            })
                            logger.debug(f"Extracted composition for {match_data['team_b']}: {agents_b}")
        except Exception as e:
            logger.error(f"Error scraping match detail: {e}")
        # ... map_items döngüsünün dışına veya hemen öncesine ...
        rounds_extracted = self.parse_rounds(soup, match_data["match_id"])
        if rounds_extracted:
            self.rounds.extend(rounds_extracted)

    def _extract_map_name(self, map_item: BeautifulSoup) -> Optional[str]:
        """VLR'nin güncel HTML yapısına göre harita ismi çeker."""
        try:
            # VLR yeni yapısında harita ismini doğrudan <div class="map"> içinde tutuyor.
            # Veya bazen "vm-stats-game-header" içindeki bir başlıkta.
            
            # 1. Öncelik: <div class="map">
            map_div = map_item.find("div", class_="map")
            if map_div:
                return map_div.get_text(strip=True)
            
            # 2. Öncelik: vm-stats-game-header içindeki span
            header = map_item.find("div", class_="vm-stats-game-header")
            if header:
                name_span = header.find("span", class_="map")
                if name_span:
                    return name_span.get_text(strip=True)

            # 3. Öncelik: Eski yapıları desteklemek için genel bir arama
            # Bu kısım eskisinden daha geniş kapsamlı
            selectors = [
                ("div", {"class": "map"}),
                ("div", {"class": "vm-map-name"}),
                ("span", {"class": "map-name"}),
            ]
            
            for tag, attrs in selectors:
                element = map_item.find(tag, attrs)
                if element:
                    text = element.get_text(strip=True)
                    if text and not text.lower().startswith("unknown"):
                        # Gerçek harita ismini metnin içinden cımbızla al
                        valid_maps = ['ascent', 'bind', 'haven', 'split', 'lotus', 'sunset', 'abyss', 'icebox', 'fracture', 'breeze', 'pearl']
                        for m in valid_maps:
                            if m in text.lower():
                                return m.capitalize()
                        return text
            
            return None
        except Exception as e:
            logger.debug(f"Harita ismi çekilirken hata: {e}")
            return None

    def _extract_map_scores(self, map_item: BeautifulSoup) -> tuple:
        """Extract team scores for a specific map."""
        try:
            team_a_score = 0
            team_b_score = 0
            
            # Look for score display in map header
            score_container = map_item.find(re.compile("^div$"), class_=re.compile("score|result"))
            if score_container:
                scores_text = score_container.get_text(strip=True)
                # Try to extract X-Y pattern
                match = re.search(r"(\d+)\s*[-–]\s*(\d+)", scores_text)
                if match:
                    team_a_score = int(match.group(1))
                    team_b_score = int(match.group(2))
                    return team_a_score, team_b_score
            
            # Alternative: look in table header rows
            tables = map_item.find_all("table")
            for table in tables:
                # Check for score row in table
                score_cells = table.find_all("td", class_=re.compile("score|result"))
                if len(score_cells) >= 2:
                    try:
                        team_a_score = int(''.join(filter(str.isdigit, score_cells[0].get_text())))
                        team_b_score = int(''.join(filter(str.isdigit, score_cells[1].get_text())))
                        return team_a_score, team_b_score
                    except (ValueError, IndexError):
                        continue
            
            logger.debug(f"Could not extract scores, using defaults: {team_a_score}-{team_b_score}")
            return team_a_score, team_b_score
        except Exception as e:
            logger.debug(f"Error extracting map scores: {e}")
            return 0, 0

    # --- ABSTRACT CLASS YER TUTUCULARI ---
    async def parse_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw match data."""
        return match_data

    async def parse_map(self, map_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw map data."""
        return map_data

    # --- GÜNCELLENMİŞ DOM PARSER'LAR ---
    async def parse_match_page(self, soup: BeautifulSoup, event: Dict[str, Any], match_url: str = "") -> Optional[Dict[str, Any]]:
        """Parse match page to extract match-level data with current VLR classes."""
        try:
            teams_section = soup.find("div", class_="match-header-vs")
            if not teams_section:
                return None

            team_elements = teams_section.find_all("div", class_="wf-title-med")
            if len(team_elements) < 2:
                return None

            team_a = Normalizers.normalize_team_name(team_elements[0].get_text(strip=True))
            team_b = Normalizers.normalize_team_name(team_elements[1].get_text(strip=True))

            if team_a.lower() in ["tbd", "tba"] or team_b.lower() in ["tbd", "tba"]:
                return None

            score_a, score_b = 0, 0
            score_section = teams_section.find("div", class_="js-spoiler")
            if score_section:
                scores = score_section.get_text(strip=True).split(":")
                if len(scores) == 2:
                    score_a = int(''.join(filter(str.isdigit, scores[0])) or 0)
                    score_b = int(''.join(filter(str.isdigit, scores[1])) or 0)

            # match_id'yi DOĞRUDAN gelen URL üzerinden çekmek
            match_id = None
            if match_url:
                match_id_match = re.search(r"vlr\.gg/(\d+)/", match_url)
                if match_id_match:
                    match_id = match_id_match.group(1)

            # Fallback artık sadece URL tamamen bozuksa devreye girer
            final_match_id = match_id or f"{event.get('source_id', 'unknown')}_{team_a}_{team_b}_{int(datetime.now().timestamp())}"

            patch = event.get("name", "").split()[-1] if event.get("name") else None

            return {
                "match_id": final_match_id,
                "event": event["name"],
                "date": datetime.now(),
                "patch": Normalizers.normalize_patch(patch) if patch else None,
                "bo_type": "bo3",
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
    async def parse_player_stats(self, map_section: BeautifulSoup, map_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse player statistics with team tracking."""
        player_stats = []
        try:
            tables = map_section.find_all("table")

            if not tables:
                logger.debug(f"Map id {map_data.get('map_id')} için hiç HTML table bulunamadı (JavaScript ile yükleniyor olabilir).")
                return player_stats
            
            for table_idx, table in enumerate(tables):
                # Try to determine which team this table is for
                # Typically: first table is team_a, second table is team_b
                team_name = None
                if table_idx == 0:
                    # Look for team name before this table
                    prev_elem = table.find_previous(re.compile("^h[2-4]$|^div$"))
                    if prev_elem:
                        team_text = prev_elem.get_text(strip=True).lower()
                        # Try to match against team aliases/names
                        for alias in ["team_a", "team_b", "team-a", "team-b"]:
                            if alias in team_text:
                                team_name = "team_a"
                                break
                elif table_idx == 1:
                    team_name = "team_b"
                
                rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 7:  # Need at least 7 cells for K D A columns
                        continue
                        
                    # Player name
                    player_name = cells[0].get_text(strip=True).split()[0] if cells[0].get_text(strip=True) else None
                    if not player_name:
                        continue
                    
                    # Agent name
                    agent = ""
                    agent_img = cells[1].find("img")
                    if agent_img and agent_img.get("title"):
                        agent = Normalizers.normalize_agent_name(agent_img.get("title", ""))
                    
                    try:
                        # Parse stats with fallback defaults
                        acs_text = ''.join(filter(lambda x: x.isdigit() or x == '.', cells[3].get_text(strip=True)))
                        acs = float(acs_text) if acs_text else 0.0
                        
                        kills_text = ''.join(filter(str.isdigit, cells[4].get_text(strip=True)))
                        kills = int(kills_text) if kills_text else 0
                        
                        deaths_text = ''.join(filter(str.isdigit, cells[5].get_text(strip=True)))
                        deaths = int(deaths_text) if deaths_text else 0
                        
                        assists_text = ''.join(filter(str.isdigit, cells[6].get_text(strip=True)))
                        assists = int(assists_text) if assists_text else 0

                        player_stats.append({
                            "map_id": map_data["map_id"],
                            "player": player_name,
                            "team": team_name,
                            "agent": agent,
                            "kills": kills,
                            "deaths": deaths,
                            "assists": assists,
                            "acs": acs,
                            "adr": 0.0,
                            "hs_percent": 0.0,
                            "source": "vlr",
                        })
                    except Exception as parse_error:
                        logger.debug(f"Error parsing player stats row: {parse_error}")
                        continue
        except Exception as e:
            logger.error(f"Error parsing player stats: {e}")
        return player_stats
    
    def parse_rounds(self, soup: BeautifulSoup, match_id: str) -> List[Dict[str, Any]]:
        """Parse VLR round timeline data dynamically and format for validation."""
        rounds_data = []
        try:
            # Harita harita geziyoruz ki map_id'yi doğru eşleştirelim
            map_sections = soup.find_all("div", class_="vm-stats-game")
            
            for map_index, game in enumerate(map_sections):
                game_id = game.get("data-game-id")
                if game_id == "all": 
                    continue
                
                # Validation için zorunlu olan map_id'yi oluşturuyoruz
                map_id = f"{match_id}_{map_index}"
                
                # Bu haritanın içindeki rauntları esnek seçici ile buluyoruz
                round_elements = game.find_all(
                    "div", 
                    class_=lambda c: c and ("vlr-round-info" in c or "rnd-sq" in c or "vlr-rounds" in c)
                )
                
                for rnd in round_elements:
                    round_number = 0
                    winner = "unknown"
                    win_type = "elimination"
                    
                    # 1. Raunt Numarası
                    num_el = rnd.find("div", class_=lambda c: c and "rnd-num" in c)
                    if num_el:
                        try:
                            round_number = int(num_el.get_text(strip=True))
                        except ValueError:
                            pass
                            
                    # 2. Kazanan Takım (VLR, 'mod-t' yani Attack ve 'mod-ct' yani Defense class'larını kullanır)
                    classes = rnd.get("class", [])
                    if any("mod-t" in c for c in classes):
                        winner = "attack"
                    elif any("mod-ct" in c for c in classes):
                        winner = "defense"
                    else:
                        winner = "draw/unknown"
                        
                    # 3. Kazanma Şekli (Spike, Defuse, Elimination)
                    icon = rnd.find("i") or rnd.find("img")
                    if icon:
                        icon_classes = icon.get("class", [])
                        src = icon.get("src", "")
                        if any("spike" in c for c in icon_classes) or "boom" in src:
                            win_type = "spike"
                        elif any("defuse" in c for c in icon_classes) or "defuse" in src:
                            win_type = "defuse"
                        elif any("time" in c for c in icon_classes) or "clock" in src:
                            win_type = "time"

                    # VALIDATION İÇİN GEREKEN KUSURSUZ SÖZLÜK YAPISI
                    rounds_data.append({
                        "round_id": f"{map_id}_{round_number}",
                        "map_id": map_id,
                        "round_number": round_number,
                        "winner": winner,
                        "win_type": win_type,
                        "spike_planted": win_type in ["spike", "defuse"], # Spike patladıysa veya çözüldüyse kurulmuş demektir
                        "source": "vlr"
                    })
                    
        except Exception as e:
            logger.error(f"Failed parsing rounds: {e}")
            
        return rounds_data

async def main():
    """Test VLR scraper."""
    scraper = VLRScraper(rate_limit=1.0)
    try:
        stats = await scraper.run()
        print(f"Scraping stats: {stats}")
        export_stats = scraper.export_to_csv()
        print(f"Export stats: {export_stats}")
    finally:
        await scraper.cleanup()

if __name__ == "__main__":
    asyncio.run(main())