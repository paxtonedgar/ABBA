#!/usr/bin/env python3
"""
Real Historical Odds Fetcher
Fetches real historical odds data using The Odds API.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
import structlog
import yaml

logger = structlog.get_logger()


@dataclass
class HistoricalOdds:
    game_id: str
    date: datetime
    home_team: str
    away_team: str
    bookmaker: str
    home_odds: float
    away_odds: float
    home_implied_prob: float
    away_implied_prob: float
    line_movement: float | None = None
    closing_line: float | None = None


class RealOddsFetcher:
    """Fetches real historical odds data from The Odds API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"

        # Create data directory
        self.data_dir = Path("real_data")
        self.data_dir.mkdir(exist_ok=True)

        # MLB team name mappings for API
        self.team_mappings = {
            'New York Yankees': 'New York Yankees',
            'Boston Red Sox': 'Boston Red Sox',
            'Tampa Bay Rays': 'Tampa Bay Rays',
            'Baltimore Orioles': 'Baltimore Orioles',
            'Toronto Blue Jays': 'Toronto Blue Jays',
            'Chicago White Sox': 'Chicago White Sox',
            'Cleveland Guardians': 'Cleveland Guardians',
            'Detroit Tigers': 'Detroit Tigers',
            'Kansas City Royals': 'Kansas City Royals',
            'Minnesota Twins': 'Minnesota Twins',
            'Houston Astros': 'Houston Astros',
            'Los Angeles Angels': 'Los Angeles Angels',
            'Oakland Athletics': 'Oakland Athletics',
            'Seattle Mariners': 'Seattle Mariners',
            'Texas Rangers': 'Texas Rangers',
            'Atlanta Braves': 'Atlanta Braves',
            'Miami Marlins': 'Miami Marlins',
            'New York Mets': 'New York Mets',
            'Philadelphia Phillies': 'Philadelphia Phillies',
            'Washington Nationals': 'Washington Nationals',
            'Chicago Cubs': 'Chicago Cubs',
            'Cincinnati Reds': 'Cincinnati Reds',
            'Milwaukee Brewers': 'Milwaukee Brewers',
            'Pittsburgh Pirates': 'Pittsburgh Pirates',
            'St. Louis Cardinals': 'St. Louis Cardinals',
            'Arizona Diamondbacks': 'Arizona Diamondbacks',
            'Colorado Rockies': 'Colorado Rockies',
            'Los Angeles Dodgers': 'Los Angeles Dodgers',
            'San Diego Padres': 'San Diego Padres',
            'San Francisco Giants': 'San Francisco Giants'
        }

        logger.info("Real Odds Fetcher initialized")

    def american_to_probability(self, american_odds: float) -> float:
        """Convert American odds to probability."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    async def fetch_historical_odds(self, start_date: str, end_date: str) -> list[HistoricalOdds]:
        """Fetch real historical odds data from The Odds API."""
        logger.info(f"💰 Fetching real historical odds from {start_date} to {end_date}")

        odds_data = []

        try:
            async with aiohttp.ClientSession() as session:
                current_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')

                while current_date <= end_dt:
                    try:
                        logger.info(f"Fetching odds for {current_date.strftime('%Y-%m-%d')}")

                        url = f"{self.base_url}/sports/baseball_mlb/odds-history"
                        params = {
                            'apiKey': self.api_key,
                            'regions': 'us',
                            'markets': 'h2h',  # moneyline
                            'date': current_date.strftime('%Y-%m-%d'),
                            'bookmakers': 'draftkings,fanduel,pinnacle,betmgm,caesars'
                        }

                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()

                                for game in data:
                                    try:
                                        home_team = game['home_team']
                                        away_team = game['away_team']
                                        game_id = game.get('id', f"{current_date.strftime('%Y%m%d')}_{home_team}_{away_team}")

                                        # Process bookmaker odds
                                        for bookmaker in game.get('bookmakers', []):
                                            bookmaker_name = bookmaker['title']

                                            for market in bookmaker.get('markets', []):
                                                if market['key'] == 'h2h':
                                                    home_odds = None
                                                    away_odds = None

                                                    for outcome in market['outcomes']:
                                                        if outcome['name'] == home_team:
                                                            home_odds = outcome['price']
                                                        elif outcome['name'] == away_team:
                                                            away_odds = outcome['price']

                                                    if home_odds is not None and away_odds is not None:
                                                        home_implied = self.american_to_probability(home_odds)
                                                        away_implied = self.american_to_probability(away_odds)

                                                        odds = HistoricalOdds(
                                                            game_id=game_id,
                                                            date=current_date,
                                                            home_team=home_team,
                                                            away_team=away_team,
                                                            bookmaker=bookmaker_name,
                                                            home_odds=home_odds,
                                                            away_odds=away_odds,
                                                            home_implied_prob=home_implied,
                                                            away_implied_prob=away_implied
                                                        )
                                                        odds_data.append(odds)

                                    except Exception as e:
                                        logger.error(f"Error processing game: {e}")
                                        continue

                                logger.info(f"✅ Fetched {len(data)} games for {current_date.strftime('%Y-%m-%d')}")

                            elif response.status == 429:
                                logger.warning("Rate limit hit, waiting 60 seconds...")
                                await asyncio.sleep(60)
                                continue
                            elif response.status == 401:
                                logger.error("Invalid API key")
                                return []
                            else:
                                logger.error(f"API error: {response.status}")
                                logger.error(f"Response: {await response.text()}")

                        # Rate limiting - be conservative
                        await asyncio.sleep(1)  # 1 second between requests
                        current_date += timedelta(days=1)

                    except Exception as e:
                        logger.error(f"Error fetching odds for {current_date}: {e}")
                        current_date += timedelta(days=1)
                        continue

            logger.info(f"✅ Successfully fetched {len(odds_data)} historical odds records")
            return odds_data

        except Exception as e:
            logger.error(f"❌ Error fetching historical odds: {e}")
            return []

    async def fetch_odds_for_specific_dates(self, dates: list[str]) -> list[HistoricalOdds]:
        """Fetch odds for specific dates."""
        logger.info(f"💰 Fetching odds for {len(dates)} specific dates")

        all_odds = []

        try:
            async with aiohttp.ClientSession() as session:
                for date_str in dates:
                    try:
                        logger.info(f"Fetching odds for {date_str}")

                        url = f"{self.base_url}/sports/baseball_mlb/odds-history"
                        params = {
                            'apiKey': self.api_key,
                            'regions': 'us',
                            'markets': 'h2h',
                            'date': date_str,
                            'bookmakers': 'draftkings,fanduel,pinnacle,betmgm,caesars'
                        }

                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()

                                for game in data:
                                    try:
                                        home_team = game['home_team']
                                        away_team = game['away_team']
                                        game_id = game.get('id', f"{date_str.replace('-', '')}_{home_team}_{away_team}")

                                        for bookmaker in game.get('bookmakers', []):
                                            bookmaker_name = bookmaker['title']

                                            for market in bookmaker.get('markets', []):
                                                if market['key'] == 'h2h':
                                                    home_odds = None
                                                    away_odds = None

                                                    for outcome in market['outcomes']:
                                                        if outcome['name'] == home_team:
                                                            home_odds = outcome['price']
                                                        elif outcome['name'] == away_team:
                                                            away_odds = outcome['price']

                                                    if home_odds is not None and away_odds is not None:
                                                        home_implied = self.american_to_probability(home_odds)
                                                        away_implied = self.american_to_probability(away_odds)

                                                        odds = HistoricalOdds(
                                                            game_id=game_id,
                                                            date=datetime.strptime(date_str, '%Y-%m-%d'),
                                                            home_team=home_team,
                                                            away_team=away_team,
                                                            bookmaker=bookmaker_name,
                                                            home_odds=home_odds,
                                                            away_odds=away_odds,
                                                            home_implied_prob=home_implied,
                                                            away_implied_prob=away_implied
                                                        )
                                                        all_odds.append(odds)

                                    except Exception as e:
                                        logger.error(f"Error processing game: {e}")
                                        continue

                                logger.info(f"✅ Fetched {len(data)} games for {date_str}")

                            elif response.status == 429:
                                logger.warning("Rate limit hit, waiting 60 seconds...")
                                await asyncio.sleep(60)
                                continue
                            else:
                                logger.error(f"API error for {date_str}: {response.status}")

                        # Rate limiting
                        await asyncio.sleep(1)

                    except Exception as e:
                        logger.error(f"Error fetching odds for {date_str}: {e}")
                        continue

            logger.info(f"✅ Successfully fetched {len(all_odds)} odds records for specific dates")
            return all_odds

        except Exception as e:
            logger.error(f"❌ Error fetching odds for specific dates: {e}")
            return []

    def save_odds_data(self, odds_data: list[HistoricalOdds]):
        """Save odds data to CSV file."""
        try:
            if not odds_data:
                logger.warning("No odds data to save")
                return

            # Convert to DataFrame
            odds_df = pd.DataFrame([
                {
                    'game_id': o.game_id,
                    'date': o.date,
                    'home_team': o.home_team,
                    'away_team': o.away_team,
                    'bookmaker': o.bookmaker,
                    'home_odds': o.home_odds,
                    'away_odds': o.away_odds,
                    'home_implied_prob': o.home_implied_prob,
                    'away_implied_prob': o.away_implied_prob
                }
                for o in odds_data
            ])

            # Save to CSV
            output_file = self.data_dir / "real_historical_odds_2024.csv"
            odds_df.to_csv(output_file, index=False)

            logger.info(f"✅ Saved {len(odds_data)} odds records to {output_file}")

            # Print summary
            print("\n📊 Odds Data Summary:")
            print(f"Total Records: {len(odds_data)}")
            print(f"Date Range: {odds_df['date'].min()} to {odds_df['date'].max()}")
            print(f"Bookmakers: {odds_df['bookmaker'].nunique()}")
            print(f"Games: {odds_df['game_id'].nunique()}")

        except Exception as e:
            logger.error(f"Error saving odds data: {e}")


async def main():
    """Main function to fetch real historical odds."""
    # Load configuration
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Get API key
    api_key = config.get('apis', {}).get('the_odds_api_key')

    if not api_key or api_key == "your_odds_api_key_here":
        logger.error("No valid Odds API key found in config")
        return

    # Initialize fetcher
    fetcher = RealOddsFetcher(api_key)

    # Fetch historical odds for 2024 MLB season
    logger.info("🚀 Starting real historical odds collection")

    # MLB 2024 season dates (approximate)
    start_date = "2024-03-28"  # Opening Day
    end_date = "2024-09-30"    # Regular season end

    # Fetch odds
    odds_data = await fetcher.fetch_historical_odds(start_date, end_date)

    # Save data
    fetcher.save_odds_data(odds_data)

    print("\n✅ Real historical odds collection completed!")
    print("Results saved to: real_data/real_historical_odds_2024.csv")


if __name__ == "__main__":
    asyncio.run(main())
