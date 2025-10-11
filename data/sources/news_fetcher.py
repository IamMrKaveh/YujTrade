import asyncio
import json
from typing import Any, Dict, List, Optional
import redis

from config.logger import logger
from .alternativeme_fetcher import AlternativeMeFetcher
from .coindesk_fetcher import CoinDeskFetcher
from .cryptopanic_fetcher import CryptoPanicFetcher
from .messari_fetcher import MessariFetcher


class NewsFetcher:
    def __init__(self, 
                cryptopanic_fetcher: CryptoPanicFetcher,
                alternativeme_fetcher: AlternativeMeFetcher,
                coindesk_fetcher: Optional[CoinDeskFetcher] = None, 
                messari_fetcher: Optional[MessariFetcher] = None, 
                redis_client: Optional[redis.Redis] = None):
        
        self.cryptopanic = cryptopanic_fetcher
        self.alternativeme = alternativeme_fetcher
        self.coindesk = coindesk_fetcher
        self.messari = messari_fetcher
        self.redis = redis_client
        self._is_closed = False

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError("NewsFetcher has been closed and cannot be used")

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        
        tasks = [self.cryptopanic.close(), self.alternativeme.close()]
        if self.coindesk:
            tasks.append(self.coindesk.close())
        if self.messari:
            tasks.append(self.messari.close())
            
        await asyncio.gather(*tasks, return_exceptions=True)

    async def fetch_fear_greed(self, limit: int = 1):
        self._check_if_closed()
        return await self.alternativeme.fetch_fear_greed(limit)

    async def fetch_news(self, currencies: List[str] = ["BTC", "ETH"], kind: Optional[str] = None):
        self._check_if_closed()
        key = "cache:news:" + ",".join(sorted(currencies)) + (f":{kind}" if kind else "")
        if self.redis:
            try:
                cached = await self.redis.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {key}: {e}")

        tasks = [self.cryptopanic.fetch_posts(currencies, kind)]

        if self.coindesk and not kind:
            tasks.append(self.coindesk.get_news(currencies))

        if self.messari and not kind:
            for currency in currencies:
                tasks.append(self.messari.get_news(f"{currency}/USDT"))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_news = []
        for res in results:
            if isinstance(res, list):
                all_news.extend(res)
            elif isinstance(res, Exception):
                logger.warning(f"News fetch failed: {res}")
        
        unique_news = list({item.get('id') or item.get('title'): item for item in all_news}.values())

        if self.redis and unique_news:
            await self.redis.set(key, json.dumps(unique_news), ex=600)

        return unique_news

    def score_news(self, news_items: List[Dict[str, Any]]):
        score = 0
        if not news_items:
            return score
        for it in news_items:
            if 'votes' in it:
                score += int(it['votes'].get('positive', 0))
                score -= int(it['votes'].get('negative', 0))
            
            title = (it.get("title") or "").lower()
            if any(k in title for k in ["bull", "rally", "surge", "gain", "pump", "partnership", "adoption", "breakthrough"]):
                score += 1
            if any(k in title for k in ["crash", "dump", "fall", "drop", "hack", "scam", "exploit", "regulatory", "ban"]):
                score -= 1
        return score

    async def fetch_sentiment_analysis(self, currencies: List[str] = ["BTC", "ETH"]) -> Dict[str, Any]:
        self._check_if_closed()
        cache_key = f"cache:sentiment:{'_'.join(sorted(currencies))}"
        
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            news_items = await self.fetch_news(currencies)
            fear_greed = await self.fetch_fear_greed(limit=1)
            
            sentiment_score = self.score_news(news_items)
            
            positive_count = sum(1 for item in news_items if 'votes' in item and item['votes'].get('positive', 0) > item['votes'].get('negative', 0))
            negative_count = sum(1 for item in news_items if 'votes' in item and item['votes'].get('negative', 0) > item['votes'].get('positive', 0))
            
            sentiment_data = {
                'overall_score': sentiment_score,
                'news_count': len(news_items),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': len(news_items) - positive_count - negative_count,
                'fear_greed_index': int(fear_greed[0]['value']) if fear_greed else None,
                'fear_greed_classification': fear_greed[0]['value_classification'] if fear_greed else None
            }
            
            if self.redis:
                await self.redis.set(cache_key, json.dumps(sentiment_data), ex=1800)
            
            return sentiment_data
        except Exception as e:
            logger.error(f"Error performing sentiment analysis: {e}")
            return {}

    async def fetch_trending_topics(self) -> List[str]:
        self._check_if_closed()
        cache_key = "cache:trending_topics"
        
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            trending_news = await self.cryptopanic.fetch_posts(kind='news')
            
            topics = []
            for item in trending_news[:20]:
                title = item.get('title', '').lower()
                currencies = item.get('currencies', [])
                for currency in currencies:
                    code = currency.get('code')
                    if code and code not in topics:
                        topics.append(code)
            
            if self.redis and topics:
                await self.redis.set(cache_key, json.dumps(topics), ex=3600)
            
            return topics
        except Exception as e:
            logger.error(f"Error fetching trending topics: {e}")
            return []
