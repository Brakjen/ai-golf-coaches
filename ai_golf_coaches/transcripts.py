"""Transcript retrieval and management with rate limiting.

Provides functionality for:
- Fetching video transcripts from YouTube with proxy support
- Rate limiting to avoid burning through proxy quotas
- Parallel processing with controlled concurrency
- Caching and local storage of transcripts
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api.proxies import WebshareProxyConfig

from ai_golf_coaches.config import get_settings
from ai_golf_coaches.models import (
    RecordStatus,
    TranscriptDoc,
    TranscriptLine,
    VideoMeta,
    VideoRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting network operations.

    Attributes:
        min_delay (float): Minimum delay between requests in seconds.
        max_delay (float): Maximum delay between requests in seconds.
        backoff_factor (float): Multiplier for delays after errors.
        max_workers (int): Maximum number of concurrent workers.
        jitter_factor (float): Amount of randomness to add (0.0 to 1.0).

    """

    min_delay: float = 4.0
    max_delay: float = 15.0
    backoff_factor: float = 1.5
    max_workers: int = 3
    jitter_factor: float = 0.5


class SpeedLimiter:
    """Rate limiter for network operations with jitter to avoid predictable patterns.

    Controls the timing of network requests with randomized delays to avoid
    creating predictable request patterns that could be detected by rate limiters.

    Attributes:
        config (RateLimitConfig): Rate limiting configuration.
        last_request_time (float): Timestamp of the last request.
        current_delay (float): Current base delay between requests.
        error_count (int): Count of consecutive errors.
        success_count (int): Count of consecutive successes.

    """

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        """Initialize the SpeedLimiter.

        Args:
            config (RateLimitConfig, optional): Rate limiting configuration.
                Defaults to RateLimitConfig().

        """
        self.config = config or RateLimitConfig()
        self.last_request_time = 0.0
        self.current_delay = self.config.min_delay
        self.error_count = 0
        self.success_count = 0
        self._lock = Lock()

    def _apply_jitter(self, delay: float) -> float:
        """Apply jitter to a delay value to avoid predictable timing.

        Args:
            delay (float): Base delay in seconds.

        Returns:
            float: Jittered delay value.

        """
        if self.config.jitter_factor <= 0:
            return delay

        # Calculate jitter range: delay ± (delay * jitter_factor)
        jitter_amount = delay * self.config.jitter_factor
        min_jittered = delay - jitter_amount
        max_jittered = delay + jitter_amount

        # Ensure we don't go below absolute minimum
        min_jittered = max(min_jittered, self.config.min_delay * 0.5)

        return random.uniform(min_jittered, max_jittered)

    def wait_for_next_request(self) -> None:
        """Wait appropriate amount of time before next network request with jitter.

        Calculates and enforces delay based on current rate limiting state,
        applying random jitter to avoid predictable request patterns.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time

            # Apply jitter to the current delay
            jittered_delay = self._apply_jitter(self.current_delay)

            # Calculate required wait time
            wait_time = max(0, jittered_delay - elapsed)

            if wait_time > 0:
                logger.debug(
                    f"Rate limiting: waiting {wait_time:.2f}s (base: {self.current_delay:.2f}s, jittered: {jittered_delay:.2f}s)"
                )
                time.sleep(wait_time)

            self.last_request_time = time.time()

    def record_success(self) -> None:
        """Record a successful request and potentially reduce delay.

        Tracks successful requests and reduces delay if we've had several
        successes in a row, allowing for faster processing when possible.
        """
        with self._lock:
            self.error_count = 0
            self.success_count += 1

            # After 5 successful requests, slightly reduce delay
            if self.success_count >= 5:
                old_delay = self.current_delay
                self.current_delay = max(
                    self.config.min_delay, self.current_delay * 0.9
                )
                self.success_count = 0
                logger.debug(
                    f"Reduced delay from {old_delay:.2f}s to {self.current_delay:.2f}s after successes"
                )

    def record_error(self) -> None:
        """Record a failed request and increase delay with jitter.

        Tracks failed requests and increases delay to back off from
        potential rate limiting or server issues.
        """
        with self._lock:
            self.success_count = 0
            self.error_count += 1

            # Increase delay after errors
            old_delay = self.current_delay
            self.current_delay = min(
                self.config.max_delay, self.current_delay * self.config.backoff_factor
            )

            logger.warning(
                f"Increased delay from {old_delay:.2f}s to {self.current_delay:.2f}s after error (count: {self.error_count})"
            )


class TranscriptFetcher:
    """Fetches video transcripts with rate limiting and proxy support.

    Manages the process of retrieving transcripts from YouTube videos
    while respecting rate limits and using proxy servers to avoid
    IP-based restrictions.

    Attributes:
        speed_limiter (SpeedLimiter): Rate limiter for network requests.
        proxy_config (dict): Proxy configuration for requests.

    """

    def __init__(self, rate_limit_config: Optional[RateLimitConfig] = None) -> None:
        """Initialize the TranscriptFetcher.

        Args:
            rate_limit_config (RateLimitConfig, optional): Rate limiting configuration.
                Defaults to RateLimitConfig().

        """
        self.speed_limiter = SpeedLimiter(rate_limit_config)
        config = get_settings()

        # Setup proxy configuration
        self.proxy_config = WebshareProxyConfig(
            proxy_username=config.proxy.username,
            proxy_password=config.proxy.password.get_secret_value(),
            proxy_port=config.proxy.port,
        )

        self.api = YouTubeTranscriptApi(proxy_config=self.proxy_config)

    def fetch_transcript(self, video_meta: VideoMeta) -> VideoRecord:
        """Fetch transcript for a single video with rate limiting.

        Args:
            video_meta (VideoMeta): Video metadata containing video_id.

        Returns:
            VideoRecord: Complete record with transcript data and status.

        """
        self.speed_limiter.wait_for_next_request()

        try:
            logger.debug(f"Fetching transcript for video {video_meta.video_id}")

            # Fetch transcript using the API with proxy
            transcript_list = self.api.fetch(
                video_id=video_meta.video_id,
                languages=["en", "en-US"],  # Prefer English
            )

            # Convert to our model format
            lines = [
                TranscriptLine(
                    start=entry.start, duration=entry.duration, text=entry.text
                )
                for entry in transcript_list
            ]

            # Create flattened text
            full_text = " ".join(line.text for line in lines)

            transcript_doc = TranscriptDoc(
                video_id=video_meta.video_id, language="en", lines=lines, text=full_text
            )

            self.speed_limiter.record_success()

            return VideoRecord(
                meta=video_meta, status=RecordStatus.ok, transcript=transcript_doc
            )

        except TranscriptsDisabled:
            logger.info(f"Transcripts disabled for video {video_meta.video_id}")
            self.speed_limiter.record_success()  # Not an error, just unavailable
            return VideoRecord(
                meta=video_meta, status=RecordStatus.transcripts_disabled
            )

        except (NoTranscriptFound, VideoUnavailable) as e:
            logger.info(f"Video {video_meta.video_id} not found or unavailable: {e}")
            self.speed_limiter.record_success()  # Not an error, just unavailable
            return VideoRecord(meta=video_meta, status=RecordStatus.not_found)

        except Exception as e:
            logger.error(f"Error fetching transcript for {video_meta.video_id}: {e}")
            self.speed_limiter.record_error()
            return VideoRecord(meta=video_meta, status=RecordStatus.error, error=str(e))

    def fetch_transcripts_parallel(
        self, video_metas: List[VideoMeta]
    ) -> List[VideoRecord]:
        """Fetch transcripts for multiple videos using parallel workers.

        Uses ThreadPoolExecutor with limited concurrency to fetch transcripts
        while maintaining rate limiting across all workers.

        Args:
            video_metas (List[VideoMeta]): List of video metadata to process.

        Returns:
            List[VideoRecord]: List of video records with transcript data.

        """
        results = []

        logger.info(
            f"Starting transcript fetch for {len(video_metas)} videos with {self.speed_limiter.config.max_workers} workers"
        )

        with ThreadPoolExecutor(
            max_workers=self.speed_limiter.config.max_workers
        ) as executor:
            # Submit all tasks
            future_to_meta = {
                executor.submit(self.fetch_transcript, meta): meta
                for meta in video_metas
            }

            # Process completed tasks
            for future in as_completed(future_to_meta):
                meta = future_to_meta[future]
                try:
                    record = future.result()
                    results.append(record)
                    logger.info(f"Completed {meta.video_id}: {record.status}")
                except Exception as e:
                    logger.error(f"Unexpected error processing {meta.video_id}: {e}")
                    # Create error record
                    error_record = VideoRecord(
                        meta=meta, status=RecordStatus.error, error=str(e)
                    )
                    results.append(error_record)

        logger.info(f"Completed transcript fetch: {len(results)} total records")
        return results


def fetch_transcripts_for_channel(
    channel_name: str, catalog: List[VideoMeta]
) -> List[VideoRecord]:
    """Fetch transcripts for all videos from a channel.

    Convenience function that creates a TranscriptFetcher and processes
    all videos for a given channel with appropriate rate limiting.

    Args:
        channel_name (str): Name of the channel for logging purposes.
        catalog (List[VideoMeta]): List of video metadata to process.

    Returns:
        List[VideoRecord]: List of video records with transcript data.

    """
    logger.info(
        f"Fetching transcripts for channel '{channel_name}': {len(catalog)} videos"
    )

    fetcher = TranscriptFetcher()
    return fetcher.fetch_transcripts_parallel(catalog)
