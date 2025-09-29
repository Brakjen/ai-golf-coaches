"""Advanced concept extraction and philosophy generation for golf coaches.

This module provides tools for analyzing golf instruction transcripts to extract
key concepts, coaching philosophies, and conceptual relationships. It helps improve
RAG system understanding by identifying related content across different videos.

Note: This module is currently unused in favor of explicit personality definitions
in personalities.py, but retained for potential future concept analysis needs.

Classes:
    ConceptExtractor: Analyzes transcripts for golf concepts and philosophies

Functions:
    enhance_query_with_concepts: Adds conceptual context to user queries
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """Extract key concepts and philosophical themes from coach transcripts.

    This class analyzes transcript files to identify frequently mentioned golf concepts,
    coach-specific terminology, and conceptual relationships between different videos.
    It can generate philosophy summaries and find conceptually related content.

    Attributes:
        GOLF_CONCEPTS: Core golf concept categories and associated terms
        EGS_TERMS: Elite Golf Schools specific terminology
        MILO_TERMS: Milo Lines Golf specific terminology
        data_path: Path to transcript data directory

    Note:
        Currently unused in favor of explicit coach personalities, but available
        for advanced concept analysis if needed.

    """

    # Core golf concepts to track
    GOLF_CONCEPTS = {
        "setup": ["setup", "stance", "posture", "alignment", "ball position", "grip"],
        "backswing": ["takeaway", "backswing", "turn", "rotation", "coil", "width"],
        "transition": [
            "transition",
            "downswing",
            "sequence",
            "timing",
            "alpha",
            "force",
        ],
        "impact": ["impact", "contact", "strike", "compression", "face", "path"],
        "follow_through": ["follow through", "finish", "release", "extension"],
        "body_mechanics": [
            "hip",
            "shoulder",
            "spine",
            "pelvis",
            "core",
            "ground force",
        ],
        "mental_game": [
            "feel",
            "visualization",
            "confidence",
            "practice",
            "course management",
        ],
        "equipment": ["club", "shaft", "loft", "lie", "fitting"],
        "fundamentals": ["fundamentals", "basics", "foundation", "principles"],
    }

    # EGS-specific terminology
    EGS_TERMS = [
        "x-factor",
        "negative alpha",
        "angular force",
        "coronal plane",
        "supination",
        "pronation",
        "kinematic sequence",
        "ground reaction",
        "biomechanics",
        "injury prevention",
        "power generation",
    ]

    # Milo-specific terminology
    MILO_TERMS = [
        "feel",
        "image",
        "drill",
        "progression",
        "simple",
        "practical",
        "course management",
        "real world",
        "what works",
        "street smarts",
    ]

    def __init__(self, data_path: Path = Path("data/raw")) -> None:
        """Initialize ConceptExtractor with data path.

        Args:
            data_path: Path to transcript data directory.

        """
        self.data_path = data_path

    def extract_coach_concepts(self, coach_dir: str) -> Dict[str, Any]:
        """Extract comprehensive concept map for a coach.

        Args:
            coach_dir: Directory name containing coach's transcript files

        Returns:
            Dict containing:
            - concept_mentions: Categorized concept occurrences by video
            - term_frequency: Most frequent coach-specific terms
            - video_concepts: Concept categories by video
            - total_videos: Number of videos analyzed

        Raises:
            None: Gracefully handles missing files and parsing errors

        """
        coach_path = self.data_path / coach_dir
        if not coach_path.exists():
            return {}

        # Concept frequency tracking
        concept_mentions = defaultdict(list)
        term_frequency = Counter()
        video_concepts = defaultdict(set)

        json_files = list(coach_path.glob("*.json"))
        json_files = [f for f in json_files if f.name != "_catalog.json"]

        logger.info(f"Analyzing {len(json_files)} videos for {coach_dir}")

        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                video_id = data.get("meta", {}).get("video_id", jf.stem)
                title = data.get("meta", {}).get("title", "")

                # Get full transcript text
                transcript = data.get("transcript", {})
                if "text" in transcript:
                    text = transcript["text"].lower()
                elif "lines" in transcript:
                    text = " ".join(
                        [line.get("text", "") for line in transcript["lines"]]
                    ).lower()
                else:
                    continue

                # Track concept categories
                for category, terms in self.GOLF_CONCEPTS.items():
                    for term in terms:
                        if term in text:
                            concept_mentions[category].append(
                                {
                                    "video_id": video_id,
                                    "title": title,
                                    "term": term,
                                    "count": text.count(term),
                                }
                            )
                            video_concepts[video_id].add(category)

                # Track coach-specific terms
                coach_terms = (
                    self.EGS_TERMS if "elitegolf" in coach_dir else self.MILO_TERMS
                )
                for term in coach_terms:
                    count = text.count(term)
                    if count > 0:
                        term_frequency[term] += count

            except Exception as e:
                logger.warning(f"Error processing {jf}: {e}")
                continue

        return {
            "concept_mentions": dict(concept_mentions),
            "term_frequency": dict(term_frequency.most_common(20)),
            "video_concepts": dict(video_concepts),
            "total_videos": len(json_files),
        }

    def generate_philosophy_summary(self, coach_dir: str) -> str:
        """Generate a comprehensive philosophy summary for a coach.

        Args:
            coach_dir: Directory name containing coach's transcript files

        Returns:
            str: Generated philosophy summary based on concept analysis,
                 or empty string if analysis fails

        Notes:
            - Analyzes concept frequency to identify coaching focus areas
            - Generates coach-specific philosophy statements
            - Currently unused in favor of explicit personality definitions

        """
        concepts = self.extract_coach_concepts(coach_dir)

        if not concepts["concept_mentions"]:
            return ""

        # Find most emphasized concept categories
        concept_scores = {}
        for category, mentions in concepts["concept_mentions"].items():
            concept_scores[category] = len(mentions)

        top_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        # Build philosophy summary
        philosophy_parts = []

        if "elitegolf" in coach_dir:
            philosophy_parts.append(
                "Elite Golf Schools emphasizes biomechanically sound golf swings using the X-Factor method."
            )

            # Add specific focus areas based on concept analysis
            if "body_mechanics" in [c[0] for c in top_concepts[:3]]:
                philosophy_parts.append(
                    "Core focus on proper body mechanics, ground force reaction, and kinematic sequencing."
                )
            if "transition" in [c[0] for c in top_concepts[:3]]:
                philosophy_parts.append(
                    "Special emphasis on transition mechanics including negative alpha force and angular momentum."
                )

            # Add coach-specific terminology
            top_terms = list(concepts["term_frequency"].keys())[:5]
            if top_terms:
                philosophy_parts.append(
                    f"Key technical concepts: {', '.join(top_terms)}."
                )

        elif "milo" in coach_dir:
            philosophy_parts.append(
                "Milo Lines Golf focuses on practical, feel-based instruction that works on the course."
            )

            if "mental_game" in [c[0] for c in top_concepts[:3]]:
                philosophy_parts.append(
                    "Emphasizes feel, visualization, and practical course management."
                )
            if "fundamentals" in [c[0] for c in top_concepts[:3]]:
                philosophy_parts.append(
                    "Strong foundation in fundamentals with simple, progressive drills."
                )

        return " ".join(philosophy_parts)

    def find_concept_connections(
        self, coach_dir: str, query_terms: List[str]
    ) -> List[Dict]:
        """Find videos that discuss related concepts, not just keyword matches.

        Args:
            coach_dir: Directory name containing coach's transcript files
            query_terms: List of terms from user query

        Returns:
            List[Dict]: Videos ranked by conceptual relevance, each containing:
            - video_id: Video identifier
            - concept_score: Relevance score based on concept overlap

        Notes:
            - Maps query terms to golf concept categories
            - Finds videos heavily discussing related concepts
            - Provides conceptual rather than keyword-based matching

        """
        concepts = self.extract_coach_concepts(coach_dir)

        # Find which concept categories the query relates to
        related_categories = set()
        query_lower = [term.lower() for term in query_terms]

        for category, terms in self.GOLF_CONCEPTS.items():
            for query_term in query_lower:
                for golf_term in terms:
                    if query_term in golf_term or golf_term in query_term:
                        related_categories.add(category)

        # Find videos that heavily discuss these concepts
        video_scores = defaultdict(int)
        for category in related_categories:
            if category in concepts["concept_mentions"]:
                for mention in concepts["concept_mentions"][category]:
                    video_scores[mention["video_id"]] += mention["count"]

        # Return top videos by concept relevance
        sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"video_id": vid, "concept_score": score}
            for vid, score in sorted_videos[:10]
        ]


def enhance_query_with_concepts(query: str, coach_dir: str) -> str:
    """Enhance a query with related conceptual terms.

    Args:
        query: Original user query
        coach_dir: Directory name containing coach's transcript files

    Returns:
        str: Enhanced query with conceptual context, or original query
             if no concept connections found

    Notes:
        - Currently unused in favor of explicit personality approach
        - Could be used for advanced query expansion in future
        - Identifies conceptually related content beyond keyword matching

    """
    extractor = ConceptExtractor()

    # Extract key terms from query
    query_terms = query.lower().split()

    # Find conceptually related videos
    related_videos = extractor.find_concept_connections(coach_dir, query_terms)

    if related_videos:
        # Add concept-based context to improve retrieval
        concept_context = "Related concepts for deeper understanding: "
        return f"{query} {concept_context}"

    return query
