# models/movie_models.py
from pydantic import BaseModel, Field, field_validator,HttpUrl
from typing import Optional, List, Dict, Any, Union

class BigFiveTrait(BaseModel):
    score: int = Field(..., ge=1, le=5)
    explanation: str

class CharacterProfileBigFive(BaseModel):
    Openness: BigFiveTrait
    Conscientiousness: BigFiveTrait
    Extraversion: BigFiveTrait
    Agreeableness: BigFiveTrait
    Neuroticism: BigFiveTrait

class CharacterProfileMyersBriggs(BaseModel):
    type: str = Field(..., min_length=4, max_length=4) # e.g., "ISTJ"
    explanation: str

    @field_validator('type')
    @classmethod
    def mbti_type_must_be_valid(cls, v: str) -> str:
        valid_types = {
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ"
        }
        if v.upper() not in valid_types:
            raise ValueError(f"Invalid MBTI type: {v}")
        return v.upper()

class GenreMix(BaseModel):
    genres: Dict[str, int] # e.g., {"action": 80, "comedy": 70}

    @field_validator('genres')
    @classmethod
    def genre_values_must_be_percentages(cls, v: Dict[str, int]) -> Dict[str, int]:
        for genre, percentage in v.items():
            if not (0 <= percentage <= 100):
                raise ValueError(f"Genre percentage for '{genre}' ({percentage}) must be between 0 and 100.")
        return v

class MatchingTags(BaseModel):
    tags: Optional[Dict[str, str]] = None # Key is tag name, value is justification

    @field_validator('tags')
    @classmethod
    def validate_tag_keys(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if v is None:
            return None
        allowed_tags = {
            "Optimistic Dystopia", "Identity Quest", "Lo-Fi Epic", "Solarpunk Saga",
            "Existential Laugh", "Third Culture Narrative", "Micro Revolution",
            "Everyday Magic", "Existential Grind", "Accidental Wholesome",
            "Imperfect Unions", "Analog Heartbeats", "Legacy Reckoning",
            "Genre Autopsy", "Retro Immersion"
        }
        for tag_name in v.keys():
            if tag_name not in allowed_tags:
                raise ValueError(f"Invalid matching_tag: '{tag_name}'. Allowed tags are: {allowed_tags}")
        return v

class RelatedMovie(BaseModel):
    title: str
    imdb_id: Optional[str] = None

class Recommendation(BaseModel):
    title: str
    year: Union[int, str] # To handle cases like "N/A" or actual year
    explanation: str
    imdb_id: Optional[str] = None

    @field_validator('year')
    @classmethod
    def validate_year_format(cls, v: Union[int, str]) -> Union[int, str]:
        if isinstance(v, int):
            if not (1800 <= v <= 2100): # Basic sanity check for year
                raise ValueError(f"Year {v} seems invalid.")
        elif isinstance(v, str):
            if v.isdigit():
                year_int = int(v)
                if not (1800 <= year_int <= 2100):
                    raise ValueError(f"Year string '{v}' seems invalid.")
                return year_int # Convert valid year string to int
            # Allow "N/A" or other specific strings if needed, or raise error for other strings
        return v


class CharacterListItem(BaseModel):
    name: str
    actor_name: str
    tmdb_person_id: Optional[int] = None # Expect this from TMDB, crucial for images
    description: str
    group: str
    aliases: Optional[List[str]] = None
    image_file: Optional[str] = None # Path to local image

class Relationship(BaseModel):
    source: str
    target: str
    type: str
    directed: bool = True # Default from your prompt analysis
    description: str
    sentiment: str # Consider Enum: "positive", "negative", "neutral", "complicated"
    strength: int = Field(..., ge=1, le=5)
    tense: str # Consider Enum: "past", "present", "evolving"

    @field_validator('sentiment')
    @classmethod
    def sentiment_must_be_valid(cls, v:str) -> str:
        if v not in ["positive", "negative", "neutral", "complicated"]:
            raise ValueError(f"Invalid sentiment: {v}")
        return v

    @field_validator('tense')
    @classmethod
    def tense_must_be_valid(cls, v:str) -> str:
        if v not in ["past", "present", "evolving"]:
            raise ValueError(f"Invalid tense: {v}")
        return v

# --- Models for LLM Call Outputs ---
class LLMCall1Output(BaseModel):
    movie_title: str
    movie_year: str # Keep as string for now to match TMDB initial input
    character_profile: str
    critical_reception: str
    visual_style: str
    most_talked_about_related_topic: str
    sequel: Optional[str] = None
    prequel: Optional[str] = None
    spin_off_of: Optional[str] = None
    spin_off: Optional[str] = None
    remake_of: Optional[str] = None
    remake: Optional[str] = None
    complex_search_queries: List[str] # Assuming it can be a list, current prompt implies one string

class LLMCall2Output(BaseModel):
    character_list: List[CharacterListItem] # LLM generates based on TMDB raw, but needs tmdb_person_id
    relationships: List[Relationship]

class LLMCall3Output(BaseModel):
    character_profile_big5: CharacterProfileBigFive
    character_profile_myersbriggs: CharacterProfileMyersBriggs
    genre_mix: GenreMix # Replaces the dict structure with a Pydantic model
    matching_tags: MatchingTags # Replaces the dict structure
    recommendations: List[Recommendation]


# --- Final Movie Data Model (for YAML output) ---
class MovieEntry(BaseModel):
    movie_title: str
    movie_year: str # Or int, if consistently parsed
    tmdb_movie_id: Optional[int] = None
    imdb_id: Optional[str] = None

    character_profile: str
    character_profile_big5: CharacterProfileBigFive
    character_profile_myersbriggs: CharacterProfileMyersBriggs
    critical_reception: str
    visual_style: str
    most_talked_about_related_topic: str
    genre_mix: GenreMix
    matching_tags: MatchingTags
    complex_search_queries: List[str] # Or str if only one

    sequel: Optional[RelatedMovie] = None
    prequel: Optional[RelatedMovie] = None
    spin_off_of: Optional[RelatedMovie] = None
    spin_off: Optional[RelatedMovie] = None
    remake_of: Optional[RelatedMovie] = None
    remake: Optional[RelatedMovie] = None

    recommendations: Optional[List[Recommendation]] = None
    character_list: Optional[List[CharacterListItem]] = None
    relationships: Optional[List[Relationship]] = None


# --- Models for TMDB API responses (examples) ---
class TMDBMovieResult(BaseModel):
    id: int
    title: str
    release_date: str # YYYY-MM-DD
    # ... other fields you care about

    @property
    def year(self) -> Optional[str]:
        if self.release_date and len(self.release_date) >= 4:
            return self.release_date[:4]
        return None

class TMDBTopRatedResponse(BaseModel):
    page: int
    results: List[TMDBMovieResult]
    total_pages: int
    total_results: int

class TMDBRawCharacter(BaseModel):
    tmdb_character_name: str
    tmdb_actor_name: str
    tmdb_person_id: int # This is critical