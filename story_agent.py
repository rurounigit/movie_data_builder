import openai
import time
import os
import yaml # For parsing and dumping YAML
import requests # For making HTTP requests
import urllib.parse # For URL encoding
from dotenv import load_dotenv
import shutil # For file operations (like saving images)

load_dotenv()

# --- Configuration ---
MODEL_ID = "gemma-3-12b-it-qat"
REQUESTED_MODEL_NAME_FOR_LOG = "gemma-3-12b-it-qat"

BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio" # For OpenAI client, not OMDB/TMDB

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# --- Desired Lengths (approximate words) ---
TARGET_PROFILE_WORDS = 1100
MAX_CHARACTERS_FROM_TMDB = 15

# --- Image Configuration ---
CHARACTER_IMAGE_SAVE_PATH = "character_images"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"
TMDB_IMAGE_SIZE = "w500" # Or "original", "w185", "w300", etc.


# --- Expected Keys for the Output YAML ---
EXPECTED_YAML_KEYS_CALL_1 = [ # LLM Call 1: Data Generation for a GIVEN movie
    "movie_title", # LLM must echo the title it was given
    "movie_year",  # LLM must echo the year it was given
    "character_profile",
    "critical_reception",
    "visual_style",
    "most_talked_about_related_topic",
    "sequel",
    "prequel",
    "spin_off_of",
    "spin_off",
    "remake_of",
    "remake",
    "complex_search_queries",
]

EXPECTED_YAML_KEYS_ENRICH_REL_CALL = [ # LLM Call 2: Character Enrichment & Relationships
    "character_list",
    "relationships",
]

EXPECTED_YAML_KEYS_ANALYTICAL_CALL = [ # LLM Call 3: Analytical Data
    "character_profile_big5",
    "character_profile_myersbriggs",
    "genre_mix",
    "matching_tags",
    "recommendations",
]

CHARACTER_LIST_ITEM_KEYS = ["name", "actor_name", "tmdb_person_id", "description", "group", "aliases", "image_file"]


EXPECTED_YAML_KEYS_FINAL = [
    "movie_title",
    "movie_year",
    "tmdb_movie_id", # Added TMDB movie ID for the main movie
    "character_profile",
    "character_profile_big5",
    "character_profile_myersbriggs",
    "critical_reception",
    "most_talked_about_related_topic",
    "genre_mix",
    "matching_tags",
    "sequel",
    "prequel",
    "spin_off_of",
    "spin_off",
    "remake_of",
    "remake",
    "complex_search_queries",
    "recommendations",
    "imdb_id",
    "character_list",
    "relationships",
    "visual_style",
]

CRITICAL_KEY_FOR_CLEAN_OUTPUT = "movie_title"
IMDB_ID_KEY = "imdb_id"
RELATED_MOVIE_KEYS = ["sequel", "prequel", "spin_off_of", "spin_off", "remake_of", "remake"]

# --- Session Control ---
NUM_NEW_MOVIES_TO_FETCH_THIS_SESSION = 5
MAX_TMDB_TOP_RATED_PAGES_TO_CHECK = 10 # Max pages of TMDB top rated to check in a session

def words_to_tokens(words):
    return int(words * 1.4)

MAX_TOKENS_CALL_1 = words_to_tokens(800 + 100) # Reduced slightly as no movie choice text
MAX_TOKENS_ENRICH_REL_CALL = words_to_tokens( (MAX_CHARACTERS_FROM_TMDB * 80) + (MAX_CHARACTERS_FROM_TMDB * 1.5 * 60) + 350)
MAX_TOKENS_ANALYTICAL_CALL = words_to_tokens(1200 + 200)
MAX_TOKENS_TITLE_EXTRACTION = words_to_tokens(20 + 15)


# --- PROMPT FOR CALL 1 (Data Generation for a GIVEN movie) ---
MOVIE_PROMPT_CALL_1_TEMPLATE = (
    "You are given the following movie details obtained from The Movie Database (TMDB):\n"
    "Movie Title: {movie_title_from_tmdb}\n"
    "Release Year: {movie_year_from_tmdb}\n\n"
    "Your task is to provide detailed information for THIS SPECIFIC MOVIE. "
    "Respond strictly in YAML format. The YAML should be a single object.\n"
    "CRITICALLY IMPORTANT: The '{expected_title_key}' key in your YAML output MUST be the exact string '{movie_title_from_tmdb}', "
    "and the '{expected_year_key}' key MUST be the exact string '{movie_year_from_tmdb}'. Do not alter them.\n\n"
    "Provide these {num_call_1_keys} top-level keys with appropriate content for the movie '{movie_title_from_tmdb}':\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[0]}' (string, exactly '{{movie_title_from_tmdb}}'),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[1]}' (string, exactly '{{movie_year_from_tmdb}}'),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[2]}' (multi-line string for the character profile text, around 150 words. Focus on the overall character landscape or the protagonist if one is very central. "
    "Format as a YAML literal block scalar. Example: character_profile: |\\n  Line one.\\n  Line two.),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[3]}' (international critical reception summary (around 150 words), multi-line string),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[4]}' (concise but concrete and tangible description of the visual style of the movie (around 200 words), multi-line string),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[5]}' (most controversial, shocking, or moving related topic or trivia about the movie (around 150 words), multi-line string),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[6]}' (name the sequel if one exists, string, otherwise null),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[7]}' (name the prequel if one exists, string, otherwise null),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[8]}' (what movie is this movie a spin-off of, if any, string, otherwise null),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[9]}' (what is this movie's spin-off, if any, string, otherwise null),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[10]}' (what movie is this movie a remake of, if any, string, otherwise null),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[11]}' (what is this movie's remake, if any, string, otherwise null),\n"
    f"'{EXPECTED_YAML_KEYS_CALL_1[12]}' (give 1 short, insightful search query to find similar movies (be vague!), examples: 'Critically acclaimed 90s mystery movies set in Europe.', 'Visually stunning post-2015 sci-fi with strong female leads.').\n"
)

# --- PROMPT FOR LLM CALL 2 (Character Enrichment & Relationships) ---
MOVIE_PROMPT_ENRICH_CHARS_AND_RELATIONSHIPS_TEMPLATE = (
    "You are processing the movie \"{movie_title}\" (released around {movie_year}).\n"
    "Contextual Information:\n"
    "TMDB Raw Character/Actor List (includes tmdb_person_id):\n"
    "This is a list of characters, the actors who played them, and their TMDB person IDs, obtained from TMDB. Your first task is to enrich this list.\n"
    "```yaml\n{raw_tmdb_characters_yaml}\n```\n\n"
    "TASK 1: Enrich Character List\n"
    "Transform the 'TMDB Raw Character/Actor List' above into a detailed `character_list`. For each character entry from the raw list:\n"
    "1. Use the `tmdb_character_name` as the value for the `name` key (this is the base character name from TMDB, use this exact string for the name key).\n"
    "2. Add an `actor_name` key with the value from `tmdb_actor_name`.\n"
    "3. Add a `tmdb_person_id` key with the integer value from `tmdb_person_id` provided in the raw input.\n"
    "4. Generate a concise `description` for the character (e.g., their role, key traits, brief arc summary in 1-2 sentences).\n"
    "5. Determine an appropriate `group` for the character (e.g., 'Dwarves', 'Tribe Members', 'Shawshank Prisoners', 'Guards', 'Family Members', 'Political Enemies', etc., be specific to the movie context and narrative). Avoid generic 'Cast'.\n"
    "6. Generate a YAML list of `aliases` (common nicknames or alternative names used for the character in the movie) or `null` if there are no significant aliases.\n"
    "The output for this task should be a YAML list under the top-level key `character_list`.\n"
    "Example for one character in `character_list` (if raw input was tmdb_character_name: \"Andy Dufresne\", tmdb_actor_name: \"Tim Robbins\", tmdb_person_id: 2):\n"
    "  - name: \"Andy Dufresne\"\n"
    "    actor_name: \"Tim Robbins\"\n"
    "    tmdb_person_id: 2\n"
    "    description: \"A banker wrongly convicted of murder, who maintains hope and seeks freedom while in Shawshank Penitentiary.\"\n"
    "    group: \"Shawshank Prisoners\"\n"
    "    aliases: null\n\n"
    "TASK 2: Generate Relationships\n"
    "Generate *ONLY* and *ALL* the relationships between the characters from the `character_list` you *just generated in TASK 1* (specifically the `name` fields from it). "
    "Respond strictly in YAML format. The YAML should be a `relationships` list.\n"
    "The `relationship` list should contain YAML dictionaries. Each dictionary represents a relationship and must have these keys: source, target, type, description, sentiment, strength, and tense.\n"
    "CRITICALLY IMPORTANT: The 'source' and 'target' values MUST EXACTLY MATCH one of the 'name' values from the `character_list`. Do not invent new characters or use variations of names not present in that list.\n"
    "IMPORTANT: 2 characters can only appear both together in ONE relationship. If 2 characters already appear in one relationship, DO NOT add create another relationship with these two characters.\n"
    "`tense` indicates if the relationship is primarily in the past, present, or evolving, changing within the movie's timeline.\n"
    "ALLOWED VALUES for relationship keys: source (string from your generated character list names), target (string from your generated character list names), type (string, e.g., \"Friend\", \"Mentor\", \"Brother\", \"Rival\"), directed (boolean: true or false), description (string, 1-2 sentences explaining the nature of the relationship), sentiment (string: \"negative\", \"positive\", \"neutral\", \"complicated\"), strength (integer: 1-5, where 5 is very strong), tense (string: \"past\", \"present\", \"evolving\").\n"
    "Example for relationships:\n"
    "  - source: \"Andy Dufresne\"\n"
    "    target: \"Ellis Boyd Redding\"\n"
    "    type: \"Friend\"\n"
    "    description: \"Andy and Red form a strong, enduring bond based on mutual respect, shared experiences, and hope within Shawshank.\"\n"
    "    sentiment: \"positive\"\n"
    "    strength: 5\n"
    "    tense: \"evolving\"\n\n"
    "FINAL RESPONSE FORMAT:\n"
    "Your entire response MUST be a single YAML object with exactly two top-level keys: `character_list` (containing your enriched list from TASK 1) and `relationships` (containing your list from TASK 2)."
)

# --- PROMPT FOR LLM CALL 3 (Analytical Data - Restored full examples and definitions) ---
MOVIE_PROMPT_ANALYTICAL_TEMPLATE = (
    "For the movie titled \"{movie_title_from_call_1}\" (released around {movie_year_from_call_1}). "
    "Provide the following analytical information and recommendations for this specific movie. "
    "Respond strictly in YAML format. The YAML should be a single object with these {num_analytical_keys} top-level keys: "
    f"'{EXPECTED_YAML_KEYS_FINAL[4]}' (using the Big Five personality model for the main character, a dictionary where keys are Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. Each key's value must be a YAML list with two items: the first being the score (an integer from 1 to 5) and the second being a very concise string explainer for that trait. " # Index adjusted for tmdb_movie_id
    "Example for this key, based on Clarice Starling from 'The Silence of the Lambs': {{{{ "
      "Openness: [4, \"Clarice shows openness by venturing into unfamiliar, dangerous psychological territory to catch Buffalo Bill.\"], "
      "Conscientiousness: [5, \"She is exceptionally diligent, disciplined, and committed to her FBI training and the case, following procedures meticulously.\"], "
      "Extraversion: [2, \"Clarice is often reserved and task-focused, more introverted than overtly outgoing, especially in high-pressure situations.\"], "
      "Agreeableness: [3, \"While driven and sometimes clashing with superiors, she shows empathy for victims and can be cooperative when necessary.\"], "
      "Neuroticism: [3, \"She experiences fear and stress, especially with Lecter, but generally manages it to perform effectively under pressure.\"] "
    "}}}}. Use the scale: 1=Strongly Disagree/Very Inaccurate, 2=Disagree/Inaccurate, 3=Neutral/Neither Accurate nor Inaccurate, 4=Agree/Accurate, 5=Strongly Agree/Very Accurate for the scores.), and "
    f"'{EXPECTED_YAML_KEYS_FINAL[5]}' (using the Myers-Briggs personality model for the main character, a dictionary where the key is the MBTI type string (e.g., \"ISTJ\") and the value is a single, concise string explainer for that type. " # Index adjusted
    "Example for this key, based on Clarice Starling from 'The Silence of the Lambs' being typed as ISTJ: {{{{ "
      "\"ISTJ\": \"Introverted, Sensing, Thinking, Judging (Logistician). Clarice demonstrates ISTJ traits through her dutiful approach to her FBI responsibilities, her methodical investigation, and her reliance on concrete facts and established procedures to navigate complex challenges.\" "
    "}}}}. these are the 16 types: (\"ESFP\": \"Extraverted, Sensing, Feeling, Perceiving (Entertainer)\", \"INTJ\": \"Introverted, Intuitive, Thinking, Judging (Architect)\", \"INTP\": \"Introverted, Intuitive, Thinking, Perceiving (Thinker)\", \"ENTJ\": \"Extraverted, Intuitive, Thinking, Judging (Commander)\", \"ENTP\": \"Extroverted, Intuitive, Thinking, Perceiving (Debater)\", \"INFJ\": \"Introverted, Intuitive, Feeling, Judging (Advocate)\", \"INFP\": \"Introverted, Intuitive, Feeling, Perceiving (Mediator)\", \"ENFJ\": \"Extraverted, Intuitive, Feeling, Judging (Protagonist)\", \"ENFP\": \"Extroverted, Intuitive, Feeling, Perceiving (Champion)\", \"ISTJ\": \"Introverted, Sensing, Thinking, Judging (Logistician)\", \"ISFJ\": \"Introverted, Sensing, Feeling, Judging (Protector)\", \"ESTJ\": \"Extraverted, Sensing, Thinking, Judging (Director)\", \"ESFJ\": \"Extroverted, Sensing, Feeling, Judging (Caregiver)\", \"ISTP\": \"Introverted, Sensing, Thinking, Perceiving (Crafter)\", \"ISFP\": \"Introverted, Sensing, Feeling, Perceiving (Adventurer)\", \"ESTP\": \"Extraverted, Sensing, Thinking, Perceiving (Entrepreneur)\")), and "
    f"'{EXPECTED_YAML_KEYS_FINAL[8]}' (what genres are apparent in this movie? Identify 6 genre names and 6 percentages of the genre mix. Example: {{{{ \"action\": 80, \"comedy\": 70, \"sci-fi\": 50, \"drama\": 20, \"romance\": 10, \"fantasy\": 50 }}}} , dictionary of genres and numbers), and " # Index adjusted
    f"'{EXPECTED_YAML_KEYS_FINAL[9]}' (a YAML dictionary for `matching_tags`. " # Index adjusted
    "**RULES FOR TAG SELECTION - READ VERY CAREFULLY:**\\n"
    "1. **Accuracy is Paramount:** Your primary goal is to select tags that are UNDENIABLY and CENTRALLY relevant to the movie. The tag MUST be a core, defining aspect of the film.\\n"
    "2. **Strict Adherence to Definitions:** You MUST strictly follow the 'TAG DEFINITIONS' provided below for each tag. Do NOT deviate or interpret loosely.\\n"
    "3. **No Stretching Definitions:** If a movie only vaguely touches upon a tag, or if the tag is a minor or incidental aspect, DO NOT apply the tag.\\n"
    "4. **Pre-conditions are Absolute:** If a tag definition includes a pre-condition (e.g., 'Movie MUST BE or HAS TO BE a Franchise continuation'), that condition MUST be met. If not, the tag CANNOT be used for this movie.\\n"
    "5. **Quality Over Quantity:** It is better to have FEW or NO tags if none strongly and centrally apply, rather than including weakly matched tags.\\n"
    "6. **Output Format:** If tags are selected, keys in the dictionary MUST be the exact strings from the 'TAGS' list. Values MUST be concise, movie-specific string explanations (1 sentence) justifying WHY that tag demonstrably and centrally applies based on its definition.\\n"
    "7. **No Matches:** If, after careful consideration of these strict rules, NO tags strongly and centrally match the movie, provide `null` as the value for the `matching_tags` key. This is a perfectly acceptable output.\\n\\n"
    "TAGS (Refer to 'TAG DEFINITIONS' below for full details. Selection MUST strictly follow these definitions and the rules above.):\\n"
    "[\"Optimistic Dystopia\", \"Identity Quest\", \"Lo-Fi Epic\", \"Solarpunk Saga\", \"Existential Laugh\", \"Third Culture Narrative\", \"Micro Revolution\", \"Everyday Magic\", \"Existential Grind\", \"Accidental Wholesome\", \"Imperfect Unions\", \"Analog Heartbeats\", \"Legacy Reckoning\", \"Genre Autopsy\", \"Retro Immersion\"]\\n\\n"
    "Example for `matching_tags` (if the movie was 'In the Mood for Love' and it strongly met these definitions per the rules):\n"
    "matching_tags:\n"
    "  \"Analog Heartbeats\": \"It focuses on the intimacy of quiet, shared moments and unspoken emotions in a pre-digital era.\"\n"
    "  \"Imperfect Unions\": \"The central bond is defined by its painful imperfection and societal constraints, remaining unconsummated.\"\n\n"
    "\\n\\nFOR YOUR REFERENCE, TAG DEFINITIONS ARE (Use these to understand the tags; do NOT include these full definitions in your output. Your explanation MUST be movie-specific and justify the tag based on these definitions): "
    "\\nOptimistic Dystopia: the film's setting has to be a world with significant challenges (e.g., extreme climate change, societal division, tech overreach) that, unlike bleak dystopias, centers on characters actively building better futures through community, innovation, empathy, or rebellion against oppressive systems. (e.g., Nausicaä of the Valley of the Wind, Tomorrowland)."
    "\\nIdentity Quest: the films has to rely mostly on diving deep into characters' internal worlds, mental health, and complex journeys of self-discovery. May use surrealism, metaphor, or unique narratives to explore anxiety, depression, neurodivergence, finding one's place, or multifaceted identities. (e.g., Eternal Sunshine of the Spotless Mind, Lady Bird)."
    "\\nLo-Fi Epic: the story of the movie has to include very grand emotional stakes or themes of significant change, but in contrast be told through a grounded, intimate, character-driven lens (e.g., The Green Knight, A Ghost Story)."
    "\\nSolarpunk Saga: film has to blend ecological themes with mythological storytelling, ancient wisdom, or a solarpunk aesthetic (optimistic, sustainable futures via renewable energy & community). Characters rediscover nature, fight for environmental or build regenerative communities. (e.g., Princess Mononoke)."
    "\\nExistential Laugh: film has to rely heavily on dark humour. Film also has to explore big existential questions (purpose, meaning, connection), e.g., Sorry to Bother You, Being John Malkovich."
    "\\nThird Culture Narrative: films story has to include experiences of individuals navigating multiple cultural identities (e.g., 'third culture kids,' immigrants, those at the intersection of different worlds). (e.g., Minari, Bend It Like Beckham)."
    "\\nMicro Revolution: film has to be centered on small-scale grassroots movements, community organizing, or everyday rebellions leading to significant change (e.g., Pride, Paddington 2)."
    "\\nEveryday Magic: film has to be about finding the fantastical, magical, or subtly surreal in specific, mundane, local settings. Not high fantasy, but magic hidden in familiar places, local legends, or ordinary people with extraordinary abilities tied to their environment. (e.g., Amélie, Stranger than Fiction)."
    "\\nExistential Grind: film has to capture the hilariously bleak, absurd, or mundane aspects of modern daily life (work, dating, bureaucracy). Characters may be trapped in ironic loops or face Sisyphean tasks. (e.g., Office Space, Triangle of Sadness)."
    "\\nAccidental Wholesome: film has to include a grumpy, cynical, or isolated character who inadvertently stumbles into heartwarming situations, forming unlikely bonds, or become part of an unexpected found family. Wholesomeness is often a surprise to the characters, emerging from chaos or absurdity. (e.g., Up, Hunt for the Wilderpeople)."
    "\\nImperfect Unions: film has to  portray a relationship (romantic or deeply platonic) where the individuals connect over shared anxieties, traumas, mental health struggles and other vulnerabilities, rather than fairytale ideals. Focuses on accepting imperfections and navigating challenges together. (e.g., Silver Linings Playbook, Punch-Drunk Love)."
    "\\nAnalog Heartbeats: film has to have a romance or story of longing/connection with a 'lo-fi' aesthetic (quiet, intimate, possibly vintage-feeling). Characters may seek tangible connections (letters, mixtapes, quiet moments) in a digital world.  (e.g., Before Sunrise, Call Me By Your Name)."
    "\\nLegacy Reckoning: PRE-CONDITION: Movie MUST BE a Franchise continuation (sequel, reboot). Has to show an aging protagonist confronting his past and some form of Meta-narrative about franchise history (e.g., Creed, Blade Runner 2049, Top Gun: Maverick, Logan, Star Wars: The Force Awakens)."
    "\\nGenre Autopsy: film has to be meta-aware, not like the typical films of the genre, has to actively dissect, deconstruct, satirize, or rebuild genre tropes. The film has to expose genre mechanics, blend styles unconventionally, and play with audience expectations. (e.g., Cabin in the Woods, Everything Everywhere All At Once, Deadpool)."
    "\\nRetro Immersion: film has to be deeply saturated in a past era's aesthetics, music, and culture (e.g., 80s/90s). (e.g., Everybody Wants Some!!, Sing Street, Stranger Things). "
    f"'), and '{EXPECTED_YAML_KEYS_FINAL[17]}' (a YAML list under the key 'recommendations', containing 5 DIFFERENT movie recommendations that would appeal to someone who liked this movie, based on its high IMDb IMDb rating, overall vibe, and main character personality. "
    "Each recommendation MUST be a YAML dictionary (map) containing exactly four items: "
    "1. `title`: The Recommended Movie Title (string)."
    "2. `year`: The Recommended Movie's 4-digit release year (integer or 4-digit string)."
    "3. `explanation`: A short, 1-sentence string explainer for why it's recommended, linking to these aspects."
    "IMPORTANT: the recommended movie should NEVER be the same as \"{movie_title_from_call_1}\"!"
    "DO NOT recommend the same movie more than once!!"
    "Focus on quality and comparable viewing experience rather than only direct thematic similarity."
    "Example if the main movie was 'The Silence of the Lambs':\n"
    "recommendations:\n"
    "  - title: \"Parasite\"\n"
    "    year: 2019\n"
    "    explanation: \"A critically acclaimed, high-rated thriller with a unique, intense vibe and masterfully crafted suspense that fans of intelligent, dark narratives would appreciate.\"\n"
    "  - title: \"No Country for Old Men\"\n"
    "    year: 2007\n"
    "    explanation: \"Shares a similar high IMDb rating and a chilling, suspenseful vibe with a memorable, unsettling antagonist and a determined protagonist.\"\n"
    "  - title: \"Mystic River\"\n"
    "    year: 2003\n"
    "    explanation: \"A highly-rated, dark crime drama with strong character performances and a somber, investigative atmosphere that resonates with the main film's intensity.\"\n"
    "  - title: \"The Girl with the Dragon Tattoo\"\n"
    "    year: 2011 # Or 2009 if you prefer the original\n"
    "    explanation: \"Features a determined, intelligent female protagonist investigating dark crimes, sharing a similar investigative intensity and a compelling, complex character dynamic.\"\n"
    "  - title: \"Blade Runner 2049\"\n"
    "    year: 2017\n"
    "    explanation: \"A visually stunning, thought-provoking sci-fi with a high IMDb rating and a solitary, investigative protagonist navigating a dark world, offering a different genre but similar depth.\"\n"
)





EXTRACT_TITLE_PROMPT_TEMPLATE = (
    "From the following YAML text, extract only the value of the 'movie_title' key. "
    "Respond with just the movie title string, no other text, no YAML formatting.\n\n"
    "YAML Text:\n```yaml\n{yaml_text_to_extract_from}\n```\n"
    "Movie Title:"
)

RAW_LOG_FILENAME = "generated_movie_data_raw_log.txt"
CLEAN_YAML_OUTPUT_FILENAME = "clean_movie_database.yaml"

client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- TMDB API Functions ---
def fetch_top_rated_movies_from_tmdb(page=1, segment_num_str="TMDBTopRated"):
    if not TMDB_API_KEY:
        print(f"      Error (Segment {segment_num_str}): TMDB_API_KEY not set. Cannot fetch top rated movies.")
        return None
    url = f"https://api.themoviedb.org/3/movie/top_rated?language=en-US&page={page}"
    # The discover call equivalent uses vote_count.gte=200. The direct /top_rated might have different (or no explicit) thresholds shown in basic docs.
    # For consistency with discover's likely higher quality filter, one might use the discover endpoint:
    # url = f"https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page={page}&sort_by=vote_average.desc&vote_count.gte=200"
    # However, sticking to /movie/top_rated as per user prompt for now.
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}
    try:
        print(f"      Querying TMDB Top Rated movies (Page {page})...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and "results" in data:
            print(f"      TMDB Top Rated (Page {page}): Found {len(data['results'])} movies. Total pages: {data.get('total_pages')}")
            return data # Return the whole response object as it contains page info
        else:
            print(f"      TMDB Top Rated (Page {page}): No results found or malformed response.")
            return None
    except requests.exceptions.Timeout:
        print(f"      Timeout (Segment {segment_num_str}): TMDB Top Rated API request timed out for page {page}.")
    except requests.exceptions.RequestException as e:
        print(f"      Error (Segment {segment_num_str}): Calling TMDB Top Rated API for page {page}: {e}")
    except Exception as e:
        print(f"      Unexpected error (Segment {segment_num_str}): During TMDB Top Rated lookup for page {page}: {e}")
    return None


# --- IMDb ID Fetching Functions ---
def get_imdb_id_from_omdb(movie_title, year=None, segment_num_str="OMDB"):
    if not OMDB_API_KEY:
        # print(f"      Info (Segment {segment_num_str}): OMDB_API_KEY not set. Skipping OMDB lookup for '{movie_title}'.")
        return None
    if not movie_title or not movie_title.strip():
        # print(f"      Info (Segment {segment_num_str}): No movie title provided for OMDB lookup for '{movie_title}'.")
        return None
    safe_title = urllib.parse.quote_plus(movie_title)
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&s={safe_title}"
    if year:
        url += f"&y={year}"
    try:
        # print(f"      Querying OMDB for '{movie_title}' (Year: {year or 'Any'})...")
        response = requests.get(url, timeout=7)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "True":
            if "Search" in data and data["Search"]:
                if year:
                    for item in data["Search"]:
                        if item.get("Year") == str(year) or str(year) in item.get("Year",""):
                            imdb_id = item.get("imdbID")
                            if imdb_id:
                                # print(f"      OMDB found (match by year): {imdb_id} for '{movie_title}' (OMDB Title: {item.get('Title')})")
                                return imdb_id
                imdb_id = data["Search"][0].get("imdbID")
                if imdb_id:
                    # print(f"      OMDB found (top result): {imdb_id} for '{movie_title}' (OMDB Title: {data['Search'][0].get('Title')})")
                    return imdb_id
            elif "imdbID" in data:
                imdb_id = data.get("imdbID")
                # print(f"      OMDB found (direct title match): {imdb_id} for '{movie_title}'")
                return imdb_id
        # elif data.get("Error"):
            # print(f"      OMDB API error for '{movie_title}': {data['Error']}")
        # else:
            # print(f"      No results or no IMDb ID from OMDB for '{movie_title}'.")
        return None
    except requests.exceptions.Timeout: # print(f"      OMDB API request timed out for '{movie_title}'.")
        pass
    except requests.exceptions.RequestException: # print(f"      Error calling OMDB API for '{movie_title}': {e}")
        pass
    except Exception: # print(f"      Unexpected error during OMDB lookup for '{movie_title}': {e}")
        pass
    return None

def search_tmdb_for_movie_id(movie_title, year=None, segment_num_str="TMDB Search"):
    # This function might still be useful for related movies if only title/year is known
    if not TMDB_API_KEY: return None, None
    if not movie_title or not movie_title.strip():
        return None, None
    safe_title = urllib.parse.quote_plus(movie_title)
    url = f"https://api.themoviedb.org/3/search/movie?query={safe_title}&include_adult=false&language=en-US&page=1"
    year_to_query_tmdb = None
    if year:
        year_str = str(year).strip()
        if year_str.isdigit() and len(year_str) == 4:
            year_to_query_tmdb = year_str
            url += f"&primary_release_year={year_to_query_tmdb}"

    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}
    try:
        # print(f"      Querying TMDB search for '{movie_title}' (Year: {year_to_query_tmdb if year_to_query_tmdb else 'Any'})...")
        response = requests.get(url, headers=headers, timeout=7)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            target_title_lower = movie_title.lower()
            best_match = None
            # More careful matching
            for result in data["results"]:
                tmdb_title_result_lower = result.get("title", "").lower()
                tmdb_year_result_api = result.get("release_date", "N/A")[:4]
                if year_to_query_tmdb and tmdb_year_result_api == year_to_query_tmdb and tmdb_title_result_lower == target_title_lower:
                    best_match = result
                    break
                if not best_match and tmdb_title_result_lower == target_title_lower: # Exact title match, different year or no year query
                    best_match = result
                if not best_match and year_to_query_tmdb and tmdb_year_result_api == year_to_query_tmdb and target_title_lower in tmdb_title_result_lower : # Year match, title substring
                    best_match = result
            if not best_match: # Fallback to first result if no good match found
                best_match = data["results"][0]
                # print(f"      TMDB search: No highly confident match for '{movie_title}'. Using top TMDB result.")

            tmdb_id = best_match.get("id")
            # tmdb_title_result = best_match.get("title")
            tmdb_year_result_found = best_match.get("release_date", "N/A")[:4]
            # print(f"      TMDB search selected: '{tmdb_title_result}' ({tmdb_year_result_found}), TMDB ID: {tmdb_id}")
            # if year_to_query_tmdb and tmdb_year_result_found != "N/A" and tmdb_year_result_found != year_to_query_tmdb:
            #      print(f"        Warning: TMDB result year '{tmdb_year_result_found}' differs from query year '{year_to_query_tmdb}'.")
            return tmdb_id, tmdb_year_result_found
    except Exception: # print(f"      Error/Timeout during TMDB search for '{movie_title}': {e}")
        pass
    return None, None

def get_imdb_id_from_tmdb_details(tmdb_movie_id, movie_title_for_log, segment_num_str="TMDB Details"):
    if not TMDB_API_KEY or not tmdb_movie_id: return None
    url = f"https://api.themoviedb.org/3/movie/{tmdb_movie_id}/external_ids"
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}
    try:
        # print(f"      Querying TMDB external IDs for TMDB ID: {tmdb_movie_id} ('{movie_title_for_log}')...")
        response = requests.get(url, headers=headers, timeout=7)
        response.raise_for_status()
        data = response.json()
        imdb_id = data.get("imdb_id")
        if imdb_id and imdb_id.startswith("tt"):
            # print(f"      TMDB details found IMDb ID: {imdb_id} for TMDB ID {tmdb_movie_id}")
            return imdb_id
        # print(f"      No IMDb ID found in TMDB external IDs for TMDB ID {tmdb_movie_id}. Found: {data}")
    except Exception: # print(f"      Error/Timeout during TMDB external IDs lookup for TMDB ID {tmdb_movie_id}: {e}")
        pass
    return None

def fetch_master_imdb_id(title_or_tmdb_id, year_hint=None, is_tmdb_id=False, segment_num_str="MasterFetch"):
    # print(f"      MasterFetch IMDb ID for: '{title_or_tmdb_id}' (Year hint: {year_hint or 'N/A'}, Is TMDB ID: {is_tmdb_id})...")
    imdb_id = None

    if is_tmdb_id and title_or_tmdb_id: # If we have a TMDB ID, use it directly
        imdb_id = get_imdb_id_from_tmdb_details(title_or_tmdb_id, f"TMDB_ID_{title_or_tmdb_id}", f"{segment_num_str}-TMDB-ExtID")
        if imdb_id: return imdb_id
        # Fall through to search by title if TMDB ID gave no IMDb ID (e.g. TMDB ID was actually a title)

    # If it was not a TMDB ID, or if TMDB ID lookup failed, title_or_tmdb_id is treated as a title
    title = str(title_or_tmdb_id)
    valid_year_for_api = None
    if year_hint:
        year_str = str(year_hint).strip()
        if year_str.isdigit() and len(year_str) == 4:
            valid_year_for_api = year_str

    # Try OMDB with title and year
    if title and valid_year_for_api:
        imdb_id = get_imdb_id_from_omdb(title, valid_year_for_api, f"{segment_num_str}-OMDB-TY")
        if imdb_id: return imdb_id

    # Try TMDB search by title and year, then get IMDb ID from TMDB details
    if title and valid_year_for_api:
        tmdb_id_from_search, _ = search_tmdb_for_movie_id(title, valid_year_for_api, f"{segment_num_str}-TMDB-Search-TY")
        if tmdb_id_from_search:
            imdb_id = get_imdb_id_from_tmdb_details(tmdb_id_from_search, title, f"{segment_num_str}-TMDB-ExtID-TY")
            if imdb_id: return imdb_id

    # Try OMDB with title only
    if title:
        imdb_id = get_imdb_id_from_omdb(title, None, f"{segment_num_str}-OMDB-T")
        if imdb_id: return imdb_id

    # Try TMDB search by title only, then get IMDb ID
    if title:
        tmdb_id_from_search_title_only, _ = search_tmdb_for_movie_id(title, None, f"{segment_num_str}-TMDB-Search-T")
        if tmdb_id_from_search_title_only:
            imdb_id = get_imdb_id_from_tmdb_details(tmdb_id_from_search_title_only, title, f"{segment_num_str}-TMDB-ExtID-T")
            if imdb_id: return imdb_id
    if not is_tmdb_id : # Only print fail if initial input was not supposed to be a direct TMDB ID that failed
      pass
        # print(f"      MasterFetch: Failed to find IMDb ID for '{title}' (Year hint: {year_hint or 'N/A'}) after all attempts.")
    return None


def fetch_raw_character_actor_list_from_tmdb(tmdb_movie_id_for_chars, movie_title_for_log, max_chars, segment_num_str="TMDBRawChars"):
    if not TMDB_API_KEY:
        print(f"      Info (Segment {segment_num_str}): TMDB_API_KEY not set. Skipping TMDB raw character lookup.")
        return None, None
    if not tmdb_movie_id_for_chars:
        print(f"      Info (Segment {segment_num_str}): No TMDB movie ID for raw character lookup for '{movie_title_for_log}'.")
        return None, None

    url = f"https://api.themoviedb.org/3/movie/{tmdb_movie_id_for_chars}/credits?language=en-US"
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}
    raw_char_actor_list = []
    try:
        print(f"      Querying TMDB Credits for raw char/actor list (TMDB ID: {tmdb_movie_id_for_chars}, '{movie_title_for_log}')...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "cast" in data and data["cast"]:
            sorted_cast = sorted(data["cast"], key=lambda x: x.get("order", float('inf')))
            for cast_member in sorted_cast[:max_chars]:
                char_name_raw = cast_member.get("character", "").strip()
                actor_name_raw = cast_member.get("name", "").strip()
                person_id = cast_member.get("id")

                if not char_name_raw or len(char_name_raw) < 2 or len(char_name_raw) > 70 :
                    # print(f"        Skipping TMDB character '{char_name_raw}' (actor: {actor_name_raw}) - problematic name.")
                    continue
                if not actor_name_raw:
                    #  print(f"        Skipping TMDB character '{char_name_raw}' - missing actor name.")
                     continue
                if not person_id: # Crucial for image fetching
                    # print(f"        Skipping TMDB character '{char_name_raw}' (actor: {actor_name_raw}) - missing person_id (actor's TMDB ID).")
                    continue

                raw_char_actor_list.append({
                    "tmdb_character_name": char_name_raw,
                    "tmdb_actor_name": actor_name_raw,
                    "tmdb_person_id": person_id
                })

            if raw_char_actor_list:
                print(f"      TMDB Credits: Retrieved {len(raw_char_actor_list)} raw character/actor entries for '{movie_title_for_log}'.")
                print(f"      TMDB Credits: {raw_char_actor_list} ")
                raw_char_actor_list_yaml = yaml.dump(raw_char_actor_list, sort_keys=False, allow_unicode=True, indent=2)
                return raw_char_actor_list, raw_char_actor_list_yaml
            else:
                print(f"      TMDB Credits: No valid raw character/actor data extracted for '{movie_title_for_log}' (TMDB ID: {tmdb_movie_id_for_chars}).")
                return None, None
        else:
            print(f"      TMDB Credits: No 'cast' array or empty cast for '{movie_title_for_log}' (TMDB ID: {tmdb_movie_id_for_chars}). Response: {str(data)[:200]}")
            return None, None
    except Exception as e:
        print(f"      Error/Timeout during TMDB Credits lookup (raw chars) for '{movie_title_for_log}' (TMDB ID: {tmdb_movie_id_for_chars}): {e}")
    return None, None

def fetch_and_save_character_image(person_id, person_name_for_log, segment_num_str="CharImg"):
    if not TMDB_API_KEY:
        # print(f"      Info (Segment {segment_num_str}): TMDB_API_KEY not set. Skipping image download for person ID {person_id}.")
        return None
    if not person_id:
        # print(f"      Info (Segment {segment_num_str}): No person ID provided. Skipping image download.")
        return None

    if not os.path.exists(CHARACTER_IMAGE_SAVE_PATH):
        try:
            os.makedirs(CHARACTER_IMAGE_SAVE_PATH)
            # print(f"      Created directory: {CHARACTER_IMAGE_SAVE_PATH}")
        except OSError as e:
            print(f"      Error (Segment {segment_num_str}): Could not create directory {CHARACTER_IMAGE_SAVE_PATH}: {e}")
            return None

    url = f"https://api.themoviedb.org/3/person/{person_id}/images"
    headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_API_KEY}"}

    try:
        # print(f"      Querying TMDB for images of person ID {person_id} ('{person_name_for_log}')...")
        response = requests.get(url, headers=headers, timeout=7)
        response.raise_for_status()
        data = response.json()

        if data.get("profiles") and len(data["profiles"]) > 0:
            first_profile = data["profiles"][0]
            file_path_suffix = first_profile.get("file_path")

            if file_path_suffix:
                image_url = f"{TMDB_IMAGE_BASE_URL}{TMDB_IMAGE_SIZE}{file_path_suffix}"
                _, file_extension = os.path.splitext(file_path_suffix)
                if not file_extension: file_extension = ".jpg"

                local_image_filename = f"{person_id}{file_extension}"
                local_image_full_path = os.path.join(CHARACTER_IMAGE_SAVE_PATH, local_image_filename)

                # print(f"      Downloading image from {image_url} to {local_image_full_path}...")
                img_response = requests.get(image_url, stream=True, timeout=10)
                img_response.raise_for_status()

                with open(local_image_full_path, 'wb') as f_img:
                    shutil.copyfileobj(img_response.raw, f_img)
                # print(f"      Successfully saved image: {local_image_full_path}")
                return local_image_full_path
            # else:
                # print(f"      No file_path found for first profile image of person ID {person_id}.")
        # else:
            # print(f"      No profile images found for person ID {person_id} ('{person_name_for_log}').")
    except requests.exceptions.Timeout: # print(f"      Timeout (Segment {segment_num_str}): TMDB image request for person ID {person_id} timed out.")
        pass
    except requests.exceptions.RequestException: # print(f"      Error (Segment {segment_num_str}): Calling TMDB images API for person ID {person_id}: {e}")
        pass
    except Exception: # print(f"      Unexpected error (Segment {segment_num_str}): During image fetch for person ID {person_id}: {e}")
        pass
    return None


# --- LLM Response and Parsing Functions ---
def get_llm_response(messages_history, max_tokens, temperature=0.3, attempt_yaml_cleanup=True):
    print(f"   Sending request to LLM (max_tokens: {max_tokens}, temp: {temperature})...")
    try:
        completion = client.chat.completions.create(
            model=MODEL_ID, messages=messages_history, temperature=temperature, max_tokens=max_tokens)
        response_content = completion.choices[0].message.content.strip()
        if attempt_yaml_cleanup:
            if response_content.startswith("```yaml"):
                response_content = response_content[7:]
                if response_content.endswith("```"):
                    response_content = response_content[:-3].strip()
            elif response_content.startswith("yaml"):
                 response_content = response_content[4:].lstrip()
            if response_content.endswith("```"):
                 response_content = response_content[:-3].strip()
        # print(f"   LLM responded with ~{len(response_content.split())} words.")
        return response_content
    except Exception as e: print(f"Error communicating with LLM: {e}")
    return None

def parse_llm_response_segment(yaml_segment_text, expected_keys_for_segment, segment_num_str, is_call_1=False, given_title_c1=None, given_year_c1=None):
    try:
        data = yaml.safe_load(yaml_segment_text)
        if not isinstance(data, dict):
            print(f"      Warning (Segment {segment_num_str}): Parsed YAML not a dictionary. Content: {str(yaml_segment_text)[:200]}")
            return None

        parsed_data = {}
        if is_call_1:
            # Explicitly define the key strings for title and year based on the global constant
            # This ensures we are using the string values from EXPECTED_YAML_KEYS_CALL_1
            actual_movie_title_key = EXPECTED_YAML_KEYS_CALL_1[0] # Should be "movie_title"
            actual_movie_year_key = EXPECTED_YAML_KEYS_CALL_1[1]   # Should be "movie_year"

            # Debug prints to verify key types (optional, can be removed after checking)
            # print(f"DEBUG C1: actual_movie_title_key = '{actual_movie_title_key}' (type: {type(actual_movie_title_key)})")
            # print(f"DEBUG C1: actual_movie_year_key = '{actual_movie_year_key}' (type: {type(actual_movie_year_key)})")

            llm_title = data.get(actual_movie_title_key)
            llm_year = data.get(actual_movie_year_key)

            if not (isinstance(llm_title, str) and llm_title.strip()):
                print(f"      Error (Segment {segment_num_str} - Call 1): Key '{actual_movie_title_key}' missing, null, or not a string in LLM response. LLM value: '{llm_title}'")
                return None
            if str(llm_title).strip().lower() != str(given_title_c1).strip().lower():
                print(f"      Error (Segment {segment_num_str} - Call 1): LLM returned title '{llm_title}' which DOES NOT MATCH given title '{given_title_c1}'.")
                return None

            parsed_data[actual_movie_title_key] = str(given_title_c1).strip() # Store the KNOWN GIVEN title

            year_validated_from_llm = None
            if llm_year is not None:
                year_str_temp = str(llm_year).strip()
                if year_str_temp.isdigit() and len(year_str_temp) == 4:
                    year_validated_from_llm = year_str_temp
                # else:
                    # print(f"      Warning (Segment {segment_num_str} - Call 1): LLM year '{llm_year}' invalid format for '{given_title_c1}'. Using GIVEN year '{given_year_c1}'.")

            if year_validated_from_llm and str(year_validated_from_llm) != str(given_year_c1):
                print(f"      Warning (Segment {segment_num_str} - Call 1): LLM year '{year_validated_from_llm}' for '{given_title_c1}' differs from GIVEN year '{given_year_c1}'. Using GIVEN year.")
            parsed_data[actual_movie_year_key] = str(given_year_c1) # Store the KNOWN GIVEN year

        # Common parsing for all calls (iterating over `expected_keys_for_segment`)
        for key_str in expected_keys_for_segment: # key_str should be a string from the list
            # Skip title and year for Call 1 as they are already handled with given values
            if is_call_1 and key_str in (EXPECTED_YAML_KEYS_CALL_1[0], EXPECTED_YAML_KEYS_CALL_1[1]):
                continue

            llm_value = data.get(key_str)
            if llm_value is not None:
                if isinstance(llm_value, (dict, list)):
                    parsed_data[key_str] = llm_value
                else:
                    parsed_data[key_str] = str(llm_value).strip()
                    if parsed_data[key_str].lower() == 'null': # Convert 'null' strings to None
                        parsed_data[key_str] = None
            else:
                parsed_data[key_str] = None
                # print(f"      Info (Segment {segment_num_str}): Expected key '{key_str}' not found in LLM output, set to None.")

        # Validation for Call 2 structure (character_list items)
        if expected_keys_for_segment == EXPECTED_YAML_KEYS_ENRICH_REL_CALL:
            char_list_from_llm = parsed_data.get("character_list")
            if isinstance(char_list_from_llm, list) and char_list_from_llm:
                first_char = char_list_from_llm[0] # Check the first item's structure
                required_char_keys = ["name", "actor_name", "tmdb_person_id", "description", "group"]
                if not (isinstance(first_char, dict) and all(k in first_char for k in required_char_keys)):
                    print(f"      Warning (Segment {segment_num_str}): Enriched 'character_list' item structure error (e.g., missing 'name', 'tmdb_person_id'). First item: {str(first_char)[:150]}")
                    # Potentially invalidate the list if structure is critical
                    # parsed_data["character_list"] = None
            elif char_list_from_llm is not None: # Present but not a list or empty
                print(f"      Warning (Segment {segment_num_str}): 'character_list' from LLM not a valid list. Value: {str(char_list_from_llm)[:100]}")
                parsed_data["character_list"] = None # Invalidate if not a list

            if parsed_data.get("relationships") is not None and not isinstance(parsed_data.get("relationships"), list):
                print(f"      Warning (Segment {segment_num_str}): 'relationships' from LLM present but not a list. Value: {str(parsed_data.get('relationships'))[:100]}")
                parsed_data["relationships"] = [] # Default to empty list if malformed
        return parsed_data

    except yaml.YAMLError as ye:
        print(f"      Critical (Segment {segment_num_str}): YAML parsing error: {ye}. Text: {str(yaml_segment_text)[:300]}")
    except Exception as e:
        print(f"      Critical (Segment {segment_num_str}): Unexpected error parsing segment: {e}. Text: {str(yaml_segment_text)[:300]}")
        # Optionally, re-raise if you want the script to stop on such errors for debugging
        # raise
    return None

def extract_movie_title_from_llm_or_parse(yaml_segment_text, segment_num_str):
    # This function is less critical now for *discovering* title from C1,
    # but can be used to *verify* LLM returned the title it was given.
    # However, parse_llm_response_segment for C1 now does this verification.
    # Keeping it in case it's needed for other recovery scenarios, but largely superseded for C1.
    try:
        data = yaml.safe_load(yaml_segment_text)
        if isinstance(data, dict):
            parsed_title = data.get(CRITICAL_KEY_FOR_CLEAN_OUTPUT) # CRITICAL_KEY_FOR_CLEAN_OUTPUT is 'movie_title'
            if parsed_title and isinstance(parsed_title, str):
                title = parsed_title.strip()
                if title and len(title) > 1 and len(title) < 150:
                    # print(f"      Direct parse extracted title (Segment {segment_num_str}): '{title}'")
                    return title
    except Exception: # print(f"      Direct parse for title failed (Segment {segment_num_str}): {e}")
        pass
    # print(f"      Could not extract title from segment {segment_num_str} via direct parse.")
    return None


# --- Helper functions to load/save movie data from/to YAML ---
def load_full_movie_data_from_yaml(filename):
    movie_data_list = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict) and entry.get(CRITICAL_KEY_FOR_CLEAN_OUTPUT):
                            movie_data_list.append(entry)
            print(f"Loaded {len(movie_data_list)} existing movie entries from '{filename}'.")
        except Exception as e:
            print(f"Error loading full movie data from '{filename}': {e}. Starting empty.")
    else:
        print(f"Clean movie database file '{filename}' not found. Starting empty.")
    return movie_data_list

def save_movie_data_to_yaml(movie_data_list, filename):
    final_data_to_write = []
    seen_titles_for_dedup = set()
    for item in reversed(movie_data_list):
        title_to_check_raw = item.get(CRITICAL_KEY_FOR_CLEAN_OUTPUT)
        title_to_check = str(title_to_check_raw).lower().strip() if title_to_check_raw else None

        if title_to_check and title_to_check not in seen_titles_for_dedup:
            ordered_item = {}
            for key in EXPECTED_YAML_KEYS_FINAL: # Ensures all keys are present and in order
                ordered_item[key] = item.get(key)

            if isinstance(ordered_item.get("character_list"), list):
                processed_char_list = []
                for char_item_raw in ordered_item["character_list"]:
                    if isinstance(char_item_raw, dict):
                        processed_char_item = {ckey: char_item_raw.get(ckey) for ckey in CHARACTER_LIST_ITEM_KEYS}
                        processed_char_list.append(processed_char_item)
                    else: # Should not happen if parsing is correct
                        processed_char_list.append(char_item_raw)
                ordered_item["character_list"] = processed_char_list

            final_data_to_write.append(ordered_item)
            seen_titles_for_dedup.add(title_to_check)

    final_data_to_write.reverse()
    try:
        with open(filename, 'w', encoding='utf-8') as f_yaml:
            # Using Dumper=yaml.SafeDumper is good practice if you don't need to dump custom objects.
            # For literal block style for multiline strings, a custom Dumper or more complex setup is needed.
            # Sticking to SafeDumper for now for simplicity. YAML output might not have perfect literal blocks.
            yaml.dump(final_data_to_write, f_yaml, sort_keys=False, indent=2, allow_unicode=True, Dumper=yaml.SafeDumper)
        # print(f"      Successfully wrote {len(final_data_to_write)} total entries to '{filename}'.")
    except Exception as e:
        print(f"      Error writing to YAML file '{filename}': {e}")


def deduplicate_and_normalize_relationships(llm_enriched_character_list, relationships_data_from_llm, segment_num_str):
    if not isinstance(llm_enriched_character_list, list) or not llm_enriched_character_list:
        # print(f"      Info (Segment {segment_num_str}): LLM-enriched character_list empty/invalid. Cannot normalize relationships.")
        return relationships_data_from_llm or []
    if not isinstance(relationships_data_from_llm, list):
        # print(f"      Info (Segment {segment_num_str}): LLM relationships data not a list. Returning empty list.")
        return []

    name_map = {}
    for char_entry in llm_enriched_character_list:
        if isinstance(char_entry, dict) and "name" in char_entry:
            canonical_name = str(char_entry.get("name", "")).strip()
            if canonical_name:
                name_map[canonical_name.lower()] = canonical_name
                aliases = char_entry.get("aliases")
                if isinstance(aliases, list):
                    for alias in aliases:
                        alias_str = str(alias).strip()
                        if alias_str: name_map[alias_str.lower()] = canonical_name
                elif isinstance(aliases, str) and aliases.strip(): # Handle if LLM gives a single alias string
                    name_map[aliases.strip().lower()] = canonical_name

    if not name_map: # Should not happen if llm_enriched_character_list is valid
        # print(f"      Warning (Segment {segment_num_str}): Name map from enriched character list empty. Cannot normalize relationships well.")
        return relationships_data_from_llm

    unique_relationships = []
    seen_mutual_pairs = set() # For handling undirected (mutual) relationships if `directed: false`
    seen_directed_pairs = set() # For handling directed relationships

    for rel in relationships_data_from_llm:
        if not (isinstance(rel, dict) and "source" in rel and "target" in rel):
            # print(f"      Warning (Segment {segment_num_str}): Malformed relationship entry skipped: {str(rel)[:100]}")
            continue
        original_source_llm = str(rel.get("source", "")).strip()
        original_target_llm = str(rel.get("target", "")).strip()
        source_norm = name_map.get(original_source_llm.lower())
        target_norm = name_map.get(original_target_llm.lower())

        if not source_norm:
            # print(f"      Warning (Segment {segment_num_str}): Rel source '{original_source_llm}' NOT FOUND in enriched list. Skipping.")
            continue
        if not target_norm:
            # print(f"      Warning (Segment {segment_num_str}): Rel target '{original_target_llm}' NOT FOUND in enriched list. Skipping.")
            continue
        if source_norm == target_norm:
            # print(f"      Warning (Segment {segment_num_str}): Rel source and target same ('{source_norm}'). Skipping.")
            continue

        rel["source"] = source_norm # Normalize names
        rel["target"] = target_norm

        is_directed = rel.get("directed", True) # Default to True if missing
        if not isinstance(is_directed, bool):
            # print(f"      Warning (Segment {segment_num_str}): Rel 'directed' for {source_norm}-{target_norm} not bool ('{rel.get('directed')}'). Defaulting to True.")
            is_directed = True
            rel["directed"] = True # Correct the data

        if is_directed is False: # Undirected/mutual
            pair_key = tuple(sorted((source_norm.lower(), target_norm.lower())))
            if pair_key not in seen_mutual_pairs:
                unique_relationships.append(rel)
                seen_mutual_pairs.add(pair_key)
                seen_directed_pairs.add((source_norm.lower(), target_norm.lower())) # Also add to directed to prevent a->b if {a,b} exists
                seen_directed_pairs.add((target_norm.lower(), source_norm.lower()))
            # else:
                # print(f"      Deduplicating mutual relationship: {source_norm} and {target_norm}")
        else: # Directed
            pair_key_directed = (source_norm.lower(), target_norm.lower())
            # Check if this directed pair or its mutual equivalent already exists
            mutual_equivalent_key = tuple(sorted((source_norm.lower(), target_norm.lower())))

            if pair_key_directed not in seen_directed_pairs and mutual_equivalent_key not in seen_mutual_pairs:
                unique_relationships.append(rel)
                seen_directed_pairs.add(pair_key_directed)
            # else:
                # print(f"      Deduplicating directed relationship: {source_norm} -> {target_norm} (or mutual exists)")

    # if len(unique_relationships) < len(relationships_data_from_llm):
        # print(f"      Normalized/deduplicated relationships from {len(relationships_data_from_llm)} to {len(unique_relationships)}.")
    # else:
        # print(f"      Relationships processed for segment {segment_num_str}. Count: {len(unique_relationships)}.")
    return unique_relationships


def run_movie_agent():
    print(f"Starting movie data generation with model: {MODEL_ID} (Logged as: {REQUESTED_MODEL_NAME_FOR_LOG})\n")
    if not OMDB_API_KEY: print("WARNING: OMDB_API_KEY not set. OMDB lookups will be reduced.")
    if not TMDB_API_KEY: print("CRITICAL WARNING: TMDB_API_KEY not set. Core movie fetching and character data will fail. Exiting.")
    if not TMDB_API_KEY: return # Exit if TMDB key is missing
    print("")

    if not os.path.exists(CHARACTER_IMAGE_SAVE_PATH):
        try:
            os.makedirs(CHARACTER_IMAGE_SAVE_PATH)
            print(f"Created character image directory: {CHARACTER_IMAGE_SAVE_PATH}")
        except OSError as e:
            print(f"ERROR: Could not create character image directory {CHARACTER_IMAGE_SAVE_PATH}: {e}. Images may not be saved.")

    with open(RAW_LOG_FILENAME, 'a', encoding='utf-8') as f_raw_log:
        f_raw_log.write(f"\n\n===== NEW SESSION: {time.asctime()} =====\n")
        f_raw_log.write(f"Movie Data Generation Raw Log - Model: {REQUESTED_MODEL_NAME_FOR_LOG}\n")
        f_raw_log.write("="*40 + "\n\n")

        all_movie_data_to_persist = load_full_movie_data_from_yaml(CLEAN_YAML_OUTPUT_FILENAME)
        processed_movie_titles_lower_set = set()
        processed_movie_titles_original_case_list = [] # For logging consistency

        for entry in all_movie_data_to_persist:
            title_raw = entry.get(CRITICAL_KEY_FOR_CLEAN_OUTPUT)
            if title_raw and isinstance(title_raw, str):
                title_str = title_raw.strip()
                title_lower = title_str.lower()
                if title_str and title_lower not in processed_movie_titles_lower_set:
                    processed_movie_titles_lower_set.add(title_lower)
                    processed_movie_titles_original_case_list.append(title_str) # Keep for sorted log

        new_movies_added_this_session = 0
        session_attempt_count = 0 # Used for segment IDs
        current_tmdb_page = 1
        exhausted_tmdb_pages = False

        # Main loop to fetch desired number of new movies
        while new_movies_added_this_session < NUM_NEW_MOVIES_TO_FETCH_THIS_SESSION and not exhausted_tmdb_pages:
            if current_tmdb_page > MAX_TMDB_TOP_RATED_PAGES_TO_CHECK:
                print(f"Checked {MAX_TMDB_TOP_RATED_PAGES_TO_CHECK} pages of TMDB Top Rated. Ending session.")
                f_raw_log.write(f"Checked {MAX_TMDB_TOP_RATED_PAGES_TO_CHECK} TMDB pages. Ending movie search.\n")
                exhausted_tmdb_pages = True
                break

            print(f"\n--- Fetching TMDB Top Rated Page: {current_tmdb_page} ---")
            f_raw_log.write(f"--- Fetching TMDB Top Rated Page: {current_tmdb_page} ---\n")
            tmdb_page_data = fetch_top_rated_movies_from_tmdb(current_tmdb_page)

            if not tmdb_page_data or not tmdb_page_data.get("results"):
                print(f"Failed to fetch or no results on TMDB Top Rated page {current_tmdb_page}. Trying next page or ending.")
                f_raw_log.write(f"Failed to fetch or no results on TMDB Top Rated page {current_tmdb_page}.\n")
                if current_tmdb_page >= tmdb_page_data.get("total_pages", current_tmdb_page) : exhausted_tmdb_pages = True
                current_tmdb_page += 1
                time.sleep(1) # Pause before fetching next page
                continue

            movies_on_this_page = tmdb_page_data["results"]
            found_new_movie_on_this_page = False

            for tmdb_movie in movies_on_this_page:
                session_attempt_count += 1 # Increment for each potential movie processed
                main_seg_id = f"P{current_tmdb_page:02d}A{session_attempt_count:03d}" # Page and Attempt based ID

                tmdb_provided_title = tmdb_movie.get("title", "").strip()
                tmdb_provided_release_date = tmdb_movie.get("release_date", "") # YYYY-MM-DD
                tmdb_provided_movie_id = tmdb_movie.get("id") # Integer TMDB ID

                if not tmdb_provided_title or not tmdb_provided_movie_id:
                    print(f"      Skipping TMDB entry with missing title or ID: {str(tmdb_movie)[:100]}")
                    f_raw_log.write(f"Segment {main_seg_id} - Skipped TMDB entry (missing title/ID): {tmdb_provided_title}\n")
                    continue

                tmdb_provided_year = tmdb_provided_release_date[:4] if tmdb_provided_release_date and len(tmdb_provided_release_date) >= 4 else None
                if not (tmdb_provided_year and tmdb_provided_year.isdigit() and len(tmdb_provided_year) == 4):
                    print(f"      Skipping TMDB entry '{tmdb_provided_title}' due to invalid/missing year: {tmdb_provided_release_date}")
                    f_raw_log.write(f"Segment {main_seg_id} - Skipped TMDB entry (invalid year): {tmdb_provided_title} ({tmdb_provided_release_date})\n")
                    continue


                normalized_tmdb_title_lower = tmdb_provided_title.lower()
                if normalized_tmdb_title_lower in processed_movie_titles_lower_set:
                    # print(f"      Movie '{tmdb_provided_title}' already processed. Skipping.")
                    f_raw_log.write(f"Segment {main_seg_id} - Already processed: {tmdb_provided_title}\n")
                    continue

                # ---- NEW MOVIE FOUND - START PROCESSING ----
                found_new_movie_on_this_page = True
                print(f"\n--- Processing New Movie from TMDB Top Rated (Page {current_tmdb_page}): '{tmdb_provided_title}' ({tmdb_provided_year}) TMDB_ID: {tmdb_provided_movie_id} ---")
                f_raw_log.write(f"--- Processing New Movie: '{tmdb_provided_title}' ({tmdb_provided_year}) TMDB_ID: {tmdb_provided_movie_id} ---\n")

                # --- LLM Call 1 (Generate data for the GIVEN movie) ---
                segment_str_call_1 = f"{main_seg_id}-C1Gen"
                prompt1_user_content = MOVIE_PROMPT_CALL_1_TEMPLATE.format(
                    movie_title_from_tmdb=tmdb_provided_title,
                    movie_year_from_tmdb=tmdb_provided_year,
                    expected_title_key=EXPECTED_YAML_KEYS_CALL_1, # movie_title
                    expected_year_key=EXPECTED_YAML_KEYS_CALL_1,  # movie_year
                    num_call_1_keys=len(EXPECTED_YAML_KEYS_CALL_1)
                )
                llm_messages_call_1 = [{"role": "system", "content": "You are an assistant that provides movie information in strict YAML format for a given movie."}, {"role": "user", "content": prompt1_user_content}]
                raw_llm_response_call_1 = get_llm_response(llm_messages_call_1, MAX_TOKENS_CALL_1)

                if not raw_llm_response_call_1:
                    print(f"LLM Call 1 (Data Gen) FAILED to respond for '{tmdb_provided_title}'. Skipping this movie.")
                    f_raw_log.write(f"Segment {segment_str_call_1} - LLM FAILED TO RESPOND for '{tmdb_provided_title}'.\n\n")
                    processed_movie_titles_lower_set.add(normalized_tmdb_title_lower) # Mark as attempted to avoid re-processing in this session
                    time.sleep(5); continue # Break from this movie, try next on TMDB page

                f_raw_log.write(f"Segment {segment_str_call_1} Raw LLM Output:\n{raw_llm_response_call_1}\n\n")
                parsed_data_call_1 = parse_llm_response_segment(raw_llm_response_call_1, EXPECTED_YAML_KEYS_CALL_1, segment_str_call_1,
                                                              is_call_1=True, given_title_c1=tmdb_provided_title, given_year_c1=tmdb_provided_year)

                if not parsed_data_call_1 or parsed_data_call_1.get(CRITICAL_KEY_FOR_CLEAN_OUTPUT) != tmdb_provided_title:
                    print(f"LLM Call 1 (Data Gen) FAILED to parse correctly or returned wrong title for '{tmdb_provided_title}'. Skipping.")
                    f_raw_log.write(f"Segment {segment_str_call_1} - PARSE FAIL or WRONG TITLE for '{tmdb_provided_title}'. LLM_Title: {parsed_data_call_1.get(CRITICAL_KEY_FOR_CLEAN_OUTPUT) if parsed_data_call_1 else 'N/A'}\n\n")
                    processed_movie_titles_lower_set.add(normalized_tmdb_title_lower)
                    time.sleep(2); continue

                # At this point, parsed_data_call_1 contains info for tmdb_provided_title and tmdb_provided_year
                # authoritative_movie_year is tmdb_provided_year
                # tmdb_movie_id for character fetching is tmdb_provided_movie_id

                character_profile_text_c1 = parsed_data_call_1.get("character_profile", "Character profile not provided by Call 1.")
                # critical_reception_text_c1 = parsed_data_call_1.get("critical_reception", "Critical reception not provided by Call 1.") # For C3

                # --- TMDB Character Fetch (using tmdb_provided_movie_id) ---
                segment_str_tmdb_raw_chars = f"{main_seg_id}-TMDBRawChars"
                raw_tmdb_chars_list, raw_tmdb_chars_yaml = fetch_raw_character_actor_list_from_tmdb(
                    tmdb_provided_movie_id, tmdb_provided_title, MAX_CHARACTERS_FROM_TMDB, segment_str_tmdb_raw_chars
                )

                if not raw_tmdb_chars_list or not raw_tmdb_chars_yaml:
                    print(f"CRITICAL: Failed to fetch raw character/actor list from TMDB for '{tmdb_provided_title}' (ID: {tmdb_provided_movie_id}). Aborting this movie.")
                    f_raw_log.write(f"Segment {segment_str_tmdb_raw_chars} - TMDB RAW CHARACTER FETCH FAIL for '{tmdb_provided_title}'. MOVIE ABORTED.\n\n")
                    processed_movie_titles_lower_set.add(normalized_tmdb_title_lower)
                    time.sleep(2); continue

                # --- LLM Call 2 (Enrich Characters & Generate Relationships) ---
                segment_str_enrich_rel = f"{main_seg_id}-C2EnrichRel"
                print(f"--- LLM Call 2 (Enrich/Rel) for '{tmdb_provided_title}' ({tmdb_provided_year}) ---")
                f_raw_log.write(f"--- LLM Call 2 (Enrich/Rel) for '{tmdb_provided_title}' ({tmdb_provided_year}) ---\n")
                prompt_enrich_rel_user_content = MOVIE_PROMPT_ENRICH_CHARS_AND_RELATIONSHIPS_TEMPLATE.format(
                    movie_title=tmdb_provided_title,
                    movie_year=tmdb_provided_year,
                    raw_tmdb_characters_yaml=raw_tmdb_chars_yaml
                )
                llm_messages_enrich_rel = [{"role": "system", "content": "You enrich character lists and generate relationships in YAML."}, {"role": "user", "content": prompt_enrich_rel_user_content}]
                raw_llm_response_enrich_rel = get_llm_response(llm_messages_enrich_rel, MAX_TOKENS_ENRICH_REL_CALL, temperature=0.4)

                enriched_character_list_from_llm = None
                llm_generated_relationships = []

                if raw_llm_response_enrich_rel:
                    f_raw_log.write(f"Segment {segment_str_enrich_rel} Raw LLM Output:\n{raw_llm_response_enrich_rel}\n\n")
                    parsed_data_enrich_rel = parse_llm_response_segment(raw_llm_response_enrich_rel, EXPECTED_YAML_KEYS_ENRICH_REL_CALL, segment_str_enrich_rel)
                    if parsed_data_enrich_rel:
                        enriched_character_list_from_llm = parsed_data_enrich_rel.get("character_list")
                        llm_generated_relationships = parsed_data_enrich_rel.get("relationships", [])
                        if not enriched_character_list_from_llm : # If list is empty or None after parsing
                             print(f"      Warning (Segment {segment_str_enrich_rel}): Enriched 'character_list' is empty or invalid after parsing.")
                             f_raw_log.write(f"Segment {segment_str_enrich_rel} - Enriched 'character_list' empty/invalid post-parse.\n")
                        # else: print(f"      Parsed enriched character list ({len(enriched_character_list_from_llm)} chars from LLM).")
                        # print(f"      Parsed relationships ({len(llm_generated_relationships)} rels from LLM).")
                    else: # Parse failed
                        print(f"LLM Call 2 (Enrich/Rel) FAILED to parse for '{tmdb_provided_title}'.")
                        f_raw_log.write(f"Segment {segment_str_enrich_rel} - PARSE FAIL.\n\n")
                else: # LLM did not respond
                    print(f"LLM Call 2 (Enrich/Rel) FAILED to respond for '{tmdb_provided_title}'.")
                    f_raw_log.write(f"Segment {segment_str_enrich_rel} - LLM FAILED TO RESPOND.\n\n")

                if not enriched_character_list_from_llm:
                    print(f"CRITICAL: Failed to get valid enriched character list from LLM Call 2 for '{tmdb_provided_title}'. Aborting this movie.")
                    f_raw_log.write(f"Segment {segment_str_enrich_rel} - NO VALID ENRICHED CHARACTER LIST. MOVIE ABORTED.\n\n")
                    processed_movie_titles_lower_set.add(normalized_tmdb_title_lower)
                    time.sleep(2); continue

                # --- Process Character Images and Finalize Character List ---
                segment_str_char_img = f"{main_seg_id}-CharImg"
                final_enriched_character_list_with_images = []
                # print(f"   Processing character images for '{tmdb_provided_title}'...")
                for char_data_llm in enriched_character_list_from_llm:
                    updated_char_data = char_data_llm.copy()
                    person_id_from_llm = char_data_llm.get("tmdb_person_id") # Should be present from LLM
                    char_name_for_log = char_data_llm.get("name", "Unknown Character")
                    image_local_path = None
                    if person_id_from_llm:
                        try:
                           person_id_int = int(person_id_from_llm)
                           image_local_path = fetch_and_save_character_image(person_id_int, char_name_for_log, segment_str_char_img)
                        except ValueError:
                            print(f"      Warning (Segment {segment_str_char_img}): Invalid person_id format '{person_id_from_llm}' for '{char_name_for_log}'. Skipping image.")
                            f_raw_log.write(f"Segment {segment_str_char_img} - Invalid person_id '{person_id_from_llm}' for char '{char_name_for_log}'.\n")
                    # else:
                        # print(f"      Info (Segment {segment_str_char_img}): No tmdb_person_id for character '{char_name_for_log}' from LLM. Skipping image.")
                        # f_raw_log.write(f"Segment {segment_str_char_img} - No tmdb_person_id from LLM for char '{char_name_for_log}'.\n")
                    updated_char_data["image_file"] = image_local_path
                   # final_enriched_character_list_with_images.append(updated_char_data)

                final_normalized_relationships = deduplicate_and_normalize_relationships(
                    final_enriched_character_list_with_images, llm_generated_relationships, segment_str_enrich_rel
                )

                # --- LLM Call 3 (Analytical Data) ---
                segment_str_analytical = f"{main_seg_id}-C3An"
                print(f"--- LLM Call 3 (Analytical) for '{tmdb_provided_title}' ({tmdb_provided_year}) ---")
                f_raw_log.write(f"--- LLM Call 3 (Analytical) for '{tmdb_provided_title}' ({tmdb_provided_year}) ---\n")
                prompt_analytical_user_content = MOVIE_PROMPT_ANALYTICAL_TEMPLATE.format(
                    movie_title_from_call_1=tmdb_provided_title, # Title from TMDB/Call1
                    movie_year_from_call_1=tmdb_provided_year,   # Year from TMDB/Call1
                    # character_profile_from_call_1=character_profile_text_c1, # Not explicitly used in this prompt template
                    # critical_reception_from_call_1=parsed_data_call_1.get("critical_reception"), # Use fresh from C1
                    num_analytical_keys=len(EXPECTED_YAML_KEYS_ANALYTICAL_CALL)
                )
                llm_messages_analytical = [{"role": "system", "content": "You provide analytical movie information in YAML."}, {"role": "user", "content": prompt_analytical_user_content}]
                raw_llm_response_analytical = get_llm_response(llm_messages_analytical, MAX_TOKENS_ANALYTICAL_CALL)

                parsed_data_analytical = None
                if raw_llm_response_analytical:
                    f_raw_log.write(f"Segment {segment_str_analytical} Raw LLM Output:\n{raw_llm_response_analytical}\n\n")
                    parsed_data_analytical = parse_llm_response_segment(raw_llm_response_analytical, EXPECTED_YAML_KEYS_ANALYTICAL_CALL, segment_str_analytical)
                else:
                    print(f"LLM Call 3 (Analytical) FAILED to respond for '{tmdb_provided_title}'.")
                    f_raw_log.write(f"Segment {segment_str_analytical} - LLM FAILED TO RESPOND.\n\n")

                if not parsed_data_analytical:
                    print(f"LLM Call 3 (Analytical) failed to parse or respond for '{tmdb_provided_title}'. Storing with missing analytical data.")
                    f_raw_log.write(f"Segment {segment_str_analytical} - PARSE FAIL or NO RESPONSE.\n\n")
                    parsed_data_analytical = {key: None for key in EXPECTED_YAML_KEYS_ANALYTICAL_CALL} # Ensure keys exist

                # --- Combine, Process, and Save ---
                print(f"   Successfully processed stages for '{tmdb_provided_title}'. Assembling final data.")
                clean_entry = {}
                clean_entry.update(parsed_data_call_1) # Data from Call 1 (should include title, year, profile, reception etc.)
                clean_entry["tmdb_movie_id"] = tmdb_provided_movie_id # Add the TMDB ID of the main movie

                clean_entry["character_list"] = final_enriched_character_list_with_images
                clean_entry["relationships"] = final_normalized_relationships
                clean_entry.update(parsed_data_analytical)

                # Fetch IMDb ID for the main movie (using its TMDB ID preferably)
                main_movie_imdb_id = fetch_master_imdb_id(tmdb_provided_movie_id, year_hint=tmdb_provided_year, is_tmdb_id=True, segment_num_str=f"{main_seg_id}-mainIMDb")
                if not main_movie_imdb_id: # Fallback to title if TMDB ID yielded nothing
                    main_movie_imdb_id = fetch_master_imdb_id(tmdb_provided_title, year_hint=tmdb_provided_year, is_tmdb_id=False, segment_num_str=f"{main_seg_id}-mainIMDbTitle")
                clean_entry[IMDB_ID_KEY] = main_movie_imdb_id
                f_raw_log.write(f"Main IMDb ID for '{tmdb_provided_title}': {main_movie_imdb_id}\n")

                for rel_key in RELATED_MOVIE_KEYS:
                    related_info_raw = clean_entry.get(rel_key) # This comes from LLM Call 1
                    if isinstance(related_info_raw, str) and related_info_raw.strip() and related_info_raw.lower() not in ['null', 'none', 'n/a', '']:
                        related_title_str = related_info_raw.strip()
                        # print(f"      Processing related movie for '{rel_key}': '{related_title_str}'...")
                        # For related movies, we don't have a TMDB ID from LLM, so year hint is None or from LLM if it provided one.
                        related_imdb_id = fetch_master_imdb_id(related_title_str, year_hint=None, is_tmdb_id=False, segment_num_str=f"{main_seg_id}-{rel_key}IMDb")
                        clean_entry[rel_key] = {"title": related_title_str, "imdb_id": related_imdb_id}
                        f_raw_log.write(f"Related '{rel_key}': {{'title': '{related_title_str}', 'imdb_id': {related_imdb_id}}}\n")
                    else:
                        clean_entry[rel_key] = None

                recommendations_from_llm = clean_entry.get("recommendations")
                if isinstance(recommendations_from_llm, list):
                    processed_recommendations = []
                    for rec_item_raw in recommendations_from_llm:
                        if isinstance(rec_item_raw, list) and len(rec_item_raw) == 3:
                            rec_title, rec_year_raw, rec_explainer = str(rec_item_raw or "").strip(), rec_item_raw, str(rec_item_raw or "").strip()
                            rec_year_validated, rec_year_for_display = None, "N/A"
                            if rec_year_raw is not None:
                                year_str_temp = str(rec_year_raw).strip()
                                if year_str_temp.isdigit() and len(year_str_temp) == 4:
                                    rec_year_validated = year_str_temp
                                    rec_year_for_display = year_str_temp
                                elif year_str_temp: rec_year_for_display = year_str_temp
                            if rec_title:
                                # print(f"        Processing recommendation: '{rec_title}' (Year hint: {rec_year_validated or 'None'})...")
                                rec_imdb_id = fetch_master_imdb_id(rec_title, rec_year_validated, is_tmdb_id=False, segment_num_str=f"{main_seg_id}-rec-{rec_title[:10].replace(' ','')}IMDb")
                                processed_recommendations.append({
                                    "title": rec_title, "year": rec_year_for_display,
                                    "explanation": rec_explainer, "imdb_id": rec_imdb_id })
                                f_raw_log.write(f"Recommendation: {{'title': '{rec_title}', 'imdb_id': {rec_imdb_id}}}\n")
                    clean_entry["recommendations"] = processed_recommendations if processed_recommendations else None
                else: clean_entry["recommendations"] = None

                for final_key in EXPECTED_YAML_KEYS_FINAL:
                    if final_key not in clean_entry: clean_entry[final_key] = None

                all_movie_data_to_persist.append(clean_entry)
                processed_movie_titles_lower_set.add(normalized_tmdb_title_lower)
                processed_movie_titles_original_case_list.append(tmdb_provided_title) # Add original case for final log
                new_movies_added_this_session += 1

                print(f"      Updating '{CLEAN_YAML_OUTPUT_FILENAME}' with '{tmdb_provided_title}'...")
                save_movie_data_to_yaml(all_movie_data_to_persist, CLEAN_YAML_OUTPUT_FILENAME)
                f_raw_log.write(f"      '{CLEAN_YAML_OUTPUT_FILENAME}' updated with '{tmdb_provided_title}'.\n\n")

                time.sleep(3) # Pause between successful movie additions
                f_raw_log.flush()

                if new_movies_added_this_session >= NUM_NEW_MOVIES_TO_FETCH_THIS_SESSION:
                    break # Exit the loop over movies_on_this_page
            # End of loop for movies on current TMDB page

            if not found_new_movie_on_this_page and not exhausted_tmdb_pages:
                print(f"No new movies found on TMDB page {current_tmdb_page} to process. Advancing to next page.")
                f_raw_log.write(f"No new movies found on TMDB page {current_tmdb_page}.\n")

            if new_movies_added_this_session >= NUM_NEW_MOVIES_TO_FETCH_THIS_SESSION:
                print(f"Target of {NUM_NEW_MOVIES_TO_FETCH_THIS_SESSION} new movies reached.")
                f_raw_log.write(f"Target of {NUM_NEW_MOVIES_TO_FETCH_THIS_SESSION} new movies reached.\n")
                break # Exit the main while loop

            current_tmdb_page += 1 # Go to next TMDB page
            if current_tmdb_page > tmdb_page_data.get("total_pages", current_tmdb_page -1 ): # total_pages might be 0 if error
                 print(f"Reached reported end of TMDB pages ({tmdb_page_data.get('total_pages', 'Unknown')}).")
                 exhausted_tmdb_pages = True

            time.sleep(1) # Brief pause before fetching next TMDB page
        # End of while loop for session movies

        if new_movies_added_this_session < NUM_NEW_MOVIES_TO_FETCH_THIS_SESSION:
            warning_msg = f"\nWARNING: Session ended. Added {new_movies_added_this_session}/{NUM_NEW_MOVIES_TO_FETCH_THIS_SESSION} new movies. "
            if exhausted_tmdb_pages: warning_msg += "Exhausted TMDB pages or page limit reached. "
            warning_msg += f"Total attempts/segments: {session_attempt_count}."
            print(warning_msg)
            f_raw_log.write(warning_msg + "\n")

        f_raw_log.write("="*40 + "\n")
        f_raw_log.write(f"--- All Titles Processed (from '{CLEAN_YAML_OUTPUT_FILENAME}' at end of session) ---\n")
        # Log the titles that were in the clean YAML (original case, sorted)
        final_titles_for_log = sorted(list(set(processed_movie_titles_original_case_list))) # Use set to ensure uniqueness before sort
        if final_titles_for_log:
            for title_val in final_titles_for_log: f_raw_log.write(f"- {title_val}\n")
        else: f_raw_log.write("No titles recorded in clean YAML during this session's processing.\n")
        f_raw_log.write("\n--- SESSION END ---\n")

    print(f"\n--- Movie Data Generation Complete. ---")
    print(f"   Added {new_movies_added_this_session} new movie(s) this session.")
    print(f"   Raw log: '{RAW_LOG_FILENAME}'")
    print(f"   Clean YAML data: '{CLEAN_YAML_OUTPUT_FILENAME}'")
    if TMDB_API_KEY:
        print(f"   Character images saved to: '{CHARACTER_IMAGE_SAVE_PATH}/'")

if __name__ == "__main__":
    run_movie_agent()