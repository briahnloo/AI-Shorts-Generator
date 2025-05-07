import time
import os
from lumaai import LumaAI
import gspread
import sys
from openai import OpenAI
import logging
from oauth2client.service_account import ServiceAccountCredentials

### Generate shorts with Lumalabs AI and automatic Sheets prompt retrieval and URL upload

# --- Configuration ---
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
LUMAAI_API_KEY = os.getenv("LUMAAI_API_KEY", "")
SHEET_ID = os.getenv("SHEET_ID", "")
SCOPE = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]
NUM_SCENES = 2

# Initialize clients
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["LUMAAI_API_KEY"] = LUMA_KEY
openai_client = OpenAI()
luma_client = LumaAI()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Google Sheets Setup ---
try:
    creds = ServiceAccountCredentials.from_json_keyfile_name("service-account.json", SCOPE)
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(SHEET_ID).sheet1
    logger.info("Sheet headers: %s", ws.row_values(1))
    rows = ws.get_all_records()
    logger.info("Found %d data rows.", len(rows))
except Exception as e:
    logger.error("Google Sheets setup failed: %s", e)
    sys.exit(1)

# --- OpenAI Story Generation ---
def generate_story_and_prompts(prompt_list):
    try:
        if len(prompt_list) == 1:
            openai_prompt = (
                "You are a creative writer tasked with generating a short 'day in the life' story based on the following topic: \n"
                f"- {prompt_list[0]}\n\n"
                f"Create a story (100-150 words) about a character whose day revolves around this topic, incorporating {NUM_SCENES} distinct scenes. "
                f"Then, provide exactly {NUM_SCENES} detailed, realistic prompts (30-50 words each) for generating videos that depict each scene in the story. "
                "Each prompt must be vivid, specific, tasteful, and suitable for a text-to-video API, ensuring visual realism, narrative coherence, and compliance with content moderation policies (no explicit or inappropriate content). "
                "Format the prompts as '1. [prompt]', '2. [prompt]', etc."
            )
            expected_prompts = NUM_SCENES
        else:
            prompt_bullets = "\n".join(f"- {p}" for p in prompt_list)
            openai_prompt = (
                "You are a creative writer tasked with generating a short 'day in the life' story based on the following prompts:\n"
                f"{prompt_bullets}\n\n"
                "Create a 100–150-word narrative that connects these prompts through a single character’s experience. "
                "The character might experience these scenes via travel, dreams, visions, or time shifts.\n\n"
                f"Then, generate exactly {len(prompt_list)} video-generation prompts (30–50 words each), "
                "each tied to one of the listed prompts. These video prompts should be vivid, realistic, and suitable for use with a text-to-video API. "
                "Format them as '1. [prompt]', '2. [prompt]', etc."
            )
            expected_prompts = len(prompt_list)

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a skilled storyteller and prompt engineer."},
                {"role": "user", "content": openai_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        logger.debug("OpenAI raw content: %s", content)

        # Parse story and prompts
        lines = content.split("\n")
        prompt_prefixes = tuple(f"{i}." for i in range(1, expected_prompts + 1))
        prompt_indices = [i for i, line in enumerate(lines) if line.strip().startswith(prompt_prefixes)]
        if len(prompt_indices) < expected_prompts:
            logger.error("Expected %d prompts, found %d", expected_prompts, len(prompt_indices))
            return None, None
        story_lines = lines[:prompt_indices[0]]
        story = "\n".join(story_lines).strip()
        detailed_prompts = []
        for idx in prompt_indices:
            prompt_line = lines[idx].strip()
            prompt_text = prompt_line.split(". ", 1)[1] if ". " in prompt_line else prompt_line
            detailed_prompts.append(prompt_text)
        detailed_prompts = detailed_prompts[:expected_prompts]
        return story, detailed_prompts
    except Exception as e:
        logger.error("Failed to generate story with OpenAI: %s", e)
        return None, None

# --- Luma Video Generation ---
def generate_luma_video(prompt):
    try:
        # Clean prompt
        clean_prompt = prompt.replace("**Prompt for Scene 1:** ", "").replace("**Prompt for Scene 2:** ", "").strip()[:250]
        
        # Create generation
        generation = luma_client.generations.create(
            prompt=clean_prompt,
            model="ray-2",
            resolution="720p",
            duration="5s",
            aspect_ratio="9:16"
        )
        logger.info(f"Started Luma job: {generation.id}")
        
        # Poll for completion
        while True:
            status = luma_client.generations.get(id=generation.id)
            if status.state == "completed":
                logger.info(f"Luma job {generation.id} completed")
                return status.assets.video
            elif status.state == "failed":
                logger.error(f"Luma generation failed: {status.failure_reason}")
                return None
            logger.debug(f"Polling Luma job {generation.id}: state={status.state}")
            time.sleep(5)
            
    except Exception as e:
        logger.error(f"Luma video generation failed: {str(e)}")
        return None

# --- Main Processing ---
try:
    prompt_list = [row.get("Topic") for row in rows if row.get("Topic")]
    if not prompt_list:
        logger.error("No prompts found")
        sys.exit(1)
    logger.info("Prompts from sheet: %s", prompt_list)

    story, detailed_prompts = generate_story_and_prompts(prompt_list)
    if not story or not detailed_prompts:
        logger.error("Failed to generate valid story or prompts")
        sys.exit(1)
    logger.info("Generated story: %s", story)
    logger.info("Detailed prompts: %s", detailed_prompts)

    # Generate videos with Luma
    video_urls = []
    for idx, prompt in enumerate(detailed_prompts, 1):
        video_url = generate_luma_video(prompt)
        if video_url:
            video_urls.append(video_url)
        else:
            logger.error("Failed to generate video for scene %d", idx)
            sys.exit(1)

    if len(video_urls) != len(detailed_prompts):
        logger.error("Not all videos were generated successfully")
        sys.exit(1)

    # Update Google Sheet
    video_result = ", ".join(video_urls)
    try:
        col = ws.find("Video URL").col
        ws.update_cell(2, col, video_result)
        logger.info("Updated row 2 with video URLs: %s", video_result)
    except gspread.exceptions.CellNotFound:
        logger.error("Missing Video URL column")

except Exception as e:
    logger.error(f"Processing failed: {str(e)}")
    sys.exit(1)

logger.info("Process completed successfully")