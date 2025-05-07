import time
import requests
import gspread
import sys
import openai
from oauth2client.service_account import ServiceAccountCredentials
import logging
import json
import urllib.request
import tempfile
import os
from dotenv import load_dotenv

### Generate shorts with Creatomate and Picsart

# --- Load Environment Variables ---
load_dotenv()  # Load variables from .env file

# --- Configuration ---
PICSART_KEY = os.getenv("PICSART_KEY", "")
CREATO_KEY = os.getenv("CREATO_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
SHEET_ID = os.getenv("SHEET_ID", "")
TEMPLATE_ID = os.getenv("TEMPLATE_ID", "")
IMGBB_KEY = os.getenv("IMGBB_KEY", "")

SCOPE       = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]
NUM_SCENES  = 2  # Using 2 scenes

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Set Up OpenAI ---
openai.api_key = OPENAI_KEY

# --- Set Up Google Sheets ---
try:
    creds = ServiceAccountCredentials.from_json_keyfile_name("service-account.json", SCOPE)
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(SHEET_ID).sheet1
    logger.info("Sheet headers: %s", ws.row_values(1))
    rows = ws.get_all_records()
    logger.info("Found %d data rows.", len(rows))
except Exception as e:
    logger.error("Failed to set up Google Sheets: %s", e)
    sys.exit(1)

# --- Generate Story and Prompts with OpenAI ---
def generate_story_and_prompts(prompt_list):
    try:
        if len(prompt_list) == 1:
            openai_prompt = (
                "You are a creative writer tasked with generating a short 'day in the life' story based on the following topic: \n"
                f"- {prompt_list[0]}\n\n"
                f"Create a story (100-150 words) about a character whose day revolves around this topic, incorporating {NUM_SCENES} distinct scenes. "
                f"Then, provide exactly {NUM_SCENES} detailed, realistic prompts (30-50 words each) for generating images that depict each scene in the story. "
                "Each prompt must be vivid, specific, tasteful, and suitable for a text-to-image API, ensuring visual realism, narrative coherence, and compliance with content moderation policies (no explicit or inappropriate content). "
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
                f"Then, generate exactly {len(prompt_list)} image-generation prompts (30–50 words each), "
                "each tied to one of the listed prompts. These image prompts should be vivid, realistic, and suitable for use with a text-to-image API. "
                "Format them as '1. [prompt]', '2. [prompt]', etc."
            )
            expected_prompts = len(prompt_list)

        response = openai.chat.completions.create(
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

# --- Generate Picsart Image ---
def generate_picsart_image(prompt):
    try:
        logger.info("Picsart prompt: %s", prompt)
        pic_post = requests.post(
            "https://genai-api.picsart.io/v1/text2image",
            headers={"X-Picsart-Api-Key": PICSART_KEY, "Accept": "application/json"},
            json={"prompt": prompt, "count": 1}
        )
        if pic_post.status_code not in (200, 202):
            logger.error("Picsart API error: %s", pic_post.text)
            return None
        pic_json = pic_post.json()
        job_id = pic_json.get("inference_id") or pic_json.get("id")
        if not job_id:
            logger.error("Picsart failed to return job ID: %s", pic_json)
            return None
        logger.info("Picsart job_id: %s", job_id)
        start = time.time()
        timeout = 300
        while True:
            poll = requests.get(
                f"https://genai-api.picsart.io/v1/text2image/inferences/{job_id}",
                headers={"X-Picsart-Api-Key": PICSART_KEY}
            )
            if poll.status_code >= 400:
                logger.error("Picsart polling HTTP error: %s", poll.text)
                return None
            p = poll.json()
            status = p.get("status", "").lower()
            logger.info("Polling Picsart status: %s", status)
            if status in ("finished", "failed"):
                break
            if time.time() - start > timeout:
                logger.warning("Picsart polling timed out after %ds", timeout)
                return None
            time.sleep(2)
        if status != "finished":
            logger.warning("Picsart job did not finish (status=%s)", status)
            return None
        image_url = (p.get("images") or p.get("data", []))[0].get("url")
        if not image_url:
            logger.error("No image URL found: %s", p)
            return None
        logger.info("Picsart image_url = %s", image_url)
        return image_url
    except Exception as e:
        logger.error("Error generating Picsart image: %s", e)
        return None

# --- Upload to ImgBB CDN ---
def upload_to_cdn(image_url):
    try:
        # Download image
        img_data = urllib.request.urlopen(image_url).read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(img_data)
            tmp_path = tmp.name
        
        # Upload to ImgBB
        with open(tmp_path, "rb") as file:
            imgbb_response = requests.post(
                "https://api.imgbb.com/1/upload",
                data={"key": IMGBB_KEY},
                files={"image": file}
            )
        if imgbb_response.status_code != 200:
            logger.error("ImgBB API error: %s", imgbb_response.text)
            return None
        public_url = imgbb_response.json()["data"]["url"]
        logger.info("Uploaded to ImgBB: %s", public_url)
        return public_url
    except Exception as e:
        logger.error("CDN upload failed: %s", e)
        return None

# --- Main Processing Logic ---
try:
    prompt_list = [row.get("Topic") for row in rows if row.get("Topic")]
    if not prompt_list:
        logger.error("No prompts found in Google Sheet")
        sys.exit(1)
    logger.info("Prompts from sheet: %s", prompt_list)

    story, detailed_prompts = generate_story_and_prompts(prompt_list)
    if not story or not detailed_prompts:
        logger.error("Failed to generate valid story or prompts")
        sys.exit(1)
    logger.info("Generated story: %s", story)
    logger.info("Detailed prompts: %s", detailed_prompts)

    # Test with Unsplash URLs for verification (uncomment to test)
    # image_urls = [
    #     "https://images.unsplash.com/photo-1600585154340-be6161a56a0c",
    #     "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
    # ]

    # Generate and upload Picsart images to ImgBB
    image_urls = []
    for idx, prompt in enumerate(detailed_prompts, 1):
        picsart_url = generate_picsart_image(prompt)
        if picsart_url:
            public_url = upload_to_cdn(picsart_url)
            if public_url:
                image_urls.append(public_url)
            else:
                logger.error("Failed to upload image %d to CDN", idx)
                sys.exit(1)
        else:
            logger.error("Failed to generate image for scene %d", idx)
            sys.exit(1)

    if len(image_urls) != len(detailed_prompts):
        logger.error("Not all images were generated or uploaded successfully")
        sys.exit(1)

    # --- Creatomate Render ---
    logger.info("Triggering Creatomate render with %d images", len(image_urls))

    elements = []
    for idx, (url, prompt) in enumerate(zip(image_urls, detailed_prompts)):
        # Image with Ken Burns effect
        elements.append({
            "type": "image",
            "time": idx * 3,
            "duration": 3,
            "source": url,
            "width": "110%",  # Allow room for movement
            "height": "110%",
            "x": "0%",
            "y": "0%",
            "animations": [
                {
                    "type": "scale",
                    "start_scale": "100%",
                    "end_scale": "110%",
                    "easing": "linear",
                    "duration": 3
                },
                {
                    "type": "move",
                    "x": ["0%", "-5%"],
                    "y": ["0%", "-5%"],
                    "easing": "linear",
                    "duration": 3
                }
            ]
        })
        
        # Text overlay
        elements.append({
            "type": "text",
            "time": idx * 3 + 0.5,
            "duration": 2.5,
            "text": prompt[:100],
            "y": "80%",
            "width": "90%",
            "font_family": "Roboto",
            "font_weight": "bold",
            "font_size": 30,
            "background_color": "rgba(0,0,0,0.7)",
            "animations": [
                {
                    "type": "fade",
                    "start": 0,
                    "end": 1,
                    "duration": 0.5
                }
            ]
        })

    modifications = {
        "main.elements": elements
    }

    # Log the final API payload
    logger.info("Final API Payload: %s", json.dumps(modifications, indent=2))

    cm_post = requests.post(
        "https://api.creatomate.com/v1/renders",
        headers={
            "Authorization": f"Bearer {CREATO_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "template_id": TEMPLATE_ID,
            "modifications": modifications
        }
    )
    if cm_post.status_code not in (200, 202):
        logger.error("Creatomate API error: %s", cm_post.text)
        sys.exit(1)
    cm_json = cm_post.json()
    render_id = (cm_json[0] if isinstance(cm_json, list) else cm_json.get("data", {})).get("id")
    if not render_id:
        logger.error("Failed to get render ID: %s", cm_json)
        sys.exit(1)
    logger.info("Creatomate render_id = %s", render_id)

    # Enhanced polling with detailed logging
    start = time.time()
    timeout = 300
    while True:
        cr = requests.get(
            f"https://api.creatomate.com/v1/renders/{render_id}",
            headers={"Authorization": f"Bearer {CREATO_KEY}"}
        )
        if cr.status_code >= 400:
            logger.error("Creatomate polling HTTP error: %s", cr.text)
            sys.exit(1)
        cj = cr.json()
        status = cj.get("status", "").lower()
        logger.info("Polling Creatomate status: %s, response: %s", status, cj)
        if status in ("succeeded", "failed"):
            break
        if time.time() - start > timeout:
            logger.warning("Creatomate polling timed out after %ds", timeout)
            sys.exit(1)
        time.sleep(3)

    if status != "succeeded":
        logger.error("Render failed: %s", cj.get("error_message", "No error message provided"))
        sys.exit(1)
    video_result = cj.get("url")
    if not video_result:
        logger.error("No video URL: %s", cj)
        sys.exit(1)
    logger.info("Video URL = %s", video_result)

    # --- Update Google Sheet ---
    try:
        col = ws.find("Video URL").col
        ws.update_cell(2, col, video_result)
        logger.info("Updated row 2 with video URL.")
    except gspread.exceptions.CellNotFound:
        logger.error("Column 'Video URL' not found; cannot write back")
except Exception as e:
    logger.error("Error processing story: %s", str(e))
    sys.exit(1)

logger.info("Story processed; exiting.")