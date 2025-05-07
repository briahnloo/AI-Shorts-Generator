import os
import logging
import time
import sys
import tempfile
import subprocess
import requests
import base64
from typing import List
from dotenv import load_dotenv
from lumaai import LumaAI
from pydub import AudioSegment
from pathlib import Path

### Manually generate with input images and prompts

# --- Load Environment Variables ---
load_dotenv()
LUMAAI_API_KEY = os.getenv("LUMAAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Client Initialization ---
if not LUMAAI_API_KEY:
    logger.error("LUMAAI_API_KEY is not set.")
    sys.exit(1)
if not ELEVENLABS_API_KEY:
    logger.error("ELEVENLABS_API_KEY is not set.")
    sys.exit(1)
if not IMGBB_API_KEY:
    logger.error("IMGBB_API_KEY is not set.")
    sys.exit(1)

luma_client = LumaAI()
os.environ["LUMAAI_API_KEY"] = LUMAAI_API_KEY

# --- Define Image Paths and Prompts ---
IMAGE_PATHS = [
    "/Users/bzliu/Desktop/EXTRANEOUS_CODE/AI Shorts Generator/reference_data/scene1_wakeup.jpg",
    "/Users/bzliu/Desktop/EXTRANEOUS_CODE/AI Shorts Generator/reference_data/scene2_training.jpeg",
    "/Users/bzliu/Desktop/EXTRANEOUS_CODE/AI Shorts Generator/reference_data/scene3_ramen.jpg"
]

PROMPTS = [
    # Scene 1: Waking up and stretching
    "The camera represents my eyes as a shinobi waking up in my room in the Hidden Leaf Village. My hands rise into view, rubbing my face briefly before stretching upward and outward in front of me, fingers flexing. Through my eyes, the wooden ceiling of my modest room comes into focus, then shifts as I turn my head toward the open window. My outstretched hands frame the view of the village below—rooftops bathed in soft morning light, the Hokage mountain in the distance, and a gentle breeze rustling the curtains. The perspective remains strictly first-person, showing only what I see through my own eyes as I stretch and take in the serene morning scene.",

    # Scene 2: Training with a weapon
    "The camera is my own viewpoint as a shinobi training in a forest clearing. My hands grip a kunai tightly in front of me, the metal glinting as I twirl it with precision between my fingers. The perspective shows my arms moving fluidly, slashing and thrusting the weapon as if I’m practicing a kata. Through my eyes, the forest unfolds—tall trees sway slightly, wooden training dummies stand in the background with kunai marks, and leaves rustle underfoot. The focus remains on my hands and the kunai, with the training ground visible only as I see it, maintaining a strict first-person view of my own actions and surroundings.",

    # Scene 3: Eating ramen with friends
    "The camera embodies my perspective as a shinobi sitting at a ramen shop table. My hands hold chopsticks in front of me, deftly picking up steaming ramen from a bowl, noodles glistening with broth. The viewpoint shows my hands bringing the food toward my mouth, then shifts slightly as I look up to see fellow shinobi across the table—some slurping loudly, others chatting animatedly. Through my eyes, the cozy shop interior is visible: wooden beams, lanterns casting a warm glow, and a window revealing a carved mountain in the distance. The bustling sounds of the shop fill the air, but the view stays locked in first-person, capturing only what I see as I eat and interact."]

# --- Audio Generation Functions ---
def create_audio_prompt(video_prompt: str) -> str:
    """Generate audio prompt based on video prompt"""
    return f"Create ambient sounds matching this first-person scene: {video_prompt[:200]}"

def generate_scene_audio(scene_prompt: str, duration_s: float) -> str:
    """Generate audio using ElevenLabs Sound Effects API"""
    url = "https://api.elevenlabs.io/v1/sound-generation"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    params = {
        "text": create_audio_prompt(scene_prompt),
        "duration_seconds": duration_s,
        "prompt_influence": 0.7,
        "output_format": "mp3_44100_128"
    }

    try:
        response = requests.post(url, headers=headers, json=params, timeout=30)
        response.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=".mp3")
        with os.fdopen(fd, 'wb') as f:
            f.write(response.content)
        return path
    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        return None

def combine_all_audio(audio_files: List[str]) -> str:
    """Combine audio files into a single track"""
    try:
        combined_audio = AudioSegment.empty()
        for audio_file in audio_files:
            audio = AudioSegment.from_file(audio_file)
            audio = audio.normalize()
            combined_audio += audio
        output_file = "combined_audio.mp3"
        combined_audio.export(output_file, format="mp3")
        return output_file
    except Exception as e:
        logger.error(f"Audio combination failed: {str(e)}")
        return None

# --- Video Generation Functions ---
def upload_image_to_cdn(image_path: str) -> str:
    """Upload image to ImgBB and return the URL"""
    url = "https://api.imgbb.com/1/upload"
    with open(image_path, "rb") as file:
        image_data = base64.b64encode(file.read()).decode('utf-8')
    payload = {
        "key": IMGBB_API_KEY,
        "image": image_data
    }
    try:
        response = requests.post(url, data=payload, timeout=30)
        response.raise_for_status()
        return response.json()["data"]["url"]
    except Exception as e:
        logger.error(f"Image upload failed: {str(e)}")
        raise

def generate_scene_from_image(image_path: str, prompt: str, scene_num: int) -> str:
    """Generate video from image using LumaAI's image-to-video API"""
    try:
        logger.info(f"Generating scene {scene_num} from image...")
        image_url = upload_image_to_cdn(image_path)
        generation = luma_client.generations.create(
            prompt=prompt[:350],
            model="ray-2",
            keyframes={
                "frame0": {
                    "type": "image",
                    "url": image_url
                }
            },
            resolution="1080p",
            duration="5s",
            aspect_ratio="16:9"
        )
        start_time = time.time()
        while time.time() - start_time < 600:
            status = luma_client.generations.get(id=generation.id)
            if status.state == "completed":
                return status.assets.video
            elif status.state == "failed":
                logger.error(f"Scene {scene_num} failed: {status.failure_reason}")
                return None
            time.sleep(max(5, min(30, (time.time() - start_time)/10)))
        logger.error(f"Scene {scene_num} timed out")
        return None
    except Exception as e:
        logger.error(f"Image-based generation error: {str(e)}")
        return None

def download_video(url: str, filename: str) -> bool:
    """Download video from URL"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Video download failed: {str(e)}")
        return False

def stitch_scenes(scene_files: List[str], output_file: str) -> bool:
    """Combine multiple 5s scenes into one video"""
    try:
        with open("concat_list.txt", "w") as f:
            for file in scene_files:
                f.write(f"file '{os.path.abspath(file)}'\n")
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', 'concat_list.txt',
            '-c', 'copy',
            output_file
        ]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        logger.error(f"Stitching failed: {str(e)}")
        return False
    finally:
        if os.path.exists("concat_list.txt"):
            os.remove("concat_list.txt")

# --- Main Workflow ---
def generate_story_video(prompts: List[str], image_paths: List[str]) -> str:
    """Generate and combine multiple 5s scenes from images and prompts"""
    if len(prompts) != len(image_paths):
        logger.error("Number of prompts must match number of image paths")
        return None
    scene_urls = []
    scene_files = []
    audio_files = []
    final_output = "shinobi_day_fpv_final.mp4"

    try:
        for i, (prompt, image_path) in enumerate(zip(prompts, image_paths), 1):
            # Generate video from image and prompt
            scene_url = generate_scene_from_image(image_path, prompt, i)
            if not scene_url:
                logger.error(f"Aborting due to scene {i} failure")
                return None
            # Download video
            scene_file = f"scene_{i}.mp4"
            if not download_video(scene_url, scene_file):
                logger.error(f"Failed to download scene {i}")
                return None
            scene_files.append(scene_file)
            # Generate audio
            audio_file = generate_scene_audio(prompt, duration_s=5)
            if not audio_file:
                logger.error(f"Audio failed for scene {i}")
                return None
            audio_files.append(audio_file)
        # Stitch videos
        if not stitch_scenes(scene_files, "combined_video.mp4"):
            logger.error("Video stitching failed")
            return None
        # Combine audio
        combined_audio = combine_all_audio(audio_files)
        if not combined_audio:
            logger.error("Audio combination failed")
            return None
        # Merge audio and video
        cmd = [
            'ffmpeg', '-y',
            '-i', 'combined_video.mp4',
            '-i', combined_audio,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            final_output
        ]
        subprocess.run(cmd, check=True)
        return final_output
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {str(e)}")
        return None
    finally:
        temp_files = scene_files + audio_files + ['combined_video.mp4', 'combined_audio.mp3']
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                logger.warning(f"Cleanup failed for {f}: {str(e)}")

# --- Run the Workflow ---
if __name__ == "__main__":
    try:
        final_video = generate_story_video(PROMPTS, IMAGE_PATHS)
        if final_video:
            logger.info(f"Success! Final video with audio: {final_video}")
        else:
            logger.error("Video generation failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)
    logger.info("Process completed successfully")