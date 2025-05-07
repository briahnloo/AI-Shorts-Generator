import os
import logging
import time
import sys
import tempfile
import subprocess
import requests
from typing import List
from dotenv import load_dotenv
from lumaai import LumaAI
from pydub import AudioSegment

### Manually generate with input prompts
# --- Load Environment Variables ---
load_dotenv()
LUMAAI_API_KEY = os.getenv("LUMAAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# --- Configuration ---
NUM_SCENES = 3
MANUAL_PROMPTS = [
    """ORBITAL ARRIVAL: Tracking shot of asteroid transport docking with ring-shaped foundry. Burning entry sparks reflect off metallic surfaces. EXTREME WIDE of industrial complex against Earth backdrop. Style: Realistic space industry. Cinematic elements: Zero-G debris field, reactor glow pulses, docking clamps engaging with bass vibration.""",

    """MOLTEN CORE: First-person POV through safety visor watching golden metal stream. 8K macro of zero-gravity droplet formation. Robotic arms perform precision welding ballet. Style: Industrial sci-fi. Cinematic elements: Molten metal surface tension ASMR, hydraulic arm whooshes, radiation warning HUD overlay.""",

    """FORGED IN VOID: Final product reveal - glowing space elevator cable strand. Camera follows nanobot swarm applying quantum coating. Earthrise illuminates finished product. Style: Futuristic manufacturing. Cinematic elements: Cosmic radiation shimmer, magnetic field distortion effects, deep orbital rumble."""
]

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Client Initialization ---
if not LUMAAI_API_KEY:
    logger.error("LUMAAI_API_KEY is not set. Please set it in environment variables.")
    sys.exit(1)
if not ELEVENLABS_API_KEY:
    logger.error("ELEVENLABS_API_KEY is not set. Please set it in environment variables.")
    sys.exit(1)

luma_client = LumaAI()
os.environ["LUMAAI_API_KEY"] = LUMAAI_API_KEY  # Add this line

# --- Audio Generation Functions ---
def create_audio_prompt(video_prompt: str) -> str:
    """Generate industrial sci-fi audio prompts"""
    scene_keywords = {
        0: {
            'sounds': "Docking clangs (low-frequency impacts), decompression hisses, reactor hum buildup",
            'music': "Pulsating industrial drones (55Hz sub-bass)"
        },
        1: {
            'sounds': "Molten metal bubble pops (high-frequency sizzle), robotic servo whines, radiation Geiger counter",
            'music': "Mechanical rhythm syncopated to arm movements"
        },
        2: {
            'sounds': "Nanobot swarm buzz (phase modulation), cable tension creaks, orbital vacuum rumble",
            'music': "Triumphant cosmic choir with metallic percussion"
        }
    }
    
    # Extract scene index from prompt
    scene_idx = None
    if "ORBITAL ARRIVAL" in video_prompt: scene_idx = 0
    elif "MOLTEN CORE" in video_prompt: scene_idx = 1
    elif "FORGED IN VOID" in video_prompt: scene_idx = 2
    
    if scene_idx is not None:
        return (
            f"Create industrial space ASMR audio:\n"
            f"- Primary sounds: {scene_keywords[scene_idx]['sounds']}\n"
            f"- Background score: {scene_keywords[scene_idx]['music']}\n"
            f"- Retention hooks: Deep metallic impacts every 2.5 seconds\n"
            f"- Platform optimization: Bass drops at 0:04, 0:11, 0:17"
        )
    
    return f"Create orbital factory soundscape for: {video_prompt[:200]}"


def generate_scene_audio(scene_prompt: str, duration_s: float) -> str:
    """Generate audio using ElevenLabs Sound Effects API"""
    url = "https://api.elevenlabs.io/v1/sound-generation"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    params = {
        "text": create_audio_prompt(scene_prompt),  # Docs say "text" not "prompt"
        "duration_seconds": duration_s,  # Required format
        "prompt_influence": 0.7,  # Default is 0.3 (0-1 scale)
        "output_format": "mp3_44100_128"  # Free tier compatible format
    }

    try:
        response = requests.post(url, headers=headers, json=params, timeout=30)
        response.raise_for_status()
        
        # Save response content directly (API returns audio bytes)
        fd, path = tempfile.mkstemp(suffix=".mp3")
        with os.fdopen(fd, 'wb') as f:
            f.write(response.content)
        return path

    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"API Response: {e.response.text}")
        return None


def combine_all_audio(audio_files: List[str]) -> str:
    """Combine audio files into a single track"""
    try:
        combined_audio = AudioSegment.empty()
        for audio_file in audio_files:
            audio = AudioSegment.from_file(audio_file)
            # Correct normalization syntax:
            audio = audio.normalize()  # Remove target_dbfs parameter
            combined_audio += audio
        
        output_file = "combined_audio.mp3"
        combined_audio.export(output_file, format="mp3")
        return output_file
    except Exception as e:
        logger.error(f"Audio combination failed: {str(e)}")
        return None

# --- Video Generation Functions ---
def generate_scene(prompt: str, scene_num: int) -> str:
    """Generate individual 5s scene"""
    try:
        logger.info(f"Generating scene {scene_num}...")
        generation = luma_client.generations.create(
            prompt=prompt[:350],  # Truncate to API limits
            model="ray-2",
            resolution="1080p",
            duration="5s",  # Must be 5s or 9s
            aspect_ratio="16:9"
        )
        
        while True:
            status = luma_client.generations.get(id=generation.id)
            if status.state == "completed":
                return status.assets.video
            elif status.state == "failed":
                logger.error(f"Scene {scene_num} failed: {status.failure_reason}")
                return None
            time.sleep(10)
    
    except Exception as e:
        logger.error(f"Scene {scene_num} error: {str(e)}")
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
        # Create FFmpeg input list
        with open("concat_list.txt", "w") as f:
            for file in scene_files:
                f.write(f"file '{os.path.abspath(file)}'\n")
        
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', 'concat_list.txt',
            '-c', 'copy',  # Stream copy (no re-encoding)
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

# --- Updated Main Workflow ---
def generate_story_video(prompts: List[str]) -> str:
    """Generate and combine multiple 5s scenes"""
    scene_urls = []
    scene_files = []
    audio_files = []
    final_output = "industrial_sci-fi.mp4"

    try:
        # Validate number of prompts
        if len(prompts) > 5:
            logger.error("Maximum 5 scenes supported")
            return None

        # Generate each scene individually
        for i, prompt in enumerate(prompts, 1):
            # Scene generation
            scene_url = generate_scene(prompt, i)
            if not scene_url:
                logger.error(f"Aborting due to scene {i} failure")
                return None
                
            # Video download
            scene_file = f"scene_{i}.mp4"
            if not download_video(scene_url, scene_file):
                logger.error(f"Failed to download scene {i}")
                return None
            scene_files.append(scene_file)
            
            # Audio generation
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

        # Merge audio/video
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
        # Enhanced cleanup
        temp_files = [
            *scene_files, 
            *audio_files,
            'combined_video.mp4',
            'combined_audio.mp3'
        ]
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                logger.warning(f"Cleanup failed for {f}: {str(e)}")

# --- Run the Workflow ---
if __name__ == "__main__":
    try:
        final_video = generate_story_video(MANUAL_PROMPTS)
        if final_video:
            logger.info(f"Success! Final video with audio: {final_video}")
        else:
            logger.error("Video generation failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)

    logger.info("Process completed successfully")
# def test_audio():
#     test_prompt = "Ancient Egyptian market sounds"
#     audio_file = generate_scene_audio(test_prompt, 5)
#     if audio_file:
#         os.system(f"open {audio_file}")

# if __name__ == "__main__":
#     test_audio()
