from .uvr5 import uvr_prediction, uvr5_names
import os
import shutil

def separate_vocals(
    input_file: str,
    output_vocal_dir: str = "output/vocals",
    output_instrumental_dir: str = "output/instrumental",
    aggressiveness: int = 10,
    output_format: str = "wav"
) -> tuple[str, str]:
    """
    Separate vocals from background music in an audio file.
    
    Args:
        input_file: Path to the input audio file
        output_vocal_dir: Directory to save extracted vocals
        output_instrumental_dir: Directory to save instrumental track
        aggressiveness: Intensity of vocal separation (1-10)
        output_format: Output audio format (wav, flac, mp3, m4a)
    
    Returns:
        tuple: (path to vocal file, path to instrumental file)
    """
    # Input validation
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directories if they don't exist
    for directory in [output_vocal_dir, output_instrumental_dir]:
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

    # Validate format
    valid_formats = ["wav", "flac", "mp3", "m4a"]
    if output_format not in valid_formats:
        raise ValueError(f"Invalid format. Must be one of: {valid_formats}")

    # Run vocal separation
    vocal_path, instrumental_path = uvr_prediction(
        model_name="VR-DeEchoNormal",
        inp_path=input_file,
        save_root_vocal=output_vocal_dir,
        save_root_ins=output_instrumental_dir,
        agg=aggressiveness,
        format0=output_format
    )

    return vocal_path, instrumental_path
