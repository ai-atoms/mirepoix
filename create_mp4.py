import os
import re
import cv2
import numpy as np
from moviepy import ImageSequenceClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from PIL import Image, ImageDraw, ImageFont


def create_video(root_dir, output_path='output.mp4', fps=12, dpa_per_ns=0.16, fov=14):
    """
    Create an annotated video from frames in the root directory.
    
    Args:
        root_dir (str): Directory containing frames named NNNNNN_X.png
        output_path (str): Output video path (default: 'output.mp4')
        fps (int): Frames per second (default: 12)
        dpa_per_ns (float): DPA increase per nanosecond (default: 0.16)
    """
    # Get all frame files and sort them by time
    frame_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
    frame_files.sort(key=lambda x: int(re.search(r'^(\d+)', x).group(1)))
    
    # Create output writer
    writer = FFMPEG_VideoWriter(output_path, (1024, 1088), fps)
    
    # Font settings (you may need to adjust the font path for your system)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/DejaVuMathTeXGyre.ttf", 32, encoding="unic")
    except:
        font = ImageFont.load_default()
    
    for frame_file in frame_files:
        # Parse time from filename (in femtoseconds)
        time_fs = int(re.search(r'^(\d+)', frame_file).group(1))
        time_ns = time_fs / 1e6  # convert to nanoseconds
        
        # Load and resize the image
        img_path = os.path.join(root_dir, frame_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1024, 1024))
        
        # Create dark gray bottom panel (1024x176)
        bottom_panel = np.full((64, 1024, 3), (50, 50, 50), dtype=np.uint8)
        
        # Combine image and bottom panel
        combined = np.vstack((img, bottom_panel))
        
        # Convert to PIL Image for drawing text
        pil_img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Add time in nanoseconds (lower left)
        time_text = f"{time_ns:.1f} ns"
        draw.text((20, 1040), time_text, font=font, fill=(255, 255, 255))
        
        # Add FOV text (center)
        fov_text = f"FOV = {fov} nm"
        text_width = draw.textlength(fov_text, font=font)
        draw.text(((1024 - text_width) // 2, 1040), fov_text, font=font, fill=(255, 255, 255))
        
        # Add DPA counter (lower right)
        dpa = time_ns * dpa_per_ns
        dpa_text = f"{dpa:.2f} dpa"
        text_width = draw.textlength(dpa_text, font=font)
        draw.text((1024 - text_width - 20, 1040), dpa_text, font=font, fill=(255, 255, 255))
        
        # Convert back to OpenCV format and write frame
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        writer.write_frame(frame)
    
    writer.close()
    print(f"Video saved to {output_path}")


def create_video_with_bkg(root_dir, output_path='output.mp4', fps=12, dpa_per_ns=0.16, fov=14):
    """
    Create an annotated video from frames with transparent backgrounds,
    using a white background.
    
    Args:
        root_dir (str): Directory containing frames named NNNNNN_X.png
        output_path (str): Output video path (default: 'output.mp4')
        fps (int): Frames per second (default: 30)
        dpa_per_ns (float): DPA increase per nanosecond (default: 0.16)
    """
    # Get all frame files and sort them by time
    frame_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
    frame_files.sort(key=lambda x: int(re.search(r'^(\d+)', x).group(1)))
    
    # Create output writer
    writer = FFMPEG_VideoWriter(output_path, (1024, 1088), fps)
    
    # Font settings (you may need to adjust the font path for your system)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/DejaVuMathTeXGyre.ttf", 32, encoding="unic")
    except:
        font = ImageFont.load_default()
    
    for frame_file in frame_files:
        # Parse time from filename
        time_fs = int(re.search(r'^(\d+)', frame_file).group(1))
        time_ns = time_fs / 1e6
        
        # Load image with alpha channel
        img_path = os.path.join(root_dir, frame_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # Create white background and composite
        if img.shape[2] == 4:  # Has alpha channel
            white_bg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            alpha = img[:,:,3] / 255.0
            for c in range(3):
                white_bg[:,:,c] = white_bg[:,:,c] * (1 - alpha) + img[:,:,c] * alpha
            img = white_bg
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize after compositing
        img = cv2.resize(img, (1024, 1024))

        # --camilofs: ideally get these from GIMP
        crop_size = 914
        start_x = 58
        start_y = 52
        img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        # Then resize back to 1024x1024 if needed:
        img = cv2.resize(img, (1024, 1024))
        
        # Create bottom panel
        bottom_panel = np.full((64, 1024, 3), (255, 255, 255), dtype=np.uint8)
        combined = np.vstack((img, bottom_panel))
        
        # Convert to PIL Image for drawing
        pil_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(pil_img)
        
        # Add annotations
        time_text = f"{time_ns:.1f} ns"
        fov_text = f"FOV = {fov} nm"
        dpa = time_ns * dpa_per_ns
        dpa_text = f"{dpa:.2f} dpa"
        
        draw.text((20, 1040), time_text, font=font, fill=(50, 50, 50))
        text_width = draw.textlength(fov_text, font=font)
        draw.text(((1024 - text_width) // 2, 1040), fov_text, font=font, fill=(50, 50, 50))
        text_width = draw.textlength(dpa_text, font=font)
        draw.text((1024 - text_width - 20, 1040), dpa_text, font=font, fill=(50, 50, 50))
        
        # Write frame
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        writer.write_frame(frame)
    
    writer.close()
    print(f"Video with white background saved to {output_path}")


def create_video_combined(root_dir, overlay_dir, output_path='output_combined.mp4', fps=12, dpa_per_ns=0.16, fov=14, overlay_opacity=0.5):
    """
    Create an annotated video combining images from root_dir with overlay images from overlay_dir.
    
    Args:
        root_dir (str): Directory containing base frames named NNNNNN_X.png
        overlay_dir (str): Directory containing overlay frames (same names as root_dir)
        output_path (str): Output video path (default: 'output_combined.mp4')
        fps (int): Frames per second (default: 12)
        dpa_per_ns (float): DPA increase per nanosecond (default: 0.16)
        fov (float): Field of view in nm (default: 14)
        overlay_opacity (float): Opacity of overlay image (0-1, default: 0.5)
    """
    # Get all frame files and sort them by time
    frame_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
    frame_files.sort(key=lambda x: int(re.search(r'^(\d+)', x).group(1)))
    
    # Create output writer
    writer = FFMPEG_VideoWriter(output_path, (1024, 1088), fps)
    
    # Font settings
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/DejaVuMathTeXGyre.ttf", 32, encoding="unic")
    except:
        font = ImageFont.load_default()
    
    for frame_file in frame_files:
        # Parse time from filename
        time_fs = int(re.search(r'^(\d+)', frame_file).group(1))
        time_ns = time_fs / 1e6
        
        # Load and resize base image
        img_path = os.path.join(root_dir, frame_file)
        base_img = cv2.imread(img_path)
        base_img = cv2.resize(base_img, (1024, 1024))
        
        # Load and resize overlay image if exists
        overlay_path = os.path.join(overlay_dir, frame_file)
        if os.path.exists(overlay_path):
            overlay_img = cv2.imread(overlay_path)
            overlay_img = cv2.resize(overlay_img, (1024, 1024))
            # --camilofs: ideally get these from GIMP
            crop_size = 914
            start_x = 58
            start_y = 52
            overlay_img = overlay_img[start_y:start_y+crop_size, start_x:start_x+crop_size]
            # Then resize back to 1024x1024 if needed:
            overlay_img = cv2.resize(overlay_img, (1024, 1024))
            # Ensure proper color space before blending
            if overlay_img.shape[2] >= 3:  # Has color channels
                overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)  # or BGRA2RGBA if has alpha
            # Blend images
            combined_img = cv2.addWeighted(base_img, 1 - overlay_opacity, overlay_img, overlay_opacity, 0)
        else:
            combined_img = base_img
        
        # Create bottom panel
        bottom_panel = np.full((64, 1024, 3), (50, 50, 50), dtype=np.uint8)
        combined = np.vstack((combined_img, bottom_panel))
        
        # Convert to PIL Image for drawing text
        pil_img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Add annotations
        time_text = f"{time_ns:.1f} ns"
        draw.text((20, 1040), time_text, font=font, fill=(255, 255, 255))
        
        fov_text = f"FOV = {fov} nm"
        text_width = draw.textlength(fov_text, font=font)
        draw.text(((1024 - text_width) // 2, 1040), fov_text, font=font, fill=(255, 255, 255))
        
        dpa = time_ns * dpa_per_ns
        dpa_text = f"{dpa:.2f} dpa"
        text_width = draw.textlength(dpa_text, font=font)
        draw.text((1024 - text_width - 20, 1040), dpa_text, font=font, fill=(255, 255, 255))
        
        # Write frame
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        writer.write_frame(frame)
    
    writer.close()
    print(f"Combined video saved to {output_path}")

# Example usage
if __name__ == "__main__":
    root_directory = "data/fpa70/selected/images"
    ovito_directory = "data/fpa70/selected/images_ovito"
    create_video(root_directory, "f70_6fps.mp4", fps=6, dpa_per_ns=0.146, fov=20.0)
    create_video_with_bkg(ovito_directory, "f70_6fps_ovito.mp4", fps=6, dpa_per_ns=0.146, fov=20.0)
    create_video_combined(root_directory, ovito_directory, "f70_6fps_combined.mp4", fps=6, dpa_per_ns=0.146, fov=20.0, overlay_opacity=0.5)