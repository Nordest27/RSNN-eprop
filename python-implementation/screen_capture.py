import mss
import numpy as np
import cv2
import time

class ScreenCaptureASCII:
    def __init__(self, region, ascii_width=80, ascii_height=24):
        """
        region: dict with 'top', 'left', 'width', 'height'
        ascii_width/height: dimensions of ASCII output
        """
        self.sct = mss.mss()
        self.region = region
        self.ascii_width = ascii_width
        self.ascii_height = ascii_height
        
        # ASCII characters from darkest to brightest
        self.ascii_chars = " .:-=+*#%@"
        
    def capture_frame(self):
        """Capture a single frame and return as grayscale numpy array"""
        screenshot = self.sct.grab(self.region)
        frame = np.array(screenshot)
        
        # Convert BGRA to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return gray
    
    def frame_to_ascii(self, frame):
        """Convert grayscale frame to ASCII art"""
        # Resize frame to ASCII dimensions
        resized = cv2.resize(frame, (self.ascii_width, self.ascii_height))
        
        # Normalize to 0-255 range
        normalized = resized.astype(np.float32)
        
        # Map pixel values to ASCII characters
        ascii_frame = []
        for row in normalized:
            ascii_row = ""
            for pixel in row:
                # Map pixel brightness (0-255) to ASCII character index
                char_index = int((pixel / 255.0) * (len(self.ascii_chars) - 1))
                ascii_row += self.ascii_chars[char_index]
            ascii_frame.append(ascii_row)
        
        return ascii_frame
    
    def display_ascii_frame(self, ascii_frame):
        """Print ASCII frame to console"""
        # Clear console (works on most terminals)
        print("\033[2J\033[H", end="")
        
        for row in ascii_frame:
            print(row)
    
    def live_capture(self, fps=10):
        """Continuously capture and display frames"""
        frame_time = 1.0 / fps
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                frame = self.capture_frame()
                
                # Convert to ASCII
                ascii_frame = self.frame_to_ascii(frame)
                
                # Display
                self.display_ascii_frame(ascii_frame)
                
                # Show frame info
                print(f"Frame shape: {frame.shape}, Min: {frame.min()}, Max: {frame.max()}")
                print("Press Ctrl+C to stop")
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nStopping capture...")

# Example usage
if __name__ == "__main__":
    # Define screen region to capture (adjust these values)
    # Format: {"top": y, "left": x, "width": w, "height": h}
    region = {
        "top": 100,    # Start 100 pixels from top
        "left": 100,   # Start 100 pixels from left
        "width": 400,  # Capture 400 pixels wide
        "height": 300  # Capture 300 pixels tall
    }
    
    # Create capture instance
    capture = ScreenCaptureASCII(region, ascii_width=200, ascii_height=150)
    
    # Test single frame capture
    print("Testing single frame capture...")
    frame = capture.capture_frame()
    ascii_frame = capture.frame_to_ascii(frame)
    capture.display_ascii_frame(ascii_frame)
    print(f"Captured frame shape: {frame.shape}")
    print(f"Pixel value range: {frame.min()} - {frame.max()}")
    
    input("\nPress Enter to start live capture (Ctrl+C to stop)...")
    
    # Start live capture
    capture.live_capture(fps=5)