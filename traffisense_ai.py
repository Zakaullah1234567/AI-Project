"""
TRAFFISENSE - AI-Based Intelligent Traffic Light Control System with Emergency Vehicle Priority
An AI-powered traffic management system using YOLOv8 for real-time vehicle detection
and adaptive signal timing with emergency vehicle prioritization.
"""

import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# ======================= CONFIGURATION =======================
# All parameters can be adjusted here

# Traffic light timing parameters (in seconds)
MIN_GREEN_TIME = 10          # Minimum green light duration
MAX_GREEN_TIME = 40          # Maximum green light duration
YELLOW_TIME = 3              # Yellow light duration
EXTRA_TIME_PER_VEHICLE = 2   # Extra seconds per waiting vehicle
EMERGENCY_HOLD_TIME = 15     # How long to hold green for emergency vehicles
CYCLE_DELAY = 1              # Delay between cycles

# YOLO and detection parameters
MODEL_PATH = "yolov8n.pt"    # YOLOv8 model (will download if not present)
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence for detection
EMERGENCY_CLASSES = ['ambulance', 'fire truck']  # Emergency vehicle classes
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle','Bike', 'bicycle'] + EMERGENCY_CLASSES

# Video sources (use 0 for webcam, or paths to video files)
VIDEO_SOURCES = {
    'north': r'D:\Hello\TraffiSense\videos\traffic_north.mp4',
    'south': r'D:\Hello\TraffiSense\videos\traffic_south.mp4',
    'east': r'D:\Hello\TraffiSense\videos\traffic_east.mp4',
    'west': r'D:\Hello\TraffiSense\videos\traffic_west.mp4'
}


# For simulation (if no cameras available)
USE_SIMULATION = False          # Set to False to use actual video sources
SIMULATION_WIDTH = 640
SIMULATION_HEIGHT = 480

# Display settings
SHOW_VIDEO = True             # Show OpenCV windows with detections
CONSOLE_UPDATE_RATE = 1       # Console update frequency in seconds

# Emergency vehicle colors (BGR format)
EMERGENCY_COLOR = (0, 0, 255)     # Red for emergency vehicles
NORMAL_COLOR = (0, 255, 0)        # Green for normal vehicles
TEXT_COLOR = (255, 255, 255)      # White text
BACKGROUND_COLOR = (40, 40, 40)   # Dark gray for UI

# ======================= TRAFFIC LIGHT SYSTEM =======================

class TrafficSignal:
    """Represents a traffic signal state"""
    def __init__(self, direction):
        self.direction = direction
        self.state = "RED"
        self.green_duration = MIN_GREEN_TIME
        self.vehicle_count = 0
        self.emergency_detected = False
        self.last_detection_time = 0
        
    def update_count(self, count, has_emergency=False):
        """Update vehicle count and emergency status"""
        self.vehicle_count = count
        if has_emergency:
            self.emergency_detected = True
            self.last_detection_time = time.time()
        elif time.time() - self.last_detection_time > EMERGENCY_HOLD_TIME:
            self.emergency_detected = False
            
    def reset(self):
        """Reset for new cycle"""
        self.state = "RED"
        
    def __str__(self):
        return f"{self.direction}: {self.state} ({self.vehicle_count} vehicles)"

class VehicleDetector:
    """Handles vehicle detection using YOLOv8"""
    def __init__(self):
        print("Loading YOLOv8 model...")
        self.model = YOLO(MODEL_PATH)
        self.class_names = self.model.names
        print(f"Model loaded with {len(self.class_names)} classes")
        
    def detect_vehicles(self, frame):
        """Detect vehicles in a frame and return counts"""
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        vehicle_count = 0
        emergency_count = 0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    class_name = self.class_names[class_id].lower()
                    
                    # Check if it's a vehicle class we care about
                    if any(vehicle_class in class_name for vehicle_class in VEHICLE_CLASSES):
                        vehicle_count += 1
                        
                        # Check if it's an emergency vehicle
                        is_emergency = any(emergency_class in class_name 
                                         for emergency_class in EMERGENCY_CLASSES)
                        if is_emergency:
                            emergency_count += 1
                            
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'class': class_name,
                            'confidence': confidence,
                            'emergency': is_emergency
                        })
        
        return detections, vehicle_count, emergency_count

class TrafficCamera:
    """Manages a single camera feed and its detection"""
    def __init__(self, direction, source):
        self.direction = direction
        self.source = source
        self.detector = VehicleDetector()
        self.current_frame = None
        self.vehicle_count = 0
        self.emergency_count = 0
        self.detections = []
        self.running = True
        self.frame_queue = queue.Queue(maxsize=50)
        
    def start_capture(self):
        """Start capturing and processing video"""
        if USE_SIMULATION:
            self.simulate_traffic()
        else:
            self.capture_real_video()
            
    def capture_real_video(self):
        """Capture from real video source"""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Trying to open video: {self.source}")
            print(f"Warning: Could not open video source for {self.direction}")
            self.simulate_traffic()
            return
            
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Loop video or break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            # Resize for consistent processing
            frame = cv2.resize(frame, (SIMULATION_WIDTH, SIMULATION_HEIGHT))
            
            # Detect vehicles
            detections, count, emergency = self.detector.detect_vehicles(frame)
            
            # Update counts
            self.vehicle_count = count
            self.emergency_count = emergency
            self.detections = detections
            
            # Annotate frame
            annotated_frame = self.annotate_frame(frame.copy())
            self.current_frame = annotated_frame
            
            # Add to queue for main thread
            if not self.frame_queue.full():
                self.frame_queue.put(annotated_frame)
                
            # Small delay to prevent overloading
            time.sleep(0.03)
            
        cap.release()
        
    def simulate_traffic(self):
        """Simulate traffic when no camera is available"""
        while self.running:
            # Create a simulated frame
            frame = np.zeros((SIMULATION_HEIGHT, SIMULATION_WIDTH, 3), dtype=np.uint8)
            frame[:] = BACKGROUND_COLOR
            
            # Add some text
            cv2.putText(frame, f"{self.direction.upper()} CAMERA", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
            cv2.putText(frame, "SIMULATION MODE", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
            
            # Simulate random vehicle count
            self.vehicle_count = np.random.randint(0, 15)
            
            # Occasionally simulate emergency vehicle (5% chance)
            if np.random.random() < 0.05:
                self.emergency_count = 1
                cv2.putText(frame, "EMERGENCY VEHICLE DETECTED!", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EMERGENCY_COLOR, 2)
            else:
                self.emergency_count = 0
                
            # Simulate some bounding boxes
            for i in range(min(self.vehicle_count, 5)):
                x = np.random.randint(50, SIMULATION_WIDTH - 100)
                y = np.random.randint(200, SIMULATION_HEIGHT - 50)
                w = np.random.randint(40, 120)
                h = np.random.randint(30, 60)
                
                color = EMERGENCY_COLOR if (self.emergency_count > 0 and i == 0) else NORMAL_COLOR
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = "AMBULANCE" if color == EMERGENCY_COLOR else "VEHICLE"
                cv2.putText(frame, label, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            self.current_frame = frame
            
            # Add to queue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
                
            # Update at reasonable rate
            time.sleep(0.5)
            
    def annotate_frame(self, frame):
        """Add annotations to the frame"""
        # Draw bounding boxes
        for detection in self.detections:
            x1, y1, x2, y2 = detection['bbox']
            color = EMERGENCY_COLOR if detection['emergency'] else NORMAL_COLOR
            thickness = 3 if detection['emergency'] else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add label
            label = f"{detection['class']} {detection['confidence']:.2f}"
            if detection['emergency']:
                label = "EMERGENCY: " + label
                
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add count overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0, 180), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        cv2.putText(frame, f"Direction: {self.direction.upper()}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        cv2.putText(frame, f"Vehicles: {self.vehicle_count}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        
        if self.emergency_count > 0:
            cv2.putText(frame, f"EMERGENCY: {self.emergency_count}", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EMERGENCY_COLOR, 2)
            
        return frame
        
    def stop(self):
        """Stop the camera"""
        self.running = False

class TrafficController:
    """Main traffic control system"""
    def __init__(self):
        self.signals = {
            'north': TrafficSignal('NORTH'),
            'south': TrafficSignal('SOUTH'),
            'east': TrafficSignal('EAST'),
            'west': TrafficSignal('WEST')
        }
        
        # Initialize cameras
        self.cameras = {}
        for direction, source in VIDEO_SOURCES.items():
            self.cameras[direction] = TrafficCamera(direction, source)
            
        self.current_phase = "north"  # Current green direction
        self.cycle_count = 0
        self.emergency_active = False
        self.emergency_direction = None
        self.log_entries = []
        self.running = True
        
        # Statistics
        self.total_vehicles_processed = 0
        self.emergency_events = 0
        self.start_time = time.time()
        
    def log(self, message, emergency=False):
        """Add log entry with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        prefix = "[EMERGENCY]" if emergency else "[LOG]"
        log_entry = f"{timestamp} {prefix} {message}"
        self.log_entries.append(log_entry)
        
        # Keep only recent logs
        if len(self.log_entries) > 10:
            self.log_entries.pop(0)
            
        print(log_entry)
        return log_entry
        
    def calculate_green_time_single(self, direction):
        """Calculate green time for a single direction"""
        count = self.signals[direction].vehicle_count
        calculated_time = MIN_GREEN_TIME + (count * EXTRA_TIME_PER_VEHICLE)
        green_time = min(max(MIN_GREEN_TIME, calculated_time), MAX_GREEN_TIME)
        return green_time
        
    def check_emergency(self):
        """Check if any emergency vehicle is detected"""
        for direction, signal in self.signals.items():
            if signal.emergency_detected and not self.emergency_active:
                self.emergency_active = True
                self.emergency_direction = direction
                self.emergency_events += 1
                
                self.log(f"Emergency vehicle detected on {direction.upper()} road!", True)
                return True
                
        return False
        
    def handle_emergency(self):
        """Handle emergency vehicle priority"""
        if not self.emergency_active or not self.emergency_direction:
            return
            
        direction = self.emergency_direction
        signal = self.signals[direction]
        
        # Set all signals to RED except emergency direction
        for sig in self.signals.values():
            sig.state = "RED"
            
        # Set emergency direction to GREEN
        signal.state = "GREEN"
        self.current_phase = f"emergency-{direction}"
        
        self.log(f"Priority GREEN for {direction.upper()} - Emergency vehicle passing", True)
        
        # Hold green for emergency vehicle
        time.sleep(EMERGENCY_HOLD_TIME)
        
        # Check if emergency still present
        if not signal.emergency_detected:
            self.emergency_active = False
            self.emergency_direction = None
            self.log("Emergency cleared, resuming normal operation")
            
    def update_counts_from_cameras(self):
        """Update vehicle counts from all cameras"""
        for direction, camera in self.cameras.items():
            has_emergency = camera.emergency_count > 0
            self.signals[direction].update_count(camera.vehicle_count, has_emergency)
            
    def display_console(self):
        """Display current status in console"""
        # print("\033c", end="")  # Clear screen
        
        print("=" * 70)
        print("        TRAFFISENSE - AI Traffic Control with Emergency Priority")
        print("=" * 70)
        
        # System status
        print(f"System Status: {'EMERGENCY MODE' if self.emergency_active else 'NORMAL OPERATION'}")
        print(f"Cycle: #{self.cycle_count} | Uptime: {int(time.time() - self.start_time)}s")
        print(f"Total Vehicles: {self.total_vehicles_processed} | Emergency Events: {self.emergency_events}")
        print("-" * 70)
        
        # Traffic light status
        print("TRAFFIC SIGNAL STATUS:")
        for direction, signal in self.signals.items():
            state_color = {
                'RED': '\033[91m]',     # Red
                'YELLOW': '\033[93m]',  # Yellow
                'GREEN': '\033[92m]'    # Green
            }.get(signal.state, '\033[0m]')
            
            reset_color = '\033[0m]'
            emergency_flag = " [EMERGENCY]" if signal.emergency_detected else ""
            
            print(f"  {state_color}{direction.upper():10} {signal.state:8} {reset_color}"
                  f"- {signal.vehicle_count:3} vehicles{emergency_flag}")
        
        print("-" * 70)
        
        # Current phase info
        if self.emergency_active:
            print(f"ACTIVE PHASE: EMERGENCY PRIORITY for {self.emergency_direction.upper()}")
            print(f"Green time held: {EMERGENCY_HOLD_TIME} seconds")
        else:
            green_time = self.signals[self.current_phase.split('-')[0]].green_duration
            print(f"ACTIVE PHASE: {self.current_phase.upper().replace('-', ' ')}")
            print(f"Green time remaining: {green_time} seconds")
        
        print("-" * 70)
        
        # Recent logs
        print("RECENT EVENTS:")
        for log in self.log_entries[-5:]:
            print(f"  {log}")
            
        print("=" * 70)
        print("Press Ctrl+C to stop the system")
        print("=" * 70)
        
    def display_video_windows(self):
        """Display all camera feeds in separate windows"""
        if not SHOW_VIDEO:
            return
            
        frames = {}
        
        # Get frames from all cameras
        for direction, camera in self.cameras.items():
            try:
                if not camera.frame_queue.empty():
                    frames[direction] = camera.frame_queue.get_nowait()
            except queue.Empty:
                frames[direction] = camera.current_frame
                
        # Display in 2x2 grid if we have all frames
        if len(frames) == 4:
            # Resize all frames to same size
            resized_frames = {}
            for direction, frame in frames.items():
                if frame is not None:
                    resized = cv2.resize(frame, (420, 340))
                    resized_frames[direction] = resized
                else:
                    # Create placeholder
                    resized = np.zeros((340, 420, 3), dtype=np.uint8)
                    resized[:] = BACKGROUND_COLOR
                    cv2.putText(resized, f"{direction.upper()} - NO FEED", 
                               (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
                    resized_frames[direction] = resized
                    
            # Create grid
            top_row = np.hstack([resized_frames['north'], resized_frames['south']])
            bottom_row = np.hstack([resized_frames['east'], resized_frames['west']])
            grid = np.vstack([top_row, bottom_row])
            
            # Add title
            title = "TRAFFISENSE - Live Camera Feeds"
            if self.emergency_active:
                title += " [EMERGENCY MODE]"
                
            cv2.putText(grid, title, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("TraffiSense - Traffic Monitoring", grid)
            
        # Handle keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
            
    def run_normal_cycle(self):
        """Run one normal traffic light cycle - FIXED VERSION"""
        self.cycle_count += 1
        
        # Update counts from cameras
        self.update_counts_from_cameras()
        
        # Get current active direction
        current_direction = self.current_phase
        
        # STEP 1: Turn current green light to YELLOW
        if self.signals[current_direction].state == "GREEN":
            self.signals[current_direction].state = "YELLOW"
            self.display_console()
            self.log(f"Light for {current_direction.upper()} turning YELLOW")
            time.sleep(YELLOW_TIME)
        
        # STEP 2: Turn current direction to RED
        self.signals[current_direction].state = "RED"
        self.display_console()
        self.log(f"Light for {current_direction.upper()} turned RED")
        time.sleep(1)  # All-red clearance interval (IMPORTANT!)
        
        # STEP 3: Determine next direction (round-robin)
        directions = ['north', 'south', 'east', 'west']
        current_index = directions.index(current_direction)
        next_index = (current_index + 1) % 4
        next_direction = directions[next_index]
        
        # STEP 4: Calculate green time for next direction
        # Get count for next direction only
        next_count = self.signals[next_direction].vehicle_count
        
        # Simple calculation for single direction
        green_time = MIN_GREEN_TIME + (next_count * EXTRA_TIME_PER_VEHICLE)
        green_time = min(max(MIN_GREEN_TIME, green_time), MAX_GREEN_TIME)
        
        self.signals[next_direction].green_duration = green_time
        
        # Log the calculation
        self.log(f"Calculated green time for {next_direction.upper()}: {green_time}s "
                f"({next_count} vehicles)")
        
        # STEP 5: Switch to next direction
        self.current_phase = next_direction
        self.signals[next_direction].state = "GREEN"
        self.display_console()
        self.log(f"Light for {next_direction.upper()} turned GREEN for {green_time}s")
        
        # STEP 6: Run green light with periodic updates
        elapsed = 0
        while elapsed < green_time and self.running:
            # Update counts periodically
            if elapsed % 3 == 0:
                self.update_counts_from_cameras()
                self.display_console()
                self.display_video_windows()
            
            time.sleep(1)
            elapsed += 1
            
            # Check for emergency every second
            if self.check_emergency():
                self.handle_emergency()
                break
                
    def run(self):
        """Main system loop"""
        print("Initializing TraffiSense AI Traffic Control System...")
        
        # Start all camera threads
        camera_threads = []
        for camera in self.cameras.values():
            thread = threading.Thread(target=camera.start_capture, daemon=True)
            thread.start()
            camera_threads.append(thread)
            time.sleep(0.1)  # Stagger starts
            
        print(f"Started {len(camera_threads)} camera threads")
        time.sleep(2)  # Allow cameras to initialize
        
        # Initial state: north gets green
        self.signals['north'].state = "GREEN"
        self.update_counts_from_cameras()
        self.signals['north'].green_duration = MIN_GREEN_TIME
        self.current_phase = "north"        
        self.log("System initialized and running")
        self.log(f"Starting with {self.current_phase.upper()} phase")
        
        try:
            while self.running:
                # Check for emergency first
                if self.check_emergency():
                    self.handle_emergency()
                else:
                    self.run_normal_cycle()
                    
                # Small delay between cycles
                if self.running:
                    time.sleep(CYCLE_DELAY)
                    self.display_console()
                    self.display_video_windows()
                    
        except KeyboardInterrupt:
            self.log("System shutdown initiated by user")
        finally:
            self.shutdown()
            
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        
        # Stop all cameras
        for camera in self.cameras.values():
            camera.stop()
            
        # Close OpenCV windows
        if SHOW_VIDEO:
            cv2.destroyAllWindows()
            
        # Show summary
        self.show_summary()
        
    def show_summary(self):
        """Display system summary"""
        print("\n" + "=" * 70)
        print("                  SYSTEM SUMMARY")
        print("=" * 70)
        print(f"Total operation time: {int(time.time() - self.start_time)} seconds")
        print(f"Traffic cycles completed: {self.cycle_count}")
        print(f"Total vehicles processed: {self.total_vehicles_processed}")
        print(f"Emergency events handled: {self.emergency_events}")
        print(f"Average vehicles per cycle: {self.total_vehicles_processed/max(1, self.cycle_count):.1f}")
        print("\nFinal vehicle counts:")
        for direction, signal in self.signals.items():
            emergency_flag = " (EMERGENCY)" if signal.emergency_detected else ""
            print(f"  {direction.upper()}: {signal.vehicle_count} vehicles{emergency_flag}")
        print("\nConfiguration used:")
        print(f"  Min/Max Green Time: {MIN_GREEN_TIME}s/{MAX_GREEN_TIME}s")
        print(f"  Yellow Time: {YELLOW_TIME}s")
        print(f"  Extra time per vehicle: {EXTRA_TIME_PER_VEHICLE}s")
        print(f"  Emergency hold time: {EMERGENCY_HOLD_TIME}s")
        print(f"  Detection confidence: {CONFIDENCE_THRESHOLD}")
        print("=" * 70)
        print("TraffiSense - Making Cities Smarter and Safer")
        print("=" * 70)

def main():
    """Main entry point"""
    print("=" * 70)
    print("         TRAFFISENSE AI Traffic Control System")
    print("=" * 70)
    print("Features:")
    print("  • Real-time vehicle detection using YOLOv8")
    print("  • Adaptive traffic signal timing based on density")
    print("  • Emergency vehicle priority system")
    print("  • Four-direction intersection monitoring")
    print("  • Live video display with bounding boxes")
    print("  • Detailed console logging and statistics")
    print("\nSystem initializing...")
    
    # Check for YOLO model
    if not Path(MODEL_PATH).exists():
        print(f"\nNote: YOLO model '{MODEL_PATH}' not found.")
        print("It will be automatically downloaded on first run.")
        print("This may take a few minutes...")
        
    # Create and run controller
    controller = TrafficController()
    
    # Run in separate thread
    controller_thread = threading.Thread(target=controller.run)
    controller_thread.daemon = True
    controller_thread.start()
    
    try:
        # Keep main thread alive
        while controller_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down system...")
        controller.running = False
        time.sleep(2)

if __name__ == "__main__":
    main()