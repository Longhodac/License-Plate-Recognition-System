from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from skimage import measure
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.transform import resize
import base64
import io
from PIL import Image
import json
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class LicensePlateProcessor:
    def __init__(self):
        self.letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        # Mock model for demonstration - in real app, load actual trained model
        self.model = None
        
    def load_model(self):
        """Load trained SVM model - mock implementation"""
        # In real implementation, uncomment this:
        # try:
        #     self.model = joblib.load('models/svc/svc.pkl')
        # except:
        #     print("Model not found - using mock recognition")
        pass
    
    def preprocess_image(self, image_path):
        """Convert image to grayscale and create binary image"""
        # Read image as grayscale
        car_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if car_image is None:
            raise ValueError("Could not load image")
        
        # Normalize to 0-255 range
        gray_car_image = car_image.astype(np.float64)
        
        # Calculate Otsu threshold
        threshold_value = threshold_otsu(gray_car_image)
        binary_car_image = gray_car_image > threshold_value
        
        return {
            'original': car_image,
            'gray': gray_car_image,
            'binary': binary_car_image,
            'threshold': threshold_value
        }
    
    def detect_license_plates(self, binary_image, original_shape):
        """Detect license plate regions using connected component analysis"""
        # Label connected components
        label_image = measure.label(binary_image)
        
        # Calculate plate dimensions (8-20% height, 15-40% width)
        min_height = 0.08 * original_shape[0]
        max_height = 0.2 * original_shape[0]
        min_width = 0.15 * original_shape[1]
        max_width = 0.4 * original_shape[1]
        
        plate_objects = []
        plate_coordinates = []
        
        # Analyze each region
        for region in regionprops(label_image):
            if region.area < 50:
                continue
                
            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            region_width = max_col - min_col
            
            # Check if region matches license plate dimensions
            if (region_height >= min_height and region_height <= max_height and
                region_width >= min_width and region_width <= max_width and
                region_width > region_height):
                
                plate_objects.append(binary_image[min_row:max_row, min_col:max_col])
                plate_coordinates.append((min_row, min_col, max_row, max_col))
        
        return plate_objects, plate_coordinates
    
    def segment_characters(self, license_plate):
        """Segment individual characters from license plate"""
        if license_plate is None or license_plate.size == 0:
            return [], []
        
        # Invert the license plate
        inverted_plate = np.invert(license_plate)
        
        # Label connected components
        labelled_plate = measure.label(inverted_plate)
        
        # Character dimension constraints
        min_height = 0.35 * license_plate.shape[0]
        max_height = 0.60 * license_plate.shape[0]
        min_width = 0.05 * license_plate.shape[1]
        max_width = 0.15 * license_plate.shape[1]
        
        characters = []
        character_positions = []
        
        for region in regionprops(labelled_plate):
            y0, x0, y1, x1 = region.bbox
            region_height = y1 - y0
            region_width = x1 - x0
            
            if (region_height > min_height and region_height < max_height and
                region_width > min_width and region_width < max_width):
                
                # Extract character region
                roi = inverted_plate[y0:y1, x0:x1]
                
                # Resize to 20x20 for recognition
                resized_char = resize(roi, (20, 20))
                characters.append(resized_char)
                character_positions.append(x0)  # For ordering
        
        # Sort characters by x position (left to right)
        if character_positions:
            sorted_pairs = sorted(zip(character_positions, characters))
            characters = [char for _, char in sorted_pairs]
            character_positions = [pos for pos, _ in sorted_pairs]
        
        return characters, character_positions
    
    def recognize_characters(self, characters):
        """Recognize characters using trained model"""
        if not characters:
            return ""
        
        # Mock recognition - replace with actual model prediction
        if self.model is None:
            # Generate mock license plate
            mock_chars = ['A', 'B', 'C', '1', '2', '3']
            return ''.join(mock_chars[:len(characters)])
        
        # Real implementation would be:
        # predictions = []
        # for char in characters:
        #     char_flat = char.reshape(1, -1)
        #     prediction = self.model.predict(char_flat)[0]
        #     predictions.append(prediction)
        # return ''.join(predictions)
        
    def array_to_base64(self, array, is_binary=False):
        """Convert numpy array to base64 string for web display"""
        if is_binary:
            # Convert boolean array to uint8
            array = (array * 255).astype(np.uint8)
        else:
            # Ensure array is uint8
            array = array.astype(np.uint8)
        
        # Create PIL image
        if len(array.shape) == 2:  # Grayscale
            img = Image.fromarray(array, mode='L')
        else:  # RGB
            img = Image.fromarray(array, mode='RGB')
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

# Initialize processor
processor = LicensePlateProcessor()
processor.load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            result = process_license_plate(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo/<demo_type>')
def demo_image(demo_type):
    """Generate demo image for testing"""
    try:
        # Create demo image
        demo_plates = {
            'car1': 'ABC123',
            'car2': 'XYZ789', 
            'car3': 'DEF456'
        }
        
        if demo_type not in demo_plates:
            return jsonify({'error': 'Invalid demo type'}), 400
        
        # Generate synthetic car image with license plate
        img = generate_demo_image(demo_plates[demo_type])
        
        # Save temporarily and process
        temp_path = f"temp_demo_{demo_type}.png"
        cv2.imwrite(temp_path, img)
        
        result = process_license_plate(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_demo_image(plate_text):
    """Generate synthetic car image with license plate"""
    # Create car image
    img = np.ones((300, 400, 3), dtype=np.uint8) * 70  # Dark gray background
    
    # Draw car body
    cv2.rectangle(img, (50, 80), (350, 230), (45, 62, 80), -1)
    
    # Draw license plate area
    plate_x, plate_y = 150, 200
    plate_w, plate_h = 100, 40
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (255, 255, 255), -1)
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (0, 0, 0), 2)
    
    # Add some noise and texture
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Convert to grayscale for processing
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Add license plate text (this will be detected by our algorithm)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(plate_text, font, 0.7, 2)[0]
    text_x = plate_x + (plate_w - text_size[0]) // 2
    text_y = plate_y + (plate_h + text_size[1]) // 2
    cv2.putText(gray_img, plate_text, (text_x, text_y), font, 0.7, (0), 2)
    
    return gray_img

def process_license_plate(image_path):
    """Main processing function"""
    try:
        # Step 1: Preprocess image
        processed = processor.preprocess_image(image_path)
        
        # Step 2: Detect license plates
        plates, coordinates = processor.detect_license_plates(
            processed['binary'], processed['original'].shape
        )
        
        # Step 3: Process best plate candidate
        characters = []
        recognized_text = "No plate detected"
        
        if plates:
            # Use the first detected plate (in real app, implement scoring)
            best_plate = plates[0]
            characters, positions = processor.segment_characters(best_plate)
            
            if characters:
                recognized_text = processor.recognize_characters(characters)
        
        # Convert images to base64 for web display
        result = {
            'success': True,
            'steps': {
                'original': processor.array_to_base64(processed['original']),
                'binary': processor.array_to_base64(processed['binary'], is_binary=True),
                'detection': create_detection_image(processed['original'], coordinates),
                'segmentation': create_segmentation_image(plates[0] if plates else None),
                'characters': [processor.array_to_base64((char * 255).astype(np.uint8)) 
                             for char in characters] if characters else [],
            },
            'result': recognized_text,
            'plates_found': len(plates),
            'characters_found': len(characters)
        }
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def create_detection_image(original_img, coordinates):
    """Create image with detected license plates highlighted"""
    # Convert to RGB for drawing
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB) if len(original_img.shape) == 2 else original_img.copy()
    
    # Draw rectangles around detected plates
    for coord in coordinates:
        min_row, min_col, max_row, max_col = coord
        cv2.rectangle(img_rgb, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)
    
    return processor.array_to_base64(img_rgb)

def create_segmentation_image(license_plate):
    """Create image showing character segmentation"""
    if license_plate is None:
        # Create empty image
        empty = np.ones((100, 300), dtype=np.uint8) * 255
        return processor.array_to_base64(empty)
    
    # Invert plate for display
    inverted_plate = np.invert(license_plate)
    
    # Resize for better visibility
    display_plate = cv2.resize(inverted_plate.astype(np.uint8), (300, 100))
    
    # Convert to RGB for drawing rectangles
    display_rgb = cv2.cvtColor(display_plate, cv2.COLOR_GRAY2RGB)
    
    # Mock character segmentation boxes
    chars, positions = processor.segment_characters(license_plate)
    if chars:
        char_width = 300 // len(chars)
        for i in range(len(chars)):
            x = i * char_width
            cv2.rectangle(display_rgb, (x + 5, 10), (x + char_width - 5, 90), (255, 0, 0), 2)
    
    return processor.array_to_base64(display_rgb)

if __name__ == '__main__':
    app.run(debug=True)