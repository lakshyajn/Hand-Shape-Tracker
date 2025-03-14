# ✋ Precise Hand Tracking with Shape Recognition 🖌️

## 🌟 Overview

This application uses computer vision and machine learning to create an intuitive hand gesture-based drawing and measurement tool. Using just your webcam, you can draw shapes in the air, measure distances between your fingers, and calculate approximate volumes - all without touching your computer!

**Perfect for:**
- Educational demonstrations
- Interactive presentations
- Touchless interfaces
- Exploring computer vision capabilities

## ✨ Features

### 🎨 Real-time Drawing
- Draw in the air by raising a single finger
- Automatic shape recognition for common geometries
- Perfect shape rendering based on your hand gestures

### 📏 Distance Measurement
- Raise two fingers to measure the distance between them
- Displays measurements in both pixels and centimeters
- Real-time updates as you move your fingers

### 📦 Volume Calculation
- Make a fist to calculate approximate volume
- Visual representation of the calculated space
- Useful for quick spatial estimations

### 🔍 Shape Recognition
The application can recognize multiple shapes:
- Lines
- Triangles
- Rectangles
- Squares
- Circles
- Diamonds
- Stars
- Ellipses
- Arrows
- Pentagons
- Hexagons
- Curved lines
- Spirals
- Complex polygons

## 🛠️ Technologies Used

- **OpenCV**: For image processing and computer vision tasks
- **MediaPipe**: For accurate hand tracking and landmark detection
- **NumPy**: For efficient numerical operations
- **Scikit-learn**: For clustering algorithms
- **Shapely**: For geometric operations
- **SciPy**: For computational geometry

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python numpy mediapipe scikit-learn shapely scipy
```

### Running the Application

1. Clone this repository
2. Navigate to the project directory
3. Run the main script:

```bash
python main.py
```

## 💡 How to Use

### Basic Controls:
- **Raise 1 finger**: Start drawing mode
- **Raise 2 fingers**: Measure distance between fingertips
- **Make a fist**: Calculate approximate volume
- **Press 'q'**: Quit the application

### Drawing Tips:
1. Raise only one finger (index finger works best)
2. Move your finger to draw in the air
3. Make a fist or raise multiple fingers to stop drawing
4. The application will automatically recognize and perfect your shape

## 📋 Technical Details

### Hand Tracking
The application uses MediaPipe's hand tracking solution to detect 21 landmarks on each hand. These landmarks are used to:
- Detect which fingers are raised
- Track fingertip positions for drawing
- Calculate distances and volumes

### Shape Recognition
Shapes are recognized using geometric analysis:
- **Circularity**: For detecting circles
- **Corner counting**: For polygons like triangles and rectangles
- **Aspect ratio analysis**: To distinguish squares from rectangles
- **Convex hull properties**: For more complex shapes

### Distance Calculation
Distances are calculated using:
- Euclidean distance formula for pixel measurements
- Approximate conversion to centimeters (can be calibrated for your specific camera)

## 🔮 Future Improvements

- [ ] Color selection using hand gestures
- [ ] Saving drawings to file
- [ ] Multi-hand interaction
- [ ] 3D shape recognition
- [ ] More precise measurements with camera calibration
- [ ] UI controls through gestures

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

> 💪 **Create, measure, and interact - all without touching your computer!**

> 🌈 **The future of human-computer interaction is here.**