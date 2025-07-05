<div align="center">
  <h1>🚗 Speed Limit Sign Detection using OpenCV & Tesseract</h1>
</div>

<p align="center">
  <em>A real-time system for detecting speed limit signs in video streams and extracting numeric values using OCR.</em>
</p>

---

## 📌 Overview

This project uses **OpenCV** and **Tesseract OCR** to detect circular **speed limit signs** in video footage, extract the speed values using image processing and display them in real-time.

Key capabilities include:

- Red color-based circle detection to find speed signs  
- OCR (Optical Character Recognition) using Tesseract for digit extraction  
- Visualization of detected signs with bounding boxes and speed labels  
- Automatic cooldown mechanism to avoid duplicate detections

---

## 🧰 Requirements

- Python 3.x  
- Tesseract OCR installed and configured  
- OpenCV  
- NumPy  
- pytesseract

To install Python dependencies:

```bash
pip install -r requirements.txt
```

> ⚠️ Make sure to update the `TESSERACT_PATH` in the script to your local Tesseract executable path.

---

## 🔧 Installation & Setup

1. **Clone the repository**:

```bash
git clone https://github.com/batuhanisiklar/speed-limit-detection.git
cd speed-limit-detection
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR** (if not already installed):

- Download from: https://github.com/tesseract-ocr/tesseract  
- Set the path inside the script:

```python
TESSERACT_PATH = r"D:\Your\Tesseract\Path\tesseract.exe"
```

---

## ▶️ Usage

Run the main detection script:

```bash
python main.py
```

You can replace the `VIDEO_PATH` inside the script to use your own footage:

```python
VIDEO_PATH = "video3.mp4"
```

---

## 📁 Project Structure

```
speed-limit-detection/
│
├── video3.mp4                # Sample video file
├── main.py                   # Main detection script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ✨ Features

- ✅ Real-time video processing
- ✅ Circle detection via Hough Transform
- ✅ Text recognition with digit whitelist
- ✅ Cooldown mechanism to avoid repetitive OCR
- ✅ Optional secondary ROI window to view detected signs

---

## 🙋‍♂️ Contributing

Contributions are welcome! Feel free to fork this project and submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

| Platform | Username / Email | Link |
|----------|------------------|------|
| 📧 Email | `batuhanisiklar0@gmail.com` | [Send Email](mailto:batuhanisiklar0@gmail.com) |
| 💼 LinkedIn | `Batuhan Işıklar` | [LinkedIn Profile](https://www.linkedin.com/in/batuhanisiklar/) |

---

> Made with ❤️ by Batuhan Işıklar
