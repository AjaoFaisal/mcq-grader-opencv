# 📝 MCQ Grader OpenCV

A Python project that automatically grades multiple-choice question (MCQ) sheets using **OpenCV**. It detects marked answers from images or webcam input and calculates the score in real-time.

---

## 🚀 Features
- Detects rectangular MCQ sheets and answer areas.
- Automatically identifies shaded (selected) answers.
- Compares detected answers against a predefined answer key.
- Draws correct/wrong markings on the sheet.
- Displays score on a separate score box.
- Save graded sheets with a single key press.
- Works with images or live webcam feed.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/AjaoFaisal/mcq-grader-opencv.git
cd mcq-grader-opencv

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
# Run the grader
python main.py
```

**Controls:**
- `ESC` → Quit the application.
- `S` → Save the graded output to a folder.

**Notes:**
- By default, the script uses the webcam. Set `use_webcam = False` in `main.py` to use an image.
- Modify `CORRECT_ANSWERS` in `main.py` to set the correct answer key.

---

## 📊 Output Example (Video)

[![Watch the output](https://img.youtube.com/vi/M3m-BqSMv9I/hqdefault.jpg)](https://youtu.be/M3m-BqSMv9I?feature=shared)

---

## 📂 Project Structure

```
mcq-grader-opencv/
│
├── main.py              # Main grading script
├── mcq_helpers.py       # Helper functions for contour detection and grading
├── utils.py             # Utility functions (image stacking, etc.)
├── example.png          # Example input image
├── output.png           # Example output image
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

---

## 🧠 Tech Stack
- **Python 3.x**
- **OpenCV**
- **NumPy**

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Install dependencies

```bash
pip install -r requirements.txt
```
