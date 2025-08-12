# 🚗 Automatic Car License Plate Recognition System (ACLPR-System)

## 📌 Overview
The **ACLPR-System** is an AI-powered application that detects and recognizes vehicle license plates from images or video streams.  
It uses **YOLOv8** for license plate detection and **OCR (Optical Character Recognition)** for reading plate numbers, enabling automated and accurate vehicle identification for various applications such as security systems, toll booths, parking management, and traffic monitoring.

---

## ✨ Features
- **Real-time license plate detection** using YOLOv8.
- **Text extraction** from plates using OCR.
- **High accuracy** on diverse datasets (cars, bikes, trucks).
- **Supports images and videos** as input.
- **Custom dataset training support** for improved performance.
- **Modular design** for easy integration with other systems.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch
- **Detection Model:** YOLOv8 (Ultralytics)
- **OCR Engine:** EasyOCR / Tesseract
- **Visualization:** OpenCV, Matplotlib
- **Environment Management:** Virtualenv

---

## 📂 Project Structure
ACLPR-System/
│
├── dataset_main/ # Dataset folder (train/valid/test images and labels)
├── models/ # YOLO model weights (tracked with Git LFS)
├── runs/ # YOLO training outputs
├── detect.py # Script for running detection
├── train.py # Script for training YOLO model
├── predict.py # Script for inference
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitattributes # LFS configuration for large files

yaml
Copy
Edit

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/CodeWithWaleed-611/ACLPR-System.git
cd ACLPR-System
2️⃣ Set Up Virtual Environment
bash
Copy
Edit
python -m venv .venv
Activate:

Windows (PowerShell):

bash
Copy
Edit
.venv\Scripts\activate
Linux/Mac:

bash
Copy
Edit
source .venv/bin/activate
3️⃣ Install Dependencies
bash
Copy
Edit
pip install --upgrade pip
pip install -r requirements.txt
4️⃣ (Optional) Install Git LFS for Large Files
If not already installed:

bash
Copy
Edit
git lfs install
▶️ Usage
Run Detection on an Image
bash
Copy
Edit
python predict.py --source path/to/image.jpg
Run Detection on a Video
bash
Copy
Edit
python predict.py --source path/to/video.mp4
Train the Model
bash
Copy
Edit
python train.py --data config.yaml --epochs 50 --weights yolov8n.pt
📊 Dataset
The dataset contains images and annotated labels in YOLO format.

You can replace dataset_main/ with your own dataset for custom training.

Large dataset files are tracked via Git LFS.

📦 Model Weights
Model weights (.pt files) are stored using Git LFS and may not be downloaded automatically with git clone.
To pull them:

bash
Copy
Edit
git lfs pull
📸 Example Output
Detection Result:

arduino
Copy
Edit
Vehicle detected → Plate detected → OCR extracted text: ABC-1234
Plates are highlighted with bounding boxes, and recognized text is displayed.

🚀 Applications
Parking management

Toll booth automation

Security surveillance

Traffic monitoring

Access control systems

🤝 Contributing
Pull requests are welcome! For major changes:

Fork the repository

Create your feature branch (git checkout -b feature-name)

Commit your changes (git commit -m "Add feature")

Push to your branch (git push origin feature-name)

Open a Pull Request


📧 Contact
Developed by Waleed Ahmad
GitHub: CodeWithWaleed-611
Email: waleedahmad4109@gmail.com
