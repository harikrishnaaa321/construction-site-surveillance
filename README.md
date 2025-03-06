# Construction Site Surveillance Using Deep Learning

## Check this out on Hugging Face
[Hugging Face Spaces - Construction Site Surveillance](https://huggingface.co/spaces/harikrishnaaa321/sitesurvilleace)

## Overview
This project focuses on monitoring construction sites using deep learning techniques for enhanced security, worker safety, and compliance enforcement. The model is trained to detect unsafe behavior, unauthorized personnel, and hazardous conditions in real time.

## Features
- Real-time detection of safety violations and unauthorized access.
- Alerts and notifications for security breaches.
- Supports multiple input formats (live camera feeds, recorded videos, images).
- Integration with existing surveillance systems.

## Dataset
The model is trained using datasets containing labeled video clips and images of construction sites, including safety violations and normal activities.

### Dataset Download
You can download the dataset using:
```bash
!wget https://www.kaggle.com/api/v1/datasets/download/mohamedmustafa/real-life-violence-situations-dataset
```

## Technologies Used
- **Programming Language:** Python
- **Deep Learning Frameworks:** TensorFlow/Keras or PyTorch
- **Computer Vision:** OpenCV, NumPy
- **Model Architectures:** CNN, YOLO, Faster R-CNN
- **Visualization:** Matplotlib, Seaborn

## Model Architecture
![Model Architecture](https://github.com/user-attachments/assets/17d2f33a-00b9-4872-af7d-b671fe41e251)

The model consists of:
1. **Feature Extraction:** Using CNN-based architectures (e.g., YOLO, ResNet) to extract spatial features.
2. **Object Detection:** Detecting workers, equipment, and safety gear.
3. **Classification & Alerts:** Determining safety violations and triggering alerts.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/construction-surveillance.git
   cd construction-surveillance
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and preprocess the dataset.

## Usage
To run the model on a sample video:
```bash
python detect_safety.py --video sample_site.mp4
```
For real-time detection via webcam:
```bash
python detect_safety.py --webcam
```

## Evaluation Metrics
![Evaluation Metrics 1](https://github.com/user-attachments/assets/4d26b055-77f6-4621-9d5e-828c0ee9acf7)
![Evaluation Metrics 2](https://github.com/user-attachments/assets/f791af41-9e13-41ba-8dd8-6d7398dc5c26)
![Confusion Matrix](https://github.com/user-attachments/assets/59db4362-9db4-405e-a088-30411ebd93aa)

The model performance is evaluated using:
- Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix

## Results
[Download Results CSV](https://github.com/user-attachments/files/19107580/results.1.csv)

## Future Enhancements
- Improve model accuracy with larger datasets.
- Optimize real-time inference speed.
- Deploy as a web application or mobile app.

## Contributing
Contributions are welcome! Feel free to fork the repository, create issues, and submit pull requests.

## License
This project is licensed under the MIT License.

## Contact
For queries, reach out via [email/LinkedIn/GitHub].

