# A Self-Supervised Learning approach for Stress Recognition in Children with Special Needs

## 🎯 Research Context

This work addresses the critical challenge of stress recognition during therapeutic activities for children with special needs. Stress can significantly impact a child's ability to engage in physiotherapy or rehabilitation exercises, making early detection essential for optimizing treatment outcomes.


The study builds upon the AKTIVES dataset, a multimodal resource containing physiological signals and video recordings of children with different special needs during serious game therapy, available at [A physiological signal database of children with different special needs for stress recognition](https://www.nature.com/articles/s41597-023-02272-2#Sec10).

**Dataset**: 
Three expert occupational therapists assigned stress Labels as Stress/No Stress each 10 seconds, each label is representative of the 10 seconds interval.

AKTIVES dataset with 52 participants with various conditions:
- Typical Development (20 participants)
- Intellectual Disabilities (15 participants)  
- Dyslexia (13 participants)
- Brachial Plexus Injury (4 participants)


## 🏗️ Technical Architecture

The pipeline implements a novel self-supervised learning approach that addresses the limitations of traditional supervised methods:

```
Raw Videos → Preprocessing → Feature Extraction → Training → Results
    ↓              ↓              ↓                 ↓         ↓
  Expert         Frame      DINOv2/Emotion         ML       Stress
 Annotations   Extraction      /Landmarks         Models   Classification
```

**Innovation**: Uses DINOv2 Vision Transformer for self-supervised visual representation learning, capturing temporal variations that supervised CNNs miss.

## 📁 Repository Structure

```
├── 1.Preprocessing/           # Video processing and label synchronization
├── 2.Feature_Extraction/      # DINOv2, emotions, landmarks
├── 3.Training/                # ML training pipeline
├── Results/                   # Experiment results
└── Processed Data/            # Datasets obtained from the Preprocessing
```

## 🔬 Feature Extraction Methods

### 1. DINOv2 Self-Supervised Features (Primary Innovation)
- **Model**: Meta AI DINOv2 Vision Transformer
- **Dimensions**: 768 features per frame
- **Regions**: Full frame, face-only, upper body
- **Advantage**: Captures temporal variations without manual supervision
- **Performance**: Best overall results across all experiments
- **Reference**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | [Hugging Face Implementation](https://huggingface.co/facebook/dinov2)

### 2. Emotion Features (FER)
- **Model**: Pre-trained facial emotion recognition
- **Features**: 6 emotion probability distributions
- **Classes**: Happy, Sad, Angry, Fear, Surprise, Disgust
- **Integration**: Combines effectively with DINOv2 features
- **Reference**: The model used is a pre-trained VGG-Face model found at [Stress Recognition from Facial Images in Children](https://github.com/FidanVural/Stress-Recognition-from-Facial-Images-in-Children)

### 3. Expression & Pose Landmarks
- **Tool**: MediaPipe facial and pose analysis
- **Features**: 468 facial landmarks + 33 pose landmarks
- **Metrics**: Eye aspect ratio, mouth aspect ratio, head tilt, shoulder alignment, face symmetry
- **Use Case**: Geometric features for stress expression analysis
- **Reference**: [MediaPipe](https://mediapipe.dev/)

## ⏱️ Temporal Representation Strategies

The pipeline explores multiple temporal aggregation approaches to capture stress dynamics:

### 1. Baseline 10 Seconds
- **Description**: Single frame at 10th second of each interval
### 2. Average per Interval  
- **Description**: Averaging all features from each 10 frame interval into one.

### 3. Triple Intervals
- **Description**: Three samples per interval (early, middle, late)
### 4. Overlapping Intervals
- **Description**: Three overlapping intervals of 5-seconds.



## 🚀 Quick Start Guide

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Preprocessing Pipeline
```bash
cd "1.Preprocessing"
python main.py

# you can also run the files separately starting from label extraction to video processing and face extraction
# if you would like to use the upper body frames too run also the upper body extraction
```

### 3. Feature Extraction
```bash
cd "2.Feature_Extraction"

# Extract DINOv2 features
python DINOv2_Feature_Extraction/2_Full_Feature_Extraction/dinov2_feature_extractor.py

# Extract emotions and landmarks
python Emotion_Extraction/fer_emotion_extractor.py
python Landmark_Extraction/facial_expression_pose_analyzer.py
```

### 4. Training & Evaluation
```bash
cd "Training"
python train.py
# Configure your preferred experiments following the guidelines in the respective ReadMe
# Train.py then runs all configured experiments
```

## 📋 System Requirements

- **Python**: 3.8+ required
- **GPU**: GPU with CUDA support (recommended for DINOv2)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: SSD for faster I/O operations
- **Dependencies**: PyTorch, TensorFlow, OpenCV, MediaPipe, Scikit-learn

## 📚 Documentation

- [Preprocessing Pipeline](1.Preprocessing/README.md) - Video processing and label synchronization
- [Feature Extraction](2.Feature_Extraction/README.md) - Multimodal feature extraction pipeline
- [Training Pipeline](3.Training/README.md) - Unified ML training and evaluation

## 🔗 Citation

If you use this work, please cite:

**This Repository:**
```bibtex
@article{SSLaktives2025,
  title={A Self-Supervised Learning approach for Stress Recognition in Children with Special Needs},
  author={Ercolani L.},
  journal={Journal Name},
  year={2025}
}
```

**AKTIVES Dataset:**
```bibtex
@article{coskun2023aktives,
  title={A physiological signal database of children with different special needs for stress recognition},
  author={Coşkun, Buket and Ay, Sevket and Erol Barkana, Duygun and Bostanci, Hilal and Uzun, İsmail and Oktay, Ayse Betul and Tuncel, Basak and Tarakci, Devrim},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={382},
  year={2023},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41597-023-02272-2}
}
```

**Key Technologies:**
- **DINOv2**: [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | [Hugging Face Implementation](https://huggingface.co/facebook/dinov2)
- **FER Model**: [Stress Recognition from Facial Images in Children](https://github.com/FidanVural/Stress-Recognition-from-Facial-Images-in-Children)
- **MediaPipe**: [Machine Learning Solutions](https://mediapipe.dev/)


**Research Note**: This work achieves state-of-the-art performance in stress recognition for children with special needs using visual data in a transparent, reliable, and reproducible way. The self-supervised learning approach significantly outperforms traditional supervised methods.

**Acknowledgments**: The preprocessing pipeline, particularly the label extraction and expert consensus approach, was inspired by the [AKTIVES Dataset repository](https://github.com/hiddenslate-dev/aktives-dataset-2022/tree/main) by hiddenslate-dev, which provides the original implementation for expert label synchronization and validation.

## 📚 References

### Datasets
- **AKTIVES Dataset**: [Coşkun et al. (2023)](https://www.nature.com/articles/s41597-023-02272-2#Sec10) - A physiological signal database of children with different special needs for stress recognition

### Core Technologies
- **DINOv2**: [Oquab et al. (2023)](https://arxiv.org/abs/2304.07193) - Learning Robust Visual Features without Supervision | [Hugging Face Implementation](https://huggingface.co/facebook/dinov2)
- **FER Model**: [Vural et al.](https://github.com/FidanVural/Stress-Recognition-from-Facial-Images-in-Children) - Stress Recognition from Facial Images in Children
- **MediaPipe**: [Google Research](https://mediapipe.dev/) - Machine Learning Solutions for Computer Vision

