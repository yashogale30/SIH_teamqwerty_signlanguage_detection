# Real-Time Translation for Indian Sign Language to Text and Speech



### Smart India Hackathon 2024  
**Problem Statement ID**: 1716  
**Problem Statement Title**: Indian Sign Language to Text/Speech Translation  
**Team Name**: qwerty  

## Project Overview

This project aims to develop a real-time translation system for Indian Sign Language (ISL) to facilitate communication between ISL users and non-signers. The solution translates ISL gestures into text and synthesized speech, thereby promoting inclusivity and accessibility for the deaf and hard-of-hearing community.

## Demo


Click on the image below to watch the demo video on YouTube:

[![Watch the Demo Video](https://img.youtube.com/vi/dkGq1RpBeQM/0.jpg)](https://youtu.be/dkGq1RpBeQM)




## Table of Contents

- [Real-Time Translation for Indian Sign Language to Text and Speech](#real-time-translation-for-indian-sign-language-to-text-and-speech)
    - [Smart India Hackathon 2024](#smart-india-hackathon-2024)
  - [Project Overview](#project-overview)
  - [Demo](#demo)
  - [Table of Contents](#table-of-contents)
  - [Technical Approach](#technical-approach)
  - [Technologies Used](#technologies-used)
  - [Feasibility and Viability](#feasibility-and-viability)
  - [Impact and Benefits](#impact-and-benefits)
  - [Future Scope](#future-scope)
  - [Contributors](#contributors)

## Technical Approach

The system leverages deep learning and computer vision techniques to detect and interpret ISL gestures in real-time:

1. **Hand Detection and Tracking**: Utilizes YOLO and MediaPipe integrated with OpenCV to accurately detect and track hand movements.
2. **Gesture Classification**: Implements a CNN-based classifier using MobileNetV2 (TensorFlow and Keras) to recognize ISL gestures.
3. **Mapping to Text**: Translates recognized gestures into corresponding text output.
4. **Text-to-Speech Conversion**: Converts text into speech using gTTS, rendering audio via Pygame.

## Technologies Used

- **YOLO (You Only Look Once)**: For fast and efficient object detection.
- **MediaPipe**: For hand tracking and landmark detection.
- **OpenCV**: For real-time computer vision tasks.
- **TensorFlow & Keras**: Deep learning frameworks used to train the gesture classifier with MobileNetV2.
- **gTTS (Google Text-to-Speech)**: Converts text into speech.
- **Pygame**: Outputs audio for the synthesized speech.

## Feasibility and Viability

- **Feasibility**: The use of YOLO and CNNs for gesture recognition is achievable, though accuracy depends on training with a diverse dataset.
- **Challenges**:
  - **Accuracy**: Gesture recognition accuracy varies with background changes.
  - **Performance**: Real-time processing may experience delays.
  - **Integration**: Complex coordination among components.
- **Solutions**:
  - Train with varied backgrounds for better accuracy.
  - Use hardware acceleration and optimization to reduce latency.
  - Employ a modular design for smoother integration and maintenance.

## Impact and Benefits

This project holds the potential to significantly enhance communication for the deaf and hard-of-hearing community by translating ISL gestures into text and speech in real time.

- **Inclusivity**: Enables more inclusive interactions and social participation.
- **Access to Opportunities**: Broadens job prospects and educational access for ISL users.
- **Cost-Effective**: Reduces dependency on physical translation services.

## Future Scope
- Multi-Language Support: Extend translation to additional spoken languages.
- Improved Accuracy: Implement advanced models and larger datasets to increase recognition accuracy.
- Mobile App Version: Develop a mobile application to enhance accessibility and portability.

## Contributors
- [Yash Ogale](https://github.com/yashogale30)
- [Harsh Ogale](https://github.com/harshogale04)
- [Rudrakshi Kubde](https://github.com/RudrakshiKubde)







