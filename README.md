# Face Recognition with FaceNet and Faiss

This project presents a facial recognition system that uses `facenet` for face encoding and the `faiss` library for efficient similarity search. The system allows adding new faces without reprocessing the entire dataset, making the system scalable and efficient, and only requires 1 image per class.
Use `yolo v8` for face detection.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Preparing Data](#preparing-data)
  - [Adding New Faces](#adding-new-faces)
- [Dataset Format](#dataset-format)

## Introduction
This project implements a face recognition system that uses Faiss (Facebook AI Similarity Search) to index and search face embeddings efficiently. It leverages `face_recognition` for face detection and encoding, making the system robust and accurate.

## Features
- **Efficient Similarity Search**: Utilizes Faiss for fast and scalable vector search.
- **Face Detection and Encoding**: Uses Yolo V8 for reliable face detection and FaceNet for feature extraction.
- **Incremental Updates**: Allows adding new faces without reprocessing the entire dataset.
- **Minimal Data Requirement**: Requires only one image per class for training.

## Installation
To run this project, ensure you have Python installed. Then, follow these steps to install the necessary dependencies:

```bash
git clone https://github.com/yourusername/face-recognition-faiss.git
cd face-recognition-faiss
pip install -r requirements.txt
```

## Usage

### Dataset Format

The dataset should be structured as follows:

``` shell
data_dir
├── class_A
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_B
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

### Preparing Data

Prepare a folder to contain all your facial data that needs to be recognized. That folder will contain different subfolders, each subfolder will correspond to a class and will contain at least 1 or more photos containing the subject's face.

Run the following script to generate the face encodings and save them to a file:

```bash
python src/prepare_dataset.py --data_dir data_dir --save_dir data --task "init"
```

This script will generate the face encodings for each face in the dataset and save them to a file. Including: save faces encoding to data/faiss_index.index and save list names to data/face_names.pkl.

### Adding New Faces

To add a new face to the system, use the provided scripts to load the existing Faiss index and names, generate the face encoding for the new image, and update the index and names list accordingly.

```bash
python src/prepare_dataset.py --data_dir data_dir --faiss_index_path data/faiss_index.index --face_names_path data/face_names.pkl --task "add"
```

### Run

```bash
python main.py
```


