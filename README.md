# Awesome Point Cloud [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A comprehensive, curated collection of point cloud processing resources, papers, datasets, and tools.

---

## Abstract
This document presents a curated list of high-quality resources for **point cloud analysis and processing**. The recent surge in 3D data research has led to significant advancements in understanding and utilizing point clouds for applications such as autonomous driving, robotics, and augmented reality. This repository organizes fundamental and advanced research papers, datasets, tools, and tutorials to support researchers and practitioners in exploring and benchmarking point cloud processing techniques. Contributions from the community are encouraged to keep this repository up to date with emerging work.

## Acknowledgments
We extend our gratitude to the research and open-source communities who have contributed to the development of point cloud analysis. This repository compiles and builds on foundational work from renowned researchers and organizations. We also acknowledge the authors of key papers and software libraries included in this list, as well as contributors who help maintain and expand the repository.

---

## Table of Contents
- [Introduction](#introduction)
- [Research Papers](#research-papers)
  - [Surveys](#surveys)
  - [Benchmarks](#benchmarks)
  - [Core Methods](#core-methods)
    - [Classification](#classification)
    - [Segmentation](#segmentation)
    - [Object Detection](#object-detection)
    - [Registration and SLAM](#registration-and-slam)
  - [Self-Supervised Learning](#self-supervised-learning)
- [Datasets](#datasets)
- [Tools and Libraries](#tools-and-libraries)
- [Applications](#applications)
- [Contributing](#contributing)
- [References](#references)

---

## Introduction
Point clouds are essential for 3D data analysis, providing a foundation for diverse applications. This repository compiles essential resources, categorizing them by type and year, and includes tools for benchmarking and comparisons.

## Research Papers

### Surveys
- **2020**: *Point Cloud Processing for 3D Perception: A Survey* - Comprehensive survey on point cloud processing techniques across applications ([link](https://arxiv.org/abs/2006.07641)).
- **2019**: *A Review of Point Cloud Registration* - Covers traditional and deep learning-based registration approaches ([link](https://arxiv.org/abs/1910.06207)).

### Benchmarks
- **2021**: *Benchmarking 3D Deep Learning Frameworks* - A benchmark of multiple 3D DL architectures ([link](https://arxiv.org/abs/2101.07521)).
- **2020**: *Comparative Study of Point Cloud Segmentation Methods* - Benchmark of segmentation techniques on popular datasets ([link](https://arxiv.org/abs/2008.09627)).

### Core Methods

#### Classification
- **PointNet** (2017): *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation* ([link](https://arxiv.org/abs/1612.00593))
- **PointNet++** (2017): *PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space* ([link](https://arxiv.org/abs/1706.02413))

#### Segmentation
- **SegCloud** (2017): *SegCloud: Semantic Segmentation of 3D Point Clouds* ([link](https://arxiv.org/abs/1710.07563))
- **RandLA-Net** (2020): *Efficient Semantic Segmentation of Large-Scale Point Clouds* ([link](https://arxiv.org/abs/1911.11236))

#### Object Detection
- **VoxelNet** (2017): *VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection* ([link](https://arxiv.org/abs/1711.06396))
- **SECOND** (2018): *SECOND: Sparsely Embedded Convolutional Detection* ([link](https://arxiv.org/abs/1804.00677))

#### Registration and SLAM
- **ICP** (1992): *Iterative Closest Point for Point Cloud Registration*
- **DeepVCP** (2019): *DeepVCP: An End-to-End Deep Neural Network for Point Cloud Registration* ([link](https://arxiv.org/abs/1902.03304))

### Self-Supervised Learning
- **BYOL-Point** (2021): *BYOL for Point Clouds* - Adapts BYOL framework to point clouds for SSL ([link](https://arxiv.org/abs/2103.14142))
- **PointContrast** (2020): *PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding* ([link](https://arxiv.org/abs/2007.10985))

## Datasets
- **ModelNet** - CAD models for 3D object classification ([link](http://modelnet.cs.princeton.edu/))
- **ShapeNet** - Richly annotated 3D shapes for object detection ([link](https://shapenet.org/))
- **KITTI** - LiDAR-based autonomous driving dataset ([link](http://www.cvlibs.net/datasets/kitti/))
- **ScanNet** - 3D indoor scene dataset for segmentation ([link](https://www.scan-net.org/))

## Tools and Libraries
- **Open3D** - Library for 3D data processing and visualization ([link](http://www.open3d.org/))
- **PCL (Point Cloud Library)** - Extensive toolkit for 3D point cloud manipulation ([link](https://pointclouds.org/))
- **PyTorch3D** - Framework for deep learning on 3D data ([link](https://github.com/facebookresearch/pytorch3d))
- **Minkowski Engine** - Sparse convolution library for efficient 3D processing ([link](https://github.com/StanfordVL/MinkowskiEngine))

## Applications
- **Autonomous Driving**: Essential for object detection and navigation in self-driving cars.
- **Robotics**: Crucial for SLAM and 3D environment mapping.
- **AR/VR**: Enhances immersive experiences and accurate spatial reconstruction.

## Contributing
We welcome contributions to expand and improve this list:
1. Fork this repository.
2. Add relevant resources in the appropriate section.
3. Submit a pull request with a brief description of the added resource.
