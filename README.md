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
- [Datasets](#datasets)
- [Research Papers](#research-papers)
  - [Surveys](#surveys)
  - [Benchmarks](#benchmarks)
  - [Core Methods](#core-methods)
    - [Classification](#classification)
    - [Segmentation](#segmentation)
    - [Object Detection](#object-detection)
    - [Registration and SLAM](#registration-and-slam)
  - [Self-Supervised Learning](#self-supervised-learning)

- [Tools and Libraries](#tools-and-libraries)
- [Applications](#applications)
- [Contributing](#contributing)
- [References](#references)

---

## Introduction

A **point cloud** is a collection of data points in a three-dimensional (3D) space. Each point represents a specific position in 3D, often defined by its **(x, y, z)** coordinates. In some cases, additional attributes, such as color (**RGB** values), intensity, or classification labels, are included, providing more context about the object or scene being represented.

## Datasets

### **Point Cloud Datasets by Domain**

#### **Object Classification and Detection**
- **[ModelNet](http://modelnet.cs.princeton.edu/)**: CAD models for 3D object classification.  
  - **Objects**: 10 categories (ModelNet10) and 40 categories (ModelNet40).  
  - **Format**: .off files representing 3D object meshes.  
  - **[Paper](https://arxiv.org/abs/1403.0623)**  

- **[ShapeNet](https://shapenet.org/)**: Richly annotated 3D shapes for object detection and classification.  
  - **Objects**: 55 categories with over 51,300 unique 3D models.  
  - **Annotations**: Includes semantic labeling and alignments.  
  - **[Paper](https://arxiv.org/abs/1512.03012)**  

---

#### **Autonomous Driving**
- **[KITTI](http://www.cvlibs.net/datasets/kitti/)**: Benchmark dataset for LiDAR-based autonomous driving tasks.  
  - **Scenarios**: Includes road detection, 3D object detection, and tracking.  
  - **Format**: LiDAR scans (.bin) with synchronized camera images.  
  - **[Paper](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf)**  

- **[nuScenes](https://www.nuscenes.org/)**: Multimodal dataset for autonomous driving.  
  - **Scenarios**: 3D bounding boxes, semantic segmentation, and tracking.  
  - **Annotations**: 1.4M LiDAR sweeps with 3D bounding boxes.  
  - **[Paper](https://arxiv.org/abs/1903.11027)**  

- **[Waymo Open Dataset](https://waymo.com/open/)**: Large-scale autonomous driving dataset.  
  - **Scenarios**: High-resolution LiDAR and camera data.  
  - **Annotations**: 12M 3D bounding boxes.  
  - **[Paper](https://arxiv.org/abs/1912.04838)**  

---

#### **Indoor Scene Understanding**
- **[ScanNet](https://www.scan-net.org/)**: Annotated RGB-D dataset for 3D indoor scene segmentation.  
  - **Scenes**: 1,513 scanned rooms with 3D meshes and RGB-D frames.  
  - **Annotations**: Semantic and instance-level labels.  
  - **[Paper](https://arxiv.org/abs/1702.04405)**  

- **[Matterport3D](https://niessner.github.io/Matterport/)**: Dataset for RGB-D and 3D mesh-based indoor scene analysis.  
  - **Scenes**: Over 10K rooms across 90 buildings.  
  - **Applications**: Scene reconstruction, object detection, and segmentation.  
  - **[Paper](https://arxiv.org/abs/1709.06158)**  

- **[SUN RGB-D](https://rgbd.cs.princeton.edu/)**: RGB-D dataset for indoor scene understanding.  
  - **Scenes**: 10,335 RGB-D images annotated with 3D bounding boxes.  
  - **[Paper](https://arxiv.org/abs/1412.0767)**  

---

#### **Aerial and Outdoor Mapping**
- **[Semantic3D](http://www.semantic3d.net/)**: Outdoor LiDAR point clouds for semantic segmentation.  
  - **Scenes**: Over 4 billion points from urban and rural areas.  
  - **Annotations**: 8 semantic classes (e.g., building, vegetation, cars).  
  - **[Paper](https://arxiv.org/abs/1704.03847)**  

- **[ISPRS 3D Semantic Labeling](http://www2.isprs.org/commissions/comm2/wg4/3d-semantic-labeling.html)**: Benchmark dataset for 3D point cloud labeling.  
  - **Scenarios**: Urban outdoor scenes.  
  - **Annotations**: Detailed semantic labels.  
  - **[Paper](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/IV-2-W1/13/2017/)**  

- **[DALES](https://www.cis.rit.edu/dales/)**: Large-scale aerial LiDAR dataset for semantic segmentation.  
  - **Scenes**: 10 urban scenes with high-resolution point clouds.  
  - **Annotations**: 8 semantic classes.  
  - **[Paper](https://arxiv.org/abs/2004.05594)**  

---

#### **Synthetic and Simulated Datasets**
- **[PartNet](https://cs.stanford.edu/~kaichun/partnet/)**: Hierarchically annotated 3D shapes for part segmentation.  
  - **Objects**: 24 object categories with over 573,000 part instances.  
  - **Applications**: Shape segmentation and understanding.  
  - **[Paper](https://arxiv.org/abs/1812.02713)**  

- **[Synthetic LiDAR Dataset (CARLA)](http://carla.org/)**: Simulated autonomous driving data from CARLA simulator.  
  - **Scenarios**: Vehicle, pedestrian, and environment modeling.  
  - **Format**: Simulated LiDAR and camera streams.  
  - **[Paper](https://arxiv.org/abs/1711.03938)**  

---

#### **Miscellaneous Datasets**
- **[Toronto-3D](https://github.com/WeikaiTan/Toronto-3D)**: Aerial point cloud dataset for urban semantic segmentation.  
  - **Scenes**: 2.7M points captured in an urban environment.  
  - **Annotations**: 8 semantic classes.  
  - **[Paper](https://arxiv.org/abs/2002.11397)**  

- **[ArCH](https://archdataset.github.io/)**: Point cloud dataset for architectural heritage.  
  - **Scenes**: High-quality scans of cultural heritage sites.  
  - **Applications**: Reconstruction and preservation analysis.  
  - **[Paper](https://arxiv.org/abs/2006.01578)**  

- **[Indoor RGB-D](http://www.cs.umd.edu/~yongjie/dataset.html)**: A large-scale dataset combining RGB-D frames and 3D point clouds.  
  - **Scenes**: Real-world indoor environments.  
  - **Applications**: Navigation, mapping, and object detection.  
  - **[Paper](https://arxiv.org/abs/1308.1200)**  

---

### **Dataset Summary**

| Dataset Name         | Point Cloud Type       | Primary Application              | Data Volume             |
|-----------------------|------------------------|-----------------------------------|-------------------------|
| ModelNet             | Synthetic CAD models  | Object classification            | 10K+ models            |
| ShapeNet             | Annotated 3D shapes  | Object detection, segmentation   | 51K+ shapes            |
| KITTI                | LiDAR scans          | Autonomous driving               | 6 hours driving data   |
| ScanNet              | RGB-D indoor scenes  | Indoor segmentation              | 1.5K scenes            |
| Semantic3D           | Outdoor LiDAR scans  | Semantic segmentation            | 4B+ points             |
| PartNet              | Annotated shapes     | Part segmentation                | 573K+ part instances   |

---

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
