
# StreamDR - A Multi-level Parallel Framework for Streaming High-dimensional Data Visualization

## Introduction
StreamDR is a novel multi-level parallelization dimensionality reduction method for streaming high-dimensional data visualization, which consists of three modules: new data embedding, embedding function updating and embedding updating. In this approach, the above modules were re-designed to support better parallelization, thus alleviating the latency caused by module waits in the serial setup. Subsequently, a series of new designs and improvements for the three modules were proposed, such as a fast and stable incremental embedding function updating method and an efficient hybrid embedding updating strategy, to achieve fast, high-quality and temporal-stable streaming high-dimensional data visualization.
![framework.png](images%2Fframework.png)

## Environment setup

This project was based on `python 3.7 and pytorch 1.7.0`. See `requirements.txt` for all prerequisites, and you can also install them using the following command.

```bash
pip install -r requirements.txt
```

## Datasets

|               | Instances | Dimensionality | Categories | Data Type |                                                                           Link                                                                           |
|:-------------:|:---------:|:--------------:|:----------:|:---------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     AReM      |  42,239   |       6        |     7      |  tabular  |                  [UCI](https://archive-beta.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem)                   |
|  Basketball   |  52,913   |       6        |     5      |  tabular  |                                          [UCI](https://archive-beta.ics.uci.edu/dataset/587/basketball+dataset)                                          |
|    Shuttle    |  43,500   |       9        |     6      |  tabular  |                                           [UCI](https://archive-beta.ics.uci.edu/dataset/148/statlog+shuttle)                                            |
|      HAR      |  10,299   |      561       |     7      |   text    |                             [UCI](https://archive-beta.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)                             |
|     MNIST     |  60,000   |      784       |     10     |   image   |                                                 [Paper](http://yann.lecun.com/exdb/mnist/)                                                  |

All the datasets are supported with **H5 format** (e.g. HAR.h5), and we need all the dataset to be stored at **`data/data_files`.**

## Simulating Streaming Dataset
To simulate the collected data set as a streaming data set with various data change pattern, try the following command:

```bash
python simulate_streaming_data.py --datasets HAR --change_modes PD
```

## Config File

The configuration files can be found under the folder `./configs`, and we provide two config files with the format `.yaml`. We give the guidance of several key parameters in this paper below.

- **n_neighbors(k):** It determines **the granularity of the local structure** to be maintained in low-dimensional space. A too small value will cause one cluster in the high-dimensional space be projected into two low-dimensional clusters, while too large value will aggravate the problem of clustering overlap. The default setting is **k = 15**.
- **preserve_weight(α):** It determines the **degree of preserving the embedding quality of the existing data** during the updating of the embedding function. The default setting is **α = 10**.
- **candidate_coefficient(β):** It determines **how much range of data will be selected as candidate neighbors** when approximately computing kNN. The default setting is **β = 10**.
- **global_pattern_change_thresh(Γ):** It determines **how many out-of-distribution data are detected to consider a change in the global pattern** of the data. The default setting is **Γ = 50**.
- **initializing_epochs**: It determines the **number of iterations on the pre-existing set when initializing** the embedding function. The default setting is **200**.
- **updating_epochs**: It determines the **number of iterations on the streaming set when updating** the embedding function. The default setting is **50**.


## Training

To launch a streaming data processing with StreamDR, check the configuration file `configs/StreamDR.yaml`, and try the following command:

```bash
python stream_main.py --configs configs/StreamDR.yaml
```

The following is a comparison of the results of different dimensionality reduction methods for streaming high-dimensional data on HAR dataset.

[//]: # (![HAR.jpg]&#40;images%2FHAR.jpg&#41;)
<div align=center><img src="images%2HAR.jpg" width=""></div>
[comment]: <> "## Cite"
