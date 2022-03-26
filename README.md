# Doodling Is All You Need

![image](https://user-images.githubusercontent.com/27071473/159839313-d89281d8-0eb5-4b64-a308-7f7e7f7a0d87.png)

Tip-of-the-tongue – when a person fails to retrieve a word from memory – poses a difficulty for image search, such as for online shopping. We propose a workaround to query images from a database by doodling the object of interest.

We do so by constructing a model that represents doodles and real images in the same embedding space, then select real images that are closest to the doodle drawn. We believe our proof-of-concept can complement Google's existing reverse image search that does not take in doodles as input.

We aim to build an image vector search engine, consisting of a database of real-life images, that takes in a doodle sketch and returns the top real-life images most relevant or similar. We study the effect of model architecture (MLP, CNN, ConvNeXt) and learning paradigm (supervised, contrastive learning) on deep learning training for our problem.

# Documentation
- [Dataset Information and Set-up](DATASET.md)
