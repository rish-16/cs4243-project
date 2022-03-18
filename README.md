# cs4243-project
Fast Image Vector Search Tool built in PyTorch

# Dataset Download
- CIFAR10
    - ![](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer4.png)
    - 60K 32x32 RGB images, 6K per class for 10 classes. 50K training and 10K test.
    - Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (mutually exclusive)
    - Download - `download-cifar.ipynb`
    - [Website](https://www.cs.toronto.edu/~kriz/cifar.html)
    - [Dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
- QuickDraw
    - ![](https://github.com/googlecreativelab/quickdraw-dataset/blob/master/preview.jpg)
    - 50M doodles across 345 categories.
    - Download - Download and unzip `quickdraw.zip` in `dataset/` folder.
    - [Code](https://github.com/googlecreativelab/quickdraw-dataset)
    - [Dataset (Numpy 28x28 grayscale bitmap .npy)](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap)
- Sketchy
    ![](https://user-images.githubusercontent.com/27071473/158042446-208bd207-c2d4-4b0c-a557-dc11d0340c70.png)
    - 75K sketches of 12K objects from 125 categories.
    - [Paper](https://sketchy.eye.gatech.edu/paper.pdf)
    - [Website](https://sketchy.eye.gatech.edu/)
    - [Code](https://github.com/CDOTAD/SketchyDatabase)
    - [Dataset (Sketches and Photos)](https://tinyurl.com/v2dj69y9)
    - [Dataset (Annotation and Info)](https://tinyurl.com/yxv6s8dv)
    - [Supplementary Report](https://sketchy.eye.gatech.edu/supp.pdf)
- TUBerlin
    - ![](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/teaser_siggraph.jpg)
    - [Paper](http://cybertron.cg.tu-berlin.de/eitz/pdf/2012_siggraph_classifysketch.pdf)
    - [Website](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)
    - [Dataset (Sketches in SVG)](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip)
    - [Dataset (Sketches in png)](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip)
