# Research Paper Assignment

# FaceNet: A Unified Embedding for Face Recognition and Clustering

**Abstract**

Despite significant recent advances in the field of face recognition [10, 14, 15, 17], implementing face verification and recognition efficiently at scale presents serious challenges to current approaches. In this paper we present a system, called FaceNet, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors. Our method uses a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches. To train, we use triplets of roughly aligned matching / non-matching face patches generated using a novel online triplet mining method. The benefit of our approach is much greater representational efficiency: we achieve state-of-the-art face recognition performance using only 128-bytes per face. On the widely used Labeled Faces in the Wild (LFW) dataset, our system achieves a new record accuracy of 99.63%. On YouTube Faces DB it achieves 95.12%. Our system cuts the error rate in comparison to the best published result [15] by 30% on both datasets. We also introduce the concept of harmonic embeddings, and a harmonic triplet loss, which describe different versions of face embeddings (produced by different networks) that are compatible to each other and allow for direct comparison between each other.

**Paper** - https://arxiv.org/abs/1503.03832

**Dataset** - http://vis-www.cs.umass.edu/lfw/

# Resources

**Refer notebook for NN4_small2 model** - https://github.com/TessFerrandez/research-papers/blob/prod/facenet/FaceNet.ipynb

**Indepth understanding** - http://krasserm.github.io/2018/02/07/deep-face-recognition/

**Code for normal ZFNet** - https://programmersought.com/article/41772863484/

**David Sandberg's FaceNet** - https://github.com/davidsandberg/facenet
