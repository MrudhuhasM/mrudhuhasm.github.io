---
title: "Image Similarity Search with ResNet and Nearest Neighbors"
date: 2024-06-01
tags: [computer-vision, image-similarity, resnet, nearest-neighbors]
description: "Building an image similarity search system using ResNet50 feature extraction and k-nearest neighbors"
math: true
---
dg-permalink: image-similarity-resnet-nearest-neighbors
description: Building an image similarity search system using ResNet50 feature extraction and k-nearest neighbors
---

# Image Similarity Search with ResNet and Nearest Neighbors

Content-based image retrieval finds visually similar images without textual metadata. Traditional approaches using handcrafted features (SIFT, SURF, color histograms) require domain expertise and struggle with semantic similarity. Deep learning feature extraction from pre-trained CNNs captures high-level visual patterns enabling semantic similarity search.

This system uses ResNet50 pre-trained on ImageNet to extract 100,352-dimensional feature vectors, reduced through PCA and visualized with t-SNE. Nearest neighbor search retrieves similar images from 5,000 samples of the Flickr30k dataset.

**Kaggle Notebook:** [Finding Similar Images using ResNet + Nearest Neighbors](https://www.kaggle.com/code/mrudhuhas/finding-similar-images-using-resnet-nneighbours)

## Architecture

**Feature Extraction Pipeline:**
1. Load image, resize to 224×224 (ResNet input)
2. Extract features from final convolutional layer (7×7×2048 = 100,352D)
3. Flatten and L2-normalize feature vector
4. Index in k-NN structure for similarity search

**Retrieval:**
1. Query image → feature extraction
2. k-NN search in feature space (Euclidean distance)
3. Return k most similar images

## Feature Extraction

### ResNet50 Pre-trained Model

ResNet50 trained on ImageNet learns hierarchical visual representations. Early layers detect edges and textures; deeper layers recognize objects and scenes. Using the final convolutional layer (before classification) provides generic visual features transferable to new domains.

```python
model = ResNet50(
    include_top=False,  # Remove classification head
    input_shape=(224,224,3),
    weights='imagenet'
)
```

Setting `include_top=False` removes the 1000-class classifier, retaining convolutional layers that output spatial features.

### Normalization

```python
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_processed = preprocess_input(img_batch)
    features = model.predict(img_processed)
    features_flatten = features.flatten()  # 100,352D vector
    normalized_features = features_flatten / norm(features_flatten)
    return normalized_features
```

L2 normalization ensures unit-length vectors, making Euclidean distance equivalent to cosine similarity:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{2(1 - \mathbf{x} \cdot \mathbf{y})}$$

for normalized vectors $\|\mathbf{x}\| = \|\mathbf{y}\| = 1$.

### Processing Dataset

```python
root_dir = '../input/flickr-image-dataset/flickr30k_images/'
file_names = get_files(root_dir)  # 5,000 images
feature_list = []
for file_name in tqdm(file_names):
    feature_list.append(feature_extraction(file_name, model))
```

Extracting features for 5,000 images takes ~15 minutes on GPU. For larger datasets, batch processing and offline indexing would be necessary.

## Nearest Neighbor Search

### k-NN Index

```python
neighbors = NearestNeighbors(
    n_neighbors=5,
    algorithm='brute',
    metric='euclidean'
).fit(feature_list)
```

**Parameters:**
- `n_neighbors=5`: Return 5 most similar images
- `algorithm='brute'`: Exhaustive search (guaranteed exact results)
- `metric='euclidean'`: Distance metric

For production systems with millions of images, approximate methods (LSH, HNSW, FAISS) trade accuracy for speed. Brute-force search is $O(n)$ per query; approximate methods achieve $O(\log n)$ or $O(1)$ with indexing.

### Query and Retrieval

```python
distances, indices = neighbors.kneighbors([feature_list[query_idx]])
```

Returns indices of k nearest neighbors and their distances. The nearest neighbor (index 0) is always the query image itself with distance ≈0.

### Results

**Query Image:** Person standing on rocky terrain

**Similar Images:**
1. Original (distance: 0.00)
2. Mountain landscape (distance: 0.42)
3. Rock formation (distance: 0.48)
4. Outdoor scene (distance: 0.51)
5. Person outdoors (distance: 0.53)

The system retrieves images with similar visual characteristics: outdoor settings, natural landscapes, rocky textures. Semantic similarity emerges from deep features without explicit object detection.

**Query Image:** Group of people indoors

**Similar Images:** Other indoor group scenes, similar compositions, comparable lighting conditions.

**Query Image:** Urban street scene

**Similar Images:** Cityscapes, buildings, street-level photography.

ResNet features capture scene type, composition, and object categories, enabling semantic retrieval beyond low-level visual similarity.

## Dimensionality Reduction and Visualization

### PCA Compression

100,352 dimensions are computationally expensive and contain redundancy. PCA projects to 100 dimensions retaining most variance:

```python
pca = PCA(n_components=100)
feature_list_compressed = pca.transform(feature_list[:300])
```

Reduced dimensionality accelerates nearest neighbor search and enables visualization.

### t-SNE Visualization

t-SNE maps high-dimensional features to 2D while preserving local neighborhood structure:

```python
tsne_results = TSNE(
    n_components=2,
    metric='euclidean'
).fit_transform(feature_list_compressed)
```

**Scatter Plot:**

Points represent images in 2D embedding space. Proximity indicates visual similarity. Clusters emerge for scene categories (indoor/outdoor), object types (people/landscapes), composition patterns.

**Image Grid:**

```python
def tsne_to_grid_plotter(x, y, image_paths):
    # Map t-SNE coordinates to 2D grid
    # Place image thumbnails at grid positions
    plot_images_in_2d(x_grid, y_grid, image_paths)
```

Arranging actual images on the t-SNE grid reveals semantic structure. Similar images cluster spatially: beach scenes together, urban scenes together, portraits together.

t-SNE visualization confirms that ResNet features encode semantic relationships. Images cluster by content despite no explicit supervision on the Flickr30k dataset.

## Applications

**Reverse Image Search:** User uploads image, system finds visually similar images in database. E-commerce applications: find similar products.

**Image Deduplication:** Detect near-duplicate images by thresholding distance. Useful for cleaning datasets or detecting copyright violations.

**Image Organization:** Cluster images by visual similarity for automatic album creation or content moderation.

**Recommendation Systems:** "Users who viewed this image also viewed..." based on feature similarity.

**Retrieval-Augmented Generation:** Image features as context for multimodal LLMs. Similar to text RAG, retrieve relevant images for visual question answering or captioning.

## Limitations

**Computational Cost:** ResNet50 inference requires GPU for real-time performance. 100K-dimensional features are memory-intensive at scale.

**Domain Shift:** ImageNet pre-training may not transfer well to specialized domains (medical imaging, satellite imagery). Fine-tuning on domain data would improve features.

**Lack of Semantic Understanding:** Features capture visual patterns but not high-level semantics. "Dog playing fetch" and "person throwing frisbee" may not be similar despite shared concept of throwing.

**No Text Integration:** Cannot search "sunset on beach" without combining with captioning models or CLIP-style vision-language embeddings.

## Improvements

**Better Embeddings:** Use models explicitly trained for similarity (SimCLR, MoCo) or multimodal models (CLIP) enabling text-to-image search.

**Approximate NN:** Replace brute-force search with FAISS, Annoy, or HNSW for sub-linear query time on large datasets.

**Fine-tuning:** Adapt ResNet50 on target domain with triplet loss or contrastive learning to improve domain-specific similarity.

**Hybrid Retrieval:** Combine visual features with metadata (tags, captions, location) for more comprehensive search.

**Feature Compression:** Use learned compression (autoencoders) instead of PCA to reduce dimensionality while preserving discriminative information.

## Conclusion

Image similarity search using ResNet50 features and k-NN retrieval achieves semantic matching on Flickr30k. The system finds visually and semantically similar images (landscapes with landscapes, portraits with portraits) without explicit labels.

Pre-trained CNN features transfer well to similarity tasks despite being trained for classification. The final convolutional layer representations capture high-level visual patterns generalizing across domains.

t-SNE visualization reveals semantic clusters in feature space, confirming that deep features encode meaningful relationships. Images with similar content, composition, and scene types cluster together.

For production systems, approximate nearest neighbor search and more efficient embeddings (CLIP) would enable scalable image retrieval. The core principle—deep feature extraction + similarity search—remains effective across domains from e-commerce to content moderation.

**Full implementation:** [Kaggle Notebook](https://www.kaggle.com/code/mrudhuhas/finding-similar-images-using-resnet-nneighbours)
