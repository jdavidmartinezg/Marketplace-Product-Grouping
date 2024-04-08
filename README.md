# Marketplace-Product-Grouping
Simple Python implementation of an unsupervised ML strategy to find and group similar products within an online marketplace

## Problem statement

Within any e-commerce marketplace there are products that are similar or identical to each other (they are products sold by different sellers). How to search for these items to group them and make them comparable to each other? This would improve the experience compared to many similar options.

---

## Executive summary of proposed solution

To find similar products we will use the text from the title of the listing as well as the main image thumbnail from the listing. With the text we can find similar items by turning the text from the listing's title into text embeddings. Once the embeddings are created for each product we can compare all products in space by using any distance metric, such as cosine similarity. On the other hand, we can use the thumbnails from the listings to find similar listings with unsupervside learning. The idea is to extract features from the images with PCA and turn those extracted features into image embedding vectors. Once these vectors are generated for each thumbnail, we can compare items with a distance metric to find similar products.

## Proposed next steps and points of improvement

* Both image and text similarity approaches can be combined into a weighted function to return a consolidated list of potential similar items for any given product. Also a custom algorithm can be implemented to only return products that appear both in the tittle text similarity as well as the image similarity.
* Data can be expanded by passing an app token as header to bypass the 1K product limit retrieval on the public API.
* If there are labeled examples of pair-wise product matches, one can think of using supervised ML to create a computer vision model to detect pairs of similar products in the marketplace. A potential model to accomplish the latter could be a CNN architecture based on Siamese convolutional neural networks.


## Data download

##### data_download.ipynb

This Python script is used to interact with the MercadoLibre's public API to download and process data related to product categories and items.

The script performs the following operations:

1. **List Categories**: It sends a GET request to the MercadoLibre API to fetch all the categories available.

2. **List Subcategories**: For a given category ID, it fetches all the subcategories under it.

3. **Get Total Number of Items in a Category**: For a given category ID, it fetches the total number of items available in that category.

4. **Download Category**: This is a function that takes a category ID, limit, and offset as parameters. It sends a GET request to the MercadoLibre API to fetch items in the given category, limited by the 'limit' parameter and offset by the 'offset' parameter. For each item, it downloads the thumbnail image and saves it locally. It also creates a DataFrame with the item's title, price, currency ID, and thumbnail ID.

5. **Download Items in a Category**: For a given category ID, it calculates the offsets needed to download all items (in this case, 1000 items as it's the maximum number of items we can retrieve with the public API). It then calls the `download_category` function for each offset and concatenates the returned DataFrames. Finally, it saves the DataFrame to a CSV file.

6. **Display DataFrame**: It displays the final DataFrame created after downloading all items in the category.

---

## Exploratory analysis and embeddings creation

#### eda.ipynb

This Python script is used for exploratory data analysis (EDA) on a dataset of items obtained from a CSV file. The script performs the following operations:

1. **Load Libraries**: It imports the pandas library, which is used for data manipulation and analysis.

2. **Load Data**: It reads a CSV file named 'items.csv' into a pandas DataFrame.

3. **Display Data**: It displays the DataFrame loaded from the CSV file.

4. **Convert Title to Lowercase**: It converts the 'title' column of the DataFrame to lowercase. This is a common preprocessing step in text analysis to ensure consistency.

5. **Install Sentence Transformers**: It installs the sentence-transformers library, which is used for creating sentence embeddings.

6. **Create Text Embeddings**: It uses the SentenceTransformer model 'paraphrase-multilingual-mpnet-base-v2' to create embeddings for the 'title' column of the DataFrame. The embeddings are created by applying the `model.encode` function to each title. The result of encoding the first title is displayed.

7. **Create DataFrame for Embeddings**: It creates a new DataFrame where each row corresponds to an item and the columns are the embeddings. The index of the new DataFrame is the same as the index of the original DataFrame.

8. **Save Embeddings to CSV**: It saves the embeddings DataFrame to a CSV file named 'title_embeddings.csv'.

9. **Display Embeddings**: It displays the embeddings DataFrame.

This script is useful for preparing the data for machine learning tasks, particularly those involving natural language processing (NLP), where sentence embeddings can be used as input features for models.

---

## Tittle matching to find similar products

#### title_matching.ipynb

This Python script is used for title matching based on the cosine similarity of title embeddings. The script performs the following operations:

1. **Load Libraries**: It imports the pandas library for data manipulation and analysis.

2. **Load Data**: It reads a CSV file named 'items.csv' into a pandas DataFrame. It also reads 'title_embeddings.csv' into another DataFrame.

3. **Display Data**: It displays the first 5 rows of both the 'data' and 'embeddings' DataFrames.

4. **Preprocess Embeddings**: It drops the 'Unnamed: 0' column from the 'embeddings' DataFrame.

5. **Calculate Cosine Similarity**: It calculates the cosine similarity between the embeddings of the titles using the `cosine_similarity` function from the sklearn library. It then creates a DataFrame with the cosine similarities, where the columns and index are the titles from the 'data' DataFrame.

6. **Plot Cosine Similarity Matrix**: It plots the cosine similarity matrix as a heatmap for the first 20 titles using the seaborn library.

7. **Create Similarity Function**: It creates a function named 'get_similar_titles' that takes a title as input and returns the top 5 most similar titles, with their respective row indexes and cosine similarities. The function works by getting the cosine similarities of the input title with all other titles, sorting the titles based on the cosine similarities, and returning the top 5 most similar titles.

8. **Test Similarity Function**: It tests the 'get_similar_titles' function by getting 5 random titles from the 'data' DataFrame and finding the most similar titles to "Michael Jackson - Thriller - Cd". It also displays the rows in the 'data' DataFrame where the title is "Michael Jackson - Thriller - Cd".

This script is useful for tasks involving title matching, such as recommendation systems, where you want to find items that are similar to a given item.

---

## Image matching to find similar products

#### image_matching.ipynb

This Python script is used for image matching based on the Euclidean distance of image features. The script performs the following operations:

1. **Load Libraries**: It imports the necessary libraries, including Clustimage for image clustering, pandas for data manipulation, numpy for numerical operations, and matplotlib for plotting.

2. **Initialize Clustimage**: It initializes a Clustimage object with the method set to 'pca' and the embedding set to 'umap'. 

3. **Import Images**: It imports the images from the "thumbnails" directory and stores the pathnames in the variable `X`.

4. **Cluster Images**: It clusters the images and stores the results in the `results` variable.

5. **Load Data**: It reads a CSV file named 'items.csv' into a pandas DataFrame.

6. **Find Image Index**: It finds the index of a specific image in the `X["pathnames"]` array.

7. **Create DataFrame for Image Features**: It creates a DataFrame with the image features from `cl.results['feat']` and sets the index to the image pathnames.

8. **Calculate Euclidean Distance Matrix**: It calculates the Euclidean distance matrix for the image features using the `euclidean_distances` function from the sklearn library. It then creates a DataFrame with the distance matrix, where the columns and index are the image pathnames.

9. **Create Similarity Function**: It creates a function named 'get_similar_items' that takes an image pathname and a number `n` as input and returns the pathnames of the top `n` images that are most similar to the input image. The function works by getting the Euclidean distances of the input image with all other images, sorting the images based on the distances, and returning the top `n` images.

10. **Test Similarity Function**: It tests the 'get_similar_items' function by finding the most similar images to a specific image.

11. **Create Plotting Function**: It creates a function named 'plot_similar_items' that takes an image pathname and a number `n` as input and plots the input image and the top `n` images that are most similar to the input image.

12. **Test Plotting Function**: It tests the 'plot_similar_items' function by plotting a specific image and its most similar images.

This script is useful for tasks involving image matching, such as recommendation systems, where you want to find images that are similar to a given image.



