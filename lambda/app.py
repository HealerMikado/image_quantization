import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
import numpy as np
import logging
import boto3
import uuid
import os

TARGET_BUCKET=os.environ["TARGET_BUCKET"]
MAX_CLUSTERS=int(os.environ["MAX_CLUSTERS"])

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client=boto3.client("s3")

def vectorize_image(img):
    # Convert from BGR to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Need to reshape the image
    # from (N,M,3) (1 matrix for each color)
    # to (N*M,3) (1 vector for each color)
    width, height, d = tuple(img.shape)
    return np.reshape(img, (width * height, d))


def get_best_kmeans(image_array):
    # get best cluster number, silhouette method
    silhouettes = []
    kmeans_clustering = []
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    for k in range(8, MAX_CLUSTERS+1):
        logger.info(f"Proces kmeans for k={k}")
        kmeans_clustering.append(KMeans(n_clusters=k, n_init="auto").fit(image_array_sample))
        labels = kmeans_clustering[-1].labels_
        silhouettes.append(silhouette_score(
            image_array_sample, labels, metric='euclidean'))

    # find best kmeans
    best = silhouettes.index(max((silhouettes)))

    # plt.plot(list(map(lambda kmeans : kmeans.inertia_, kmeans_clustering)),'bx-')
    # plt.show()
    logger.info(f"best kmeans : {len(kmeans_clustering[best].cluster_centers_)} clusters")
    return kmeans_clustering[best]


def recreate_image(centroid, points, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return np.array(centroid[points].reshape(h, w, -1), dtype=np.uint8)


def lambda_handler(event, context):

    logger.info(event)
    

    # get bucket and object key from event object
    source_bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = event["Records"][0]["s3"]["object"]["key"]

    # Generate a temp name, and set location for our original image
    object_key = str(uuid.uuid4()) + '-' + key
    img_download_path = f'/tmp/{key}'


    # Load the image
    with open(img_download_path, "wb") as img_file:
        s3_client.download_fileobj(source_bucket, key, img_file)
    
    img = cv.imread(img_download_path)

    image_array = vectorize_image(img)

    # image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    best_kmeans = get_best_kmeans(image_array)


    labels = best_kmeans.predict(image_array)
    labels = list(labels)

    centroid = best_kmeans.cluster_centers_
    # Compute the % for each centroid for the pie chart
    percent = []
    for i in range(len(centroid)):
        j = labels.count(i)
        j = j/(len(labels))
        percent.append(j)
    print(percent)

    palette_list = list()
    for color in centroid:
        palette_list.append([[tuple(color)]])


    # plt.pie(percent, colors=np.array(centroid/255), labels=np.arange(len(centroid)))
    # plt.show()

    # plt.axis('off')
    # plt.imshow(recreate_image(centroid, labels, img.shape[1], img.shape[0]))
    # plt.show()

    img_out = cv.cvtColor(recreate_image(centroid, labels, img.shape[1], img.shape[0]), cv.COLOR_BGR2RGB)
    
    quantized_img_path = f"/tmp/{key}"
    cv.imwrite(quantized_img_path, img_out)
    upload_key = f'quantized-{key}'
    s3_client.upload_file(quantized_img_path, TARGET_BUCKET,upload_key)



