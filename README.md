# Color quantization


This projet is just a small projet to use AWS Serverless Application Model (SAM) and doing some image treatment in python.

The application architecture is :
- A s3 bucket to store incoming image
- A lambda to process the image
- A s3 bucket to store the processed image.

![Architecture diagram](assets/architecture.png)

## Image treatment

This app reduce the number of color of an image. It use the K-means clustering method to extract the most used K colors and apply them to the original image (see https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html). The lambda will try different numbers of colors (from 8 to 32), and keeps the K-means clustering with the best silhouette (see https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html). It's not perfect a perfect method, because for a lot of image it keeps only the K-means with the minimum number of cluster. I need to do some more research to understand why.

To speed up the process time, the lambda don't use the full image, but only a sample of 1000 points. With full images, it can take more than one hour to compute all the K-means results and their silhouette. But it's not a big deal because I compute a simplified image at the and. A little loss in quality is acceptable for speed up computation.

## What I learned

I find two limitations with this project.
- First sckitlearn + opencv don't fit in basic python lambda. It's just to big (>250 Mo unzipped). I had to use a docker image. But the image need some external dependencies (see [Dockerfile](Dockerfile)). Maybe using EKS is a better solution.
- Second, it's easy to encounter circular dependencies in SAM. I can't refer my input bucket by its reference in lambda policies but only by its name (see [template.yml](template.yml))

Although SAM has some limitations for defining the architecture it has some advantages. First it's easy to create a lambda triggered after a `s3:CreateObject`, second it's simpler to debug a lambda locally with a `sam local invoke --event event/event.json --env-vars env.json` than uploading the lambda to try it.

Overall, for a small serverless project AWS SAM seems a good solution !