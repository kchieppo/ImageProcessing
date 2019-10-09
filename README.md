# ImageProcessing
Playing with OpenCV and image processing ideas for educational purposes. Manually implementing most algorithms even though appropriate
functions exist in OpenCV, again for educational purposes.

See "canny.jpg" in the root directory for an example result of my work. This image shows the stages of the Canny edge detection
algorithm. The top-left square is the original image. The next image (to the right) blurs the original image using a Gaussian filter.
The next image is the result of an X and Y Sobel filter applied to the blurred image. This shows the edges. The next image (bottom-left)
is the result of non-maximal suppression. Basically, redundant bits that define edges are removed. The next image shows the strong, weak,
and non-edges from double thresholding. The white edges are strong, and the darker edges are weak. The final image is the result of
edge tracking through hysteresis - keep all strong edges and only weak edges that are touching strong edges.