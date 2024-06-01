Enhances a given image using one or more of the upscaler methods available in the A1111 WebUI.

Supports the `image` kwarg which is the path to the image you wish to enhance. Defaults to the Stable Diffusion output image.

Supports the `models` kwarg which is a delimited list of upscaler model names to use.

Supports the `scale` kwarg which is the scale factor to use. Defaults to 1.

Supports the `visibility` kwarg which is the alpha value to use when blending the result back into the original image. Defaults to 1.

Supports the `keep_res` parg which will maintain the original resolution of the image.