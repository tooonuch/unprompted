Returns various types of metadata about the image, including quality assessment via the pyiqa toolbox.

Supports the `file` kwarg which is the path to the image file. It can also take a PIL Image object. If not specified, this shortcode will analyze the current SD image.

Supports the `width` parg for retrieving the width of the image in pixels.

Supports the `height` parg for retrieving the height of the image in pixels.

Supports the `aspect_ratio` parg for retrieving the aspect ratio of the image (`width` / `height`).

Supports the `filename` parg which is the base name of the image file.

Supports the `filetype` parg which is the file extension of the image file.

Supports the `filesize` parg which is the size of the image file in bytes.

Supports the `iqa` kwarg which is an image quality assessment metric to process the image with. Please refer to the [pyiqa docs](https://github.com/chaofengc/IQA-PyTorch) for a list of supported metrics. I like `laion_aes` for calculating an aesthetic score.

Supports the `pixel_count` parg which is the total number of pixels in the image.

Supports the `unload_metrics` parg which will unload the pyiqa metrics from memory after the shortcode is processed.

```
[image_info file="a:/inbox/somanypixels.png" pixel_count]
```