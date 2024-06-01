Resizes an image to the given dimensions, works with the SD image by default.

The first parg is the path to your `image`. It can also take a PIL Image object.

Supports the `save_out` parg which is the path to save the resized image to. If you do not specify this, the new image will overwrite the original.

Supports `width` and `height` kwargs which are the new dimensions of the image.

Supports the `unit` kwarg which is the unit of measurement for the `width` and `height` kwargs. Options include `px` and `%`. Defaults to `px`.

Supports the `technique` kwarg which is the method of resizing. Options include `scale` and `crop`. Defaults to `scale`.

Supports the `resample_method` kwarg which is the method of resampling when using the `scale` technique. Options include `Nearest Neighbor`, `Box`, `Bilinear`, `Hamming`, `Bicubic`, and `Lanczos`. Defaults to `Lanczos`.

Supports the `origin` kwarg which is the anchor point of the image when using the `crop` technique. Options include `top_left`, `top_center`, `top_right`, `middle_left`, `middle_center`, `middle_right`, `bottom_left`, `bottom_center`, and `bottom_right`. Defaults to `middle_center`.

Supports the `keep_ratio` parg which will preserve the aspect ratio of the image. Note that if you specify both `width` and `height`, it will take precedence over `keep_ratio`.

Supports the `min_width` and `min_height` kwargs which can be used to set a minimum size for the image. This is applied after the `keep_ratio` parg. If the image is smaller than the minimum, it will be scaled up to the minimum.

```
[resize "a:/inbox/picture.png" width=350]
```
