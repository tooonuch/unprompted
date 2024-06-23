Performs a wide variety of image operations.

You may perform multiple operations on the same image in one call. **Arguments are processed in order, first pargs and then kwargs.**

Each operation must be unique. For example, you cannot run `paste` twice inside of the same block.

Any parg may be specified as a kwarg instead. For example, `[image_edit autotone]` and `[image_edit autotone=1]` are both valid, and the latter allows you to delay the autotone operation if other kwargs precede it. (As stated above, all pargs are processed before kwargs.)

## General setting arguments

### input

A path to an image on your system, or a PIL Image object.

If `input` is not specified, then `[image_edit]` will modify the WebUI's output image by default.

All operations in the `[image_edit]` block will affect the same `input`.

```
input="C:/some/path/to/picture.png"
```

### return

A parg that causes `[image_edit]` to return the final, modified PIL image object.

Useful when you want to save the new image as a variable for later use.

```
[set new_image][image_edit width=500 height=500 return][/set]
```

### copy

A parg that causes `[image_edit]` to duplicate the `input` image for its operations, preventing changes to the original.

I recommend using `copy` when also using `return`.

```
[set new_image][image_edit width=500 height=500 return copy][/set]
```

## Image operation arguments

### add_noise

Adds noise to the `input` image. The value of this kwarg indicates the strength of the noise.

- `noise_monochromatic`: parg that controls whether the noise is monochromatic or not. Defaults to true.
- `noise_type`: string kwarg that determines the type of noise to add. Options include `gaussian`, `salt_pepper`, `poisson`, and `speckle`. Defaults to `gaussian`.
- `max_noise_variance`: integer kwarg that determines the potential variation in the noise. Defaults to 500.

```
[image_edit add_noise=10]
```

### autotone

Adjusts the black point of the `input` image for enhanced contrast. The algorithm produces results that are virtually identical to the **Image > Auto Tone** feature in Photoshop.

```
[image_edit autotone]
```

### brightness

Adjusts the brightness of the `input` image. The value of this kwarg indicates the strength of the adjustment. Powered by the PIL ImageEnhance module.

A value of `1` will not change the image, while `0` will make the image completely black.

```
[image_edit brightness=0.5]
```

### contrast

Adjusts the contrast of the `input` image. The value of this kwarg indicates the strength of the adjustment. Powered by the PIL ImageEnhance module.

A value of `1` will not change the image, while `0` will make the image completely gray.

```
[image_edit contrast=0.5]
```

### sharpness

Adjusts the sharpness of the `input` image. The value of this kwarg indicates the strength of the adjustment. Powered by the PIL ImageEnhance module.

```
[image_edit sharpness=3]
```

### blur

Blurs the `input` image. The value of this kwarg indicates the pixel radius of the blur. Powered by the PIL ImageFilter module.

- `blur_type`: kwarg that determines the type of blur to apply. Options include `gaussian`, `box`, and `unsharpen`. Defaults to `gaussian`.

```
[image_edit blur=5]
```

### intensity

Adjusts the intensity of the `input` image. The value of this kwarg indicates the strength of the adjustment. Powered by the PIL ImageEnhance module.

Note that intensity and saturation are similar but not identical.

```
[image_edit intensity=20]
```

### hue, saturation, value

Modify the `input` image in the HSV color space.

- `shift_relative`: parg that determines whether the values are modified relative to the original image, or set absolutely. Defaults to false.

```
[image_edit hue=20 saturation=20 value=20]
```

### red, green, blue

Modify the `input` image in the RGB color space.

- `shift_relative`: parg that determines whether the values are modified relative to the original image, or set absolutely. Defaults to false.

```
[image_edit red=20 green=20 blue=20]
```

### rotate

Rotates the `input` image by the specified number of degrees.

```
[image_edit rotate=90]
```

### flip_vertical

Flips the `input` image vertically.

```
[image_edit flip_vertical]
```

### flip_horizontal

Flips the `input` image horizontally.

```
[image_edit flip_horizontal]
```

### colorize

Colorizes the `input` image with the specified color.

The value of this kwarg can either be a hex color code (e.g. `#FF0000`) or a tuple of RGB values (e.g. `255, 0, 0`).

```
[image_edit colorize="#FF0000"]
```

### color_match

A path to an image on your system, or a PIL image object.

The `color_match` image will be used to grade the colors of the `input` image.

Powered by the [color-matcher](https://github.com/hahnec/color-matcher) module.

- `color_match_strength`: The opacity of the new grading, from 0 to 1. For example, set to 0.5 to blend halfway between the old and new image. Defaults to 1, or full strength.
- `color_match_method`: Algorithm for color grading, supports `hm`, `mvgd`, `mkl`, `hm-mvgd-hm`, `hm-mkl-hm`. Defaults to `mkl`.

```
[image_edit color_match="C:/path/to/grading.png" color_match_strength=0.5]
```

### height, width

Specify `height` and/or `width` to adjust the dimensions of the `input` image.

Only one of the two arguments is required.

- `unit`: kwarg which is the unit of measurement for the `width` and `height` kwargs. Options include `px` and `%`. Defaults to `px`.
- `resize`: kwarg which is the behavior when resizing the `input` image. Options include `scale` and `crop`. Defaults to `scale`.
- `resample_method`: kwarg which is the method of resampling when using the `scale` technique. Options include `Nearest Neighbor`, `Box`, `Bilinear`, `Hamming`, `Bicubic`, and `Lanczos`. Defaults to `Lanczos`.
- `origin`: kwarg which is the anchor point of the image when using the `crop` technique. Options include `top_left`, `top_center`, `top_right`, `middle_left`, `middle_center`, `middle_right`, `bottom_left`, `bottom_center`, and `bottom_right`. Defaults to `middle_center`.
- `keep_ratio`: parg which will preserve the aspect ratio of the image. Note that if you specify both `width` and `height`, it will override and disable `keep_ratio`.
- `min_width`, `min_height`: kwargs which can be used to set a minimum size for the image. This is applied after the `keep_ratio` parg. If the image is smaller than the minimum, it will be scaled up to the minimum.

```
[image_edit height=1024 width=768 resize="crop"]
```

### mask

A path to an image on your system, or a PIL image object.

The `mask` is used to control the alpha of the `input`. In simpler terms, the mask is used to mask out parts of the image (shocking, right?)

If necessary, the `mask` is resized to match the dimensions of `input` and converted to `L` mode.

- `keep_masked_rgb`: Parg that will preserve the RGB information of the `input` pixels even where the alpha channel is `0` (fully transparent.) In most cases, you probably want to dispose this information.

```
[image_edit mask="C:/some/path/to/mask.png"]
```

### mode

Converts the `image` mode to the specified type, e.g. `RGB`, `RGBA`, `L`.

```
[image_edit mode="L" return copy]
```

### paste

A path to an image on your system, or a PIL image object.

The `paste` image will be pasted onto `input`.

The `paste` image will be automatically converted to `RGBA` mode.

- `paste_x`: The horizontal placement of the `paste` image in pixels, relative to the top-left coordinate of `input` (0,0)
- `paste_y`: The vertical placement of the `paste` image in pixels, relative to the top-left coordinate of `input` (0,0)

```
[image_edit paste="C:/some/path/to/overlay.png" paste_x=20 paste_y=30]
```

### save

A path to save the `input` image to disk.

Since arguments are processed in order, you can run `save` after some operations but before others. This is useful for debugging purposes.

```
[image_edit width=350 height=350 save="C:/image/after_resizing.png" mask="C:/some/mask.png"]
```