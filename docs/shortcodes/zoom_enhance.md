Upscales a selected portion of an image via `[img2img]` and `[txt2mask]`, then pastes it seamlessly back onto the original.

Greatly improves low-resolution details like faces and hands. It is significantly faster than Hires Fix and more flexible than the "Restore Faces" option.

Supports the `_alt` parg which engages alternate processing. May help those who are having trouble with the shortcode, but is perhaps not fully compatible with ControlNet.

Supports the `mask` keyword argument which is a region to search for within your image. Defaults to `face`. Note that if multiple non-contiguous regions are found, they will be processed independently.

Supports the `replacement` keyword argument which is the prompt that will be used on the `mask` region via `[img2img]`. Defaults to `face`. If you're generating a specific character--say Walter White--you'll want to set `replacement` to a more specific value, like `walter white face`.

Supports the `negative_replacement` keyword argument, which is the negative prompt that will be used on the mask region via `[img2img]`. Defaults to an empty string.

Both `replacement` and `negative_replacement` support multiple, delimited search terms via `Unprompted.config.syntax.delimiter`.

Supports the `inherit_negative` parg, which copies your main negative prompt to the `[img2img]` replacement task. If used in conjunction with `negative_replacement`, the replacement negative becomes: `main_negative_prompt negative_replacement_value`. If you have multiple, delimited `negative_replacement` values, your main negative prompt will be preprended to all of them.

Supports the `background_mode` parg which will invert the class mask and disable the zoom_enhance step. In other words, you can use this when you want to replace the background instead of the subject. When using this mode, you will likely need to increase `mask_precision` to ~150 or so.

Supports `mask_sort_method` which is used when multiple, non-contiguous masks are detected. Defaults to `left-to-right`. Options include: `left-to-right`, `right-to-left`, `top-to-bottom`, `bottom-to-top`, `big-to-small`, `small-to-big`, `unsorted`.

Supports the `mode` keyword argument, which determines how the shortcode will interact with a pre-existing image mask. Defaults to `subtract`, which will remove your masked pixels from the shortcode's calculations. Options include: `add`, `subtract`, `discard`.

Supports the `bypass_adaptive_hires` positional argument. By default, the shortcode will scale up some inference settings such as CFG scale and sharpness depending on the resolution of the init image. Include this argument to disable that behavior.

Supports the `hires_size_max` keyword argument which is a hard limit on the size of the upscaled image, in order to avoid OOM errors. Defaults to 1024.

Supports the `blur_size` keyword argument, which corresponds to the radius of the gaussian blur that will be applied to the mask of the upscaled image - this helps it blend seamlessly back into your original image. Defaults to `0.03`. Note: this is a float that is a percentage of the total canvas size; 0.03 means 3% of the total canvas.

Supports the `sharpen_amount` argument, which is a float that determines the strength of the unsharp filter that is applied in post-processing.

Supports the `denoising_max` keyword argument. The `[zoom_enhance]` shortcode is equipped with **dynamic denoising strength** based on a simple idea: the smaller the mask region, the higher denoise we should apply. This argument lets you set the upper limit of that feature.

Supports the `mask_size_max` keyword argument. Defaults to `0.5`. If a mask region is determined to be greater than this value, it will not be processed by `[zoom_enhance]`. The reason is that large objects generally do not benefit from upscaling.

Supports the `min_area` keyword argument. Defaults to `50`. If the pixel area of a mask is smaller than this, it may be a false-positive mask selection or at least not worth upscaling.

Supports the `contour_padding` keyword argument. This is the radius in pixels to extend the mask region by. Defaults to `0`.

Supports the `upscale_width` and `upscale_height` arguments. This is the resolution to use with `[img2img]` and should usually match the native resolution of your Stable Diffusion model. Defaults to `512` unless an SDXL model is loaded, in which case it defaults to `1024`.

Supports the `include_original` positional argument. This will append the original, "non-zoom-enhanced" image to your output window. Useful for before-after comparisons.

Supports the `upscale_method` and `downscale_method` arguments which determine the algorithms for image rescaling. Upscale defaults to `Nearest Neighbor`. Downscale defaults to `Lanczos`. Options include: `Nearest Neighbor`, `Box`, `Bilinear`, `Hamming`, `Bicubic`, `Lanczos`.

Supports the `color_correction_method` argument which will attempt to match the color grading of the upscaled image to the original. Defaults to `none`. Options include: `none`,`mvgd`,`mkl`,`hm-mvgd-hm`,`hm-mkl-hm`.

Supports the `color_correct_strength` argument which is an integer that determines how many times to run the color correction algorithm. Defaults to 1.

Supports the `color_correct_timing` argument which determines when to run the color correction algorithm. Defaults to `pre`, which will run color correction before upscaling. Options include `pre` and `post`.

Supports the `controlnet_preset` kwarg which is the name of a file in `templates/presets/controlnet` containing instructions for loading one more ControlNet units.

Supports the experimental `use_starting_face` parg which will upscale the initial image's face as opposed to the resulting img2img's face. (Irrelevant in txt2img mode.)

Supports the `debug` positional argument, which will output a series of images to your WebUI folder over the course of processing.

Supports the `no_sync` parg which will prevent synchronization between your user variables and Stable Diffusion's `p` object at runtime. This may improve compatibility with other shortcodes or extensions.

```
[after][zoom_enhance][/after]
```