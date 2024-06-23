A port of [the script](https://github.com/ThereforeGames/txt2mask) by the same name, `[txt2mask]` allows you to create a region for inpainting based only on the text content (as opposed to the brush tool.) This shortcode only works in the img2img tab of the A1111 WebUI.

Supports the `method` argument which determines the technology to use for masking:

- `clipseg` ([source](https://github.com/timojl/clipseg)): The default method. It is able to select even small objects with relatively good precision. However, it does not support instance-based masking. Meaning if you try to mask "left dog" in an image with two dogs, it will mask both dogs.
- `panoptic_sam` (source): A powerful method that combines multiple technologies to yield high precision and instance-based masking. However, it requires more VRAM than `clipseg` and sometimes fails to select small objects correctly. For example, attempting to mask a `shirt` may result in the selection of the entire person.
- `tris` (source): Utilizes [Segment Anything](https://segment-anything.com/) technology.
- `fastsam` (source): Utilizes [Segment Anything](https://segment-anything.com/) technology.
- `clip_surgery` (source): Utilizes [Segment Anything](https://segment-anything.com/) technology.

For methods that support instance-based masking (currently just `panoptic_sam`), the following additional kwargs are available:

- `mask_sort_method`: A string that determines sort order for selected masks. Options include `left-to-right`, `top-to-bottom`, `in-to-out`, `big-to-small` and `random`. Defaults to `left-to-right`.
- `reverse_mask_sort`: A boolean that reverses the sort order, e.g. right-to-left instead of left-to-right. Defaults to false.
- `mask_index`: An integer that selects a specific mask from the sorted list. Defaults to 0, which selects the first mask.

Supports the `mode` argument which determines how the text mask will behave alongside a brush mask:
- `add` will overlay the two masks. This is the default value.
- `discard` will ignore the brush mask entirely.
- `subtract` will remove the brush mask region from the text mask region.

Supports the optional `precision` argument which determines the confidence of the mask. Default is 100, max value is 255. Lowering this value means you may select more than you intend.

Supports the optional `padding` argument which increases the radius of your selection by a given number of pixels.

Supports the optional `smoothing` argument which refines the boundaries of the mask, allowing you to create a smoother selection. Default is 20. Try increasing this value if you find that your masks are looking blocky.

Supports the `blur` kwarg which applies a gaussian blur of specified pixel radius to the mask. This is particularly useful outside of img2img mode, which has its own blur setting you can use.

Supports the optional `size_var` argument which will cause the shortcode to calculate the region occupied by your mask selection as a percentage of the total canvas. That value is stored into the variable you specify. For example: `[txt2mask size_var=test]face[/txt2mask]` if "face" takes up 40% of the canvas, then the `test` variable will become 0.4.

Supports the `aspect_var` kwarg which is the name of a variable to store the aspect ratio of the mask. For example, if the mask is 512x768, the variable will become `0.667`.

Supports the optional `negative_mask` argument which will subtract areas from the content mask.

Supports the optional `neg_precision` argument which determines the confidence of the negative mask. Default is 100, the valid range is 1 to 255. Lowering this value means you may select more than you intend.

Supports the optional `neg_padding` which is the same as `padding` but for the negative prompts.

Supports the optional `neg_smoothing` which is the same as `smoothing` but for the negative prompts.

Supports the optional `sketch_color` argument which enables support for "Inpaint Sketch" mode. Using this argument will force "Inpaint Sketch" mode regardless of which img2img tab you are on. The `sketch_color` value can either be a preset color string, e.g. `sketch_color="tan"` ([full list of color strings available here](https://github.com/python-pillow/Pillow/blob/12028c9789c3c6ac15eb147a092bfc463ebbc398/src/PIL/ImageColor.py#L163)) or an RGB tuple, e.g. `sketch_color="127,127,127"`. Currently, txt2mask only supports single-color masks.

Supports the optional `sketch_alpha` argument, which should be paired with `sketch_color`. The `sketch_alpha` value is the level of mask transparency, from 0 (invisible) to 255 (fully opaque.)

Due to a limitation in the A1111 WebUI at the time of writing, the `sketch_alpha` parameter is **not** the same as the "mask transparency" option in the UI. "Mask transparency" is not stored in the `p` object as far as I can tell, so txt2mask implements its own custom solution.

Supports the optional `save` argument which will output the final mask as a PNG image to the given filepath.

Supports the optional `show` positional argument which will append the final mask to your generation output window.

Supports the optional `legacy_weights` positional argument which will utilize the original CLIPseg weights. By default, `[txt2mask]` will use the [refined weights](https://github.com/timojl/clipseg#new-fine-grained-weights).

Supports the `unload_model` argument, which will unload the masking model after processing. On my GTX 3090, this adds about 3 seconds to inference time (using the clipseg model). Defaults to `False`, and should only be enabled on devices with low memory.

The content and `negative_mask` both support the vertical pipe delimiter (`|`) which allows you to specify multiple subjects for masking.

Supports the optional `stamp` kwarg that pastes a temporary PNG onto the init image before running mask processing, useful for redacting a portion of the image for example. The value of `stamp` is the name of a file in `images/stamps` without extension.

Supports the optional `stamp_method` kwarg to choose the sizing and positioning of stamp logic. Valid options are `stretch` and `center`.

Supports the optional `stamp_x` and `stamp_y` kwargs for precise positioning of the stamp. Both default to 0.

Supports the optional `stamp_blur` parg which is the pixel radius of the stamp's gaussian blur filter. Defaults to 0, which disables the filter altogether.

```
[txt2mask]head and shoulders[/txt2mask]Walter White
```