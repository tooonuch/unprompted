Automatically adjusts the width and height parameters in img2img mode based on the proportions of the input image.

Stable Diffusion generates images in sizes divisible by 8 pixels. If your initial image is something like 504x780, this shortcode will set the width and height to 512x768.

Note that this shortcode is architecture-aware, meaning it will return different values depending on if you're using a SD 1.5 or SDXL checkpoint.

Supports the `unit` kwarg which lets you specify the resolution multiplier. Must be divisible by 8. Defaults to 64.

Supports `target_size` which is the minimum possible size of at least one dimension. Defaults to 512 for SD 1.5 and 1024 for SDXL.

Supports `only_full_res` which, if true, will bypass this shortcode unless the "full resolution inpainting" setting is enabled. Defaults to false.

```
[img2img_autosize] Photo of a cat
```