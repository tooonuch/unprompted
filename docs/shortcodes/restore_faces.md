Improves the quality of faces in target image using various models.

Supports `methods` kwarg which takes one or more restoration method names. Defaults to `GPEN`, which is a custom implementation exclusive to Unprompted. It also supports `GFPGAN`, `CodeFormer`, `RestoreFormer` and `RestoreFormerPlusPlus`. Specify multiple methods with the `Config.syntax.delimiter`.

Supports `visibility` kwarg which is an alpha value between 0 and 1 with which to blend the result back into the original face. Defaults to 1.

Supports the `unload` parg which disables the caching features of this shortcode. Caching improves inference speed at the cost of VRAM.

There are several additional parameters that apply only to GPEN:

Supports `resolution_preset` kwarg that determines which GPEN model to use: 512, 1024, or 2048. Defaults to `512`. Please be aware that higher resolutions may lead to an oversharpened look, which is possible to counteract to an extent by lowering `visibility`.

Supports `downscale_method` which is the interpolation method to use when resizing the restored face for pasting back onto the original image. Options include Nearest Neighbor, Bilinear, Area, Cubic and Lanczos. Defaults to `Area`.