Gets or sets image from the given `path` as the initial image for use with img2img. Call this without any arguments to get the current image.

Note that `path` must be an absolute path, including the file extension.

If the given `path` ends with the `*` wildcard, `[init_image]` will choose a random file in that directory.

**Important:** At the moment, you still have to select an image in the WebUI before pressing Generate, or this shortcode will throw an error. You can select any image - it doesn't matter what it is, just as long as the field isn't empty.

```
[init_image "C:/pictures/my_image.png"]
```