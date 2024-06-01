Allows you to modify or replace your img2img mask with arbitrary files.

Supports the `mode` argument which determines how the file mask will behave alongside the existing mask:
- `add` will overlay the two masks. This is the default value.
- `discard` will scrap the existing mask entirely.
- `subtract` will remove the file mask region from the existing mask region.

Supports the optional `_show` positional argument which will append the final mask to your generation output window.

```
Walter White[file2mask "C:/pictures/my_mask.png"]
```