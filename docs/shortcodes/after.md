Processes the content after the main task is complete.

This is particularly useful with the A1111 WebUI, as it gives you the ability to queue up additional tasks. For example, you can run img2img after txt2img from the same template.

Supports the optional `after_index` argument which lets you control the order of multiple `[after]` blocks. Defaults to 0. For example, the `[after 2]` block will execute before the `[after 3]` block.

You can `[get after_index]` inside of the `[after]` block, which can be useful when working with arrays and for loops.

Supports the optional `allow_unsafe_scripts` parg which will disable the shortcode's normal behavior of bypassing extensions with known compatibility issues.

Supports the `dupe_index_mode` kwarg which determines how the `[after]` block will handle duplicate indexes:

- `concat` (default): The `[after]` block will be appended to the existing `[after]` block at the specified index.
- `skip`: The `[after]` block will be ignored.
- `append`: The `[after]` block will be added to the next available index.
- `replace`: The existing `[after]` block at the specified index will be overwritten.

```
Photo of a cat
[after]
	[sets prompt="Photo of a dog" denoising_strength=0.75]
	[img2img]
[/after]
```