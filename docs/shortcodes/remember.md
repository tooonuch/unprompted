Allows you to keep one or more variables in memory for the duration of a batch run (i.e. runs where `batch_count` > 1). This overrides Unprompted's default behavior of completely resetting the `shortcode_user_vars` object after each image.

Here is a practical example:

Let's say you have a template where you want to pass different values to `[zoom_enhance]` that correspond to the `batch_index` of the run.

To do this, we can create an `[array]` and append a new value to it each step of the run. We will mark the array with `[remember]` and tell `[zoom_enhance]` to look up the `batch_index` position of the array.


```
[array zoom_replacements _append="{get subject}"]
[if batch_index=0]
	[remember zoom_replacements]
	[after][zoom_enhance replacement="{array zoom_replacements batch_index}" negative_replacement="worst quality"][/after]
[/if]
```
```
[remember var_a var_b]
```