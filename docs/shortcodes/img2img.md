Used within the `[after]` block to append an img2img task to your generation.

The image resulting from your main prompt (e.g. the txt2img result) will be used as the initial image for `[img2img]`.

While this shortcode does not take any arguments, most img2img settings can be set in advance. **Does not currently support batch_size or batch_count** - coming soon!

```
Photo of a cat
[after]
	[sets prompt="Photo of a dog" denoising_strength=0.75]
	[img2img]
[/after]
```