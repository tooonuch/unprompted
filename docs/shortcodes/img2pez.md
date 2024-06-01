Performs an advanced CLIP interrogation technique on the initial image known as [Hard Prompts Made Easy](https://github.com/YuxinWenRick/hard-prompts-made-easy).

Be aware that this technique is essentially a training routine and will significantly lengthen your inference time, at least on the default settings. On a Geforce 3090, it appears to take around 1-2 minutes.

By default, this shortcode is only compatible with SD 1.5 models. If you wish to use it with SD 2.1 or Midjourney, please set `clip_model` to `ViT-H-14` and `clip_pretrain` to `laion2b_s32b_b79k`. It does work surprisingly well with Midjourney.

Supports the optional `image_path` argument. This is a path to file(s) or a directory to use as the initial image. If not provided, it will default to the initial image in your img2img tab. Note: you can supply multiple paths delimited by `Unprompted.Config.syntax.delimiter`, and img2pez will optimize a single prompt across all provided images. To use this with an entire directory, provide a wildcard as follows: `c:/my/images/*.png`

Supports the optional `target_prompt` kwarg. This is a prompt to optimize with img2pez that can be used on its own or in addition to `image_path`.

Supports the optional `prompt_length` argument, which is the length of the resulting prompt in tokens. Default to 16.

Supports the optional `iterations` argument, which is the total number of training steps to perform. Defaults to 200.

Supports the optional `learning_rate` argument. Defaults to 0.1.

Supports the optional `weight_decay` argument. Defaults to 0.1.

Supports the amusingly-named `prompt_bs` argument, which is described by the technique's authors as "number of intializations." Defaults to 1.

Supports the optional `clip_model` argument. Defaults to ViT-L-14.

Supports the optional `pretrain_clip` argument. Defaults to openai.

Supports the optional `clip_preset` kwarg. This provides an easy way to select a compatible pair of `clip_model` and `pretrain_clip` settings.

Supports the optional `free_memory` argument which attempts to free the CLIP model from memory as soon as the img2pez operation is finished. This isn't recommended unless you are running into OOM issues.