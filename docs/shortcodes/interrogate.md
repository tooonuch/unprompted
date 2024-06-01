Generates a caption for the given image using one of several techniques.

Supports the `image` kwarg which is the path to the image you wish to interrogate. Defaults to the Stable Diffusion input/output image.

Supports the `method` kwarg which is the interrogation technique to use. Defaults to `CLIP`, which relies on the WebUI's stock methods for completing the task. Other options include `WaifuDiffusion`, `BLIP-2`, and `CLIPxGPT`, all of which will download a large checkpoint upon first use.

Supports the `model` kwarg which is the model to use for the interrogation. For BLIP-2, this can be a Hugging Face string, e.g. `Salesforce/blip2-opt-2.7b`.

Supports the `context` kwarg which is a starting prompt to supply to the model. At the moment, this is only compatible BLIP-2. This can help shape its response.

Supports the `question` kwarg which is the question to ask the model. At the moment, this is only compatible with BLIP-2. This overrides `context`.

Supports the `max_tokens` kwarg which is the maximum number of tokens the model can produce for its response. At the moment, this is only compatible with BLIP-2. Defaults to 50.

The WaifuDiffusion method is oriented around tags as opposed to natural language, and thus supports a few special settings:

- The `confidence_threshold` kwarg, which is a value between 0.0 and 1.0 that indicates how likely the image must contain the tag before being returned as part of the prompt. Defaults to 0.35.
- Supports the `blacklist_tags` kwarg, which is a delimited list of tags to ban.

Note that WaifuDiffusion tags usually contain underscores. You can replace these with spaces as follows: `[replace _from="_" _to=" "][interrogate][/replace]`