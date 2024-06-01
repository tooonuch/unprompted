Processes the content with a given GPT-2 model. This is similar to the "Magic Prompts" feature of Dynamic Prompts, if you're familiar with that.

This shortcode requires the "transformers" package which is included with the WebUI by default, but you may need to install the package manually if you're using Unprompted as a standalone program.

You can leave the content blank for a completely randomized prompt.

Supports the `model` kwarg which can accept a pretrained model identifier from the HuggingFace hub. Defaults to `LykosAI/GPT-Prompt-Expansion-Fooocus-v2`. The first time you use a new model, it will be downloaded to the `unprompted/models/gpt` folder.

Please see the Wizard UI for a list of suggested models.

Supports the `task` kwarg which determines behavior of the transformers pipeline module. Defaults to `text-generation`. You can set this to `summarization` if you want to shorten your prompts a la Midjourney.

Supports the `instruction` kwarg which is a string to be prepended to the prompt. This text will be excluded from the final result. Example: `[gpt instruction="Generate a list of animals"]cat,[/gpt]` may return `cat, dog, bird, horse, cow`.

Supports the `max_length` kwarg which is the maximum number of words to be returned by the shortcode. Defaults to 50.

Supports the `min_length` kwarg which is the minimum number of words to be returned by the shortcode. Defaults to 1.

Supports the `prefix` and `affix` kwargs to include custom strings in the returned result.

Supports the `tokenizer` kwarg to load a separate model as the tokenizer.

Supports the `transformers_class` to specify the methods of inference, defaults to `auto`. Also supports `t5`.

Supports the `unload` parg to prevent keeping the model and tokenizer in memory between runs.