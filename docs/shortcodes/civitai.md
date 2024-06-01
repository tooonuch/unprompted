Downloads a file using the [Civitai API](https://github.com/civitai/civitai/wiki/REST-API-Reference#get-apiv1models), adding the result to your prompt with the correct syntax. If the specified file is already on your filesystem, this shortcode will not send a request to Civitai.

All of your kwargs are sent as URL parameters to the API (with the exception of system kwargs beginning with `_`) so please review the documentation linked above for a complete list of valid parameters. For example, `[civitai query="something" user="someuser"]` will pass `query` and `user` to the API.

Supports shorthand syntax with pargs, where the first parg is `types` (e.g. LORA or TextualInversion), the second parg is `query` (model name search terms), the third parg is `_weight` (optional, defaults to 1.0), and the fourth parg (also optional) is the `_file`. For example: `[civitai lora EasyNegative 0.5]`.

The `query` value is used as the filename to look for on your filesystem. You can typically search Civitai for a direct model filename (e.g. `query="kkw-new-neg-v1.4"` will return the 'New Negative' model). However, if this isn't working for whatever reason, you can override the filesystem search with the `_file` kwarg: `[civitai query="New Negative" _file="kkw-new-neg-v1.4"]`.

This shortcode will auto-correct the case-sensitivity of `types` to the API's expected format. The API is a bit inconsistent in this regard (e.g. lora = `LORA`, controlnet = `Controlnet`, aestheticgradient = `AestheticGradient`...) but Unprompted will handle it for you. Here are the other edge cases that Unprompted will catch:

- If types is set to `lora`, it will search for both `LORA` and `LoCon` models
- Converts `SD` to `Checkpoint`
- Converts `Embedding` and `TI` to `TextualInversion`
- Converts `Pose` and `OpenPose` to `Poses`
- Converts `CN` to `Controlnet`

Supports the `_debug` parg to print diagnostic information to the console.

Supports the `_api` kwarg which is the URL of the API to communicate with. Defaults to `https://civitai.com/api/v1/models`.

Supports the `_timeout` kwarg to cap the wait time on the API request in seconds. Defaults to 60.

Supports the `_id` kwarg to query the API with a specific modelId, eliminating the need for other parameters.

If a file has multiple versions, you can specify the `_mvid` kwarg instead of `_id` to select a specific version.

Supports the `_words` parg which will retrieve the trigger words from Civitai and include them in the prompt. This will also write the words into a companion JSON file as `activation text` for future use.

```
[civitai lora "HD Helper" 0.5 "hd_helper_v1"]
```