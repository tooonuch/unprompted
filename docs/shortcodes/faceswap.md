Swaps the face in target image using an arbitrary image.

The first parg is a filepath to the face image you wish to swap in.

Supports the optional `body` kwarg which is an image path to perform the swap on. Defaults to the Stable Diffusion output image. Note that this value will essentially override your SD output, so when using this you should minimize your inference steps (i.e. lower it to 1) for faster execution.

Supports the `pipeline` kwarg which is the faceswap method to use. Options include `insightface`, `ghost`, and `face_fusion`. Defaults to insightface, which results in the best quality. You can chain multiple pipelines together with `Config.syntax.delimiter`.

Supports the special `unload` pipeline to free all components from the cache on demand.

The `insightface` pipeline is currently the most developed option as it supports several unique features:

- It can process multiple face images (e.g. `[faceswap "C:/pictures/face1.png|C:/pictures/face2.png"]` using `Config.syntax.delimiter` as a separator.)
- It performs facial similarity analysis to swap the new face onto the best candidate in a picture containing more than one person.
- It supports the `minimum_similarity` kwarg to bypass the faceswap if no one in the target picture bears resemblance to the new face. This kwarg takes a float value, although I haven't determined the upper and lower boundaries yet. A greater value means "more similar" and the range appears to be something like -10 to 300.
- It supports the `export_embedding` parg which takes the average of all input faces and exports it to a safetensors embedding file. This file represents a composite face that can be used in lieu of individual images.
- It supports the `embedding_path` kwarg which is the path to use in conjunction with `export_embedding`. Defaults to `unprompted/user/faces/blended_faces.safetensors`.
- It supports the `gender_bonus` kwarg to boost facial similarity score when source and target genders are equal.
- It supports the `age_influence` kwarg to penalize facial similarity score based on the difference of ages between source and target faces.

Supports the `visibility` kwarg which is the alpha value with which to blend the result back into the original image. Defaults to 1.0.

Supports the `unload` kwarg which allows you to free some or all of the faceswap components after inference. Useful for low memory devices, but will increase inference time. You can pass the following as a delimited string with `Config.syntax.delimiter`: `model`, `face`, `all`.

Supports the `prefer_gpu` kwarg to run on the video card whenever possible.

It is recommended to follow this shortcode with `[restore_faces]` in order to improve the resolution of the swapped result. Or, use the included Facelift template as an all-in-one solution.

Additional pipelines may be supported in the future. Attempts were made to implement support for SimSwap, however this proved challenging due to multiple dependency conflicts.