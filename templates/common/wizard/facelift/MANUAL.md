### Creating re-usable embeddings

You can speed up your workflow by converting your face images to a safetensors embedding.

Load one or more images into `faces` then select the `make_embedding` preset.

Upon running the template, Facelift will create a small `blended_face.safetensors` file in `unprompted/user/faces`.

You can rename this file and move it to any location of your choosing. The next time you use Facelift, change your preset back to e.g. `fast_v2` and load the safetensors file into `faces` instead of the raw images.

### Always use the same face with the same character

First, check the Unprompted manual for a guide on "Setting up Replacement Terms."

It explains how you can replace a simple string like `tommy wiseau` with relevant LoRAs and other terms for your characters.

We can take this a step further and implement Facelift support for our replacement term!

You simply need to set up the `faces` value for your character like so:

```
"tommy wiseau":"<lora:tommy:1.0>tommy, other terms go here [array faces _append='C:/path/to/face/embeddings.safetensors']"
```

Now whenever you auto-include the Facelift template and use the phrase `tommy wiseau`, it will automatically load the correct face file for you. This can save so much time!

