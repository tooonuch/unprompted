## How does Control Freak work?

Using the `control_subject` field, the Control Freak template analyzes filenames in your ControlNet image collection for matching terms and automatically sends the best candidate to the ControlNet extension.

Optionally, any unused terms in the selected file can be automatically appended to the prompt.

In my opinion, curating a dataset of ControlNet images is superior to LoRA when it comes to introducing new poses and concepts to Stable Diffusion. I hope that by automating the process with this template, collections of CN images may gain traction on places like Civitai.

## Initial Setup

You can start by building your collection at the following location (PNG files only):

`unprompted/templates/user/presets/control_freak/images`

Feel free to create subfolders to help organize your images - subfolders will not affect the template results.

Additionally, you need to check the ControlNet preset file to ensure you have the required model. The default preset can be found here:

`unprompted/templates/common/presets/controlnet/xl_promax_v1.txt`

Look for the `cn_0_model` and make sure you have that file installed in your `webui/models/controlnet` folder. Most ControlNet models are available on Civitai.

Alternatively, you can set the ControlNet preset to `none` and configure your ControlNet extension by hand.

## Captioning Your Images

Control Freak compares comma-delimited terms to your prompt when deciding which ControlNet image to use. For example, let's say your collection has the following image:

```
robot, backflip, baseball cap.png
```

The image receives a "score" based on the presence of `robot`, `backflip`, and `baseball cap` in your prompt. If it outscores all other images in your collection, it will be picked as the ControlNet image for this generation.

**Terms starting with an underscore are bypassed from the scoring system.** This can be useful if you want to add an author name to the image, or if you have more than one image with identical captions. For example:

```
robot, backflip, baseball cap, _1.png
robot, backlfip, baseball cap, _2.png
```

Compared to captions for training, you need to think about ControlNet captions a little differently. You should only include terms that *always* apply to the image.

For example, an image like, `sitting, looking at viewer, front view, thumbs up, red hair, freckles` might be worth shortening to just `sitting, front view` - the rest of the terms are optional, as you might want to use this CN image in slightly different contexts. But the beauty of this system is that you can create a database that works best for you!

## Control Freak Effects

To squeeze more variety out of your ControlNet images, it is possible to apply filters to the image before inference. By default, the template uses `may_flip_horizontal` which adds a 50% chance to mirror the image.

You can create your own effects at `unprompted/templates/user/presets/control_freak/effects`.

Some ideas for additional effects that I haven't yet implemented:

- `more_precise`: Certain preprocessor types such as softedge can benefit from a simple sharpening pass. A sharper ControlNet image helps Stable Diffusion produce more faithful results.
- `more_creative`: Similar to above, applying a slight blur to the image gives Stable Diffusion more creative leeway.
- `wiggle`: Perhaps the overall position of the ControlNet image can be shifted along the x or y axis for variety?
- `perspective_shift`: Perhaps we can tilt the perspective of the image for added variety?