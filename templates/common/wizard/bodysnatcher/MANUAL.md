<details open><summary>⚠️ Important info, please read carefully</summary>

To achieve compatibility between Unprompted and ControlNet, you must manually rename the `unprompted` extension folder to `_unprompted`. This is due to [a limitation in the Automatic1111 extension framework](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/8011) whereby priority is determined alphabetically. Restart the WebUI after making this change.

</details>

<details><summary>Usage Guide</summary>

Bodysnatcher's default settings are designed to work well for most users. However, here are some general tips to help you get the best results:

- **Always use this template on the img2img inpaint tab.** If you want to perform inference on the entire image (rather than a masked subject), set `mask_method` to `none` but still use the inpaint tab.
- **Inpainting masks are generated automatically from the `class`.** If the original image contains multiple subjects of the same class, you can manually paint out the areas you *don't* want to process. By default, your manual mask is subtracted from the automatic mask.
- **Masked content mode:** For standard models, set this to `Original`. For inpainting models, set this to `Fill`.
- For inpainting models, I recommend choosing a high value for `Inpainting conditioning mask strength` in the WebUI settings (e.g. between 0.8 and 1). Lower values will result in more of the original image being retained, at the cost of the new subject's likeness. I have found it is better to retain qualities of the original image via ControlNet models.
- **You must download ControlNet models separately before using the ControlNet presets.** Check the `unprompted/templates/common/presets/controlnet` folder for more information on model names. In most cases, you can simply Google the model name to find a working download link.
- **ControlNet presets are NOT cross-compatible between SD 1.5 and SDXL!** At the time of writing, most of the presets are for SD 1.5. SDXL does not have many great ControlNet models yet. The best I've found for use with this template are `controlnetxlCNXL_ecomxlSoftedge` and `controlnetxlCNXL_xinsirCannyv2`.

</details>

<details><summary>Recommended inference settings for SD 1.5</summary>

Here are some guidelines for achieving the best results with this template:

- The new `vivarium_v1` ControlNet body preset enforces its own inference settings, thus many of the tips below do not apply when using this preset!
- Use either `magic_mirror` or `fidelity` ControlNet body preset.
- Use either the `subtle` or `general` inference preset. The former provides the best image quality while the latter provides improved subject likeness.
- Use an inpainting model with strong knowledge of human anatomy, such as EpicPhotoGasm or AbsoluteReality.
- Use the WebUI's Refiner to switch to a non-inpainting model at ~0.8 steps. This will add extra detail to your final image.
- If you have a strong computer, set your inpaint area to `Only masked` mode and increase the resolution by as much as your GPU and checkpoint can handle. On my 3090, I get great results with 768x768.
- You can paint out areas of the image you *don't* want to process and they will be subtracted from the final mask. This is useful for images containing multiple subjects.
- Enable the Facelift template to improve the quality of faces.

</details>

<details><summary>ControlNet Preset Cheatsheet for SD 1.5</summary>

Each of the included ControlNet presets was designed with a specific purpose in mind. 

- (NEW!) `vivarium` (1 unit): This is an experimental preset that utilizes the `instructp2p` model, opinionated inference settings, and some files from Civitai for fast, accurate bodyswaps. It is the only preset that does not necessarily benefit from using an inpainting model or `[txt2mask]`.
- `fidelity` (4 units): Strives to produce the most accurate swap results without making any concessions. As such, it is also the most resource-intensive and slow.
- `magic_mirror` (3 units): Strikes a good balance between accuracy and speed. If we auto-enabled the CN features, this would be the default preset.
- `quickshot` (1 unit): Produces a decent swap with only a single unit, although it's not as accurate as the other presets.
- `face_doctor` (2 unit): Preset designed for face closeups only.

Alternatively, you can set the preset to `none` and configure your ControlNet units manually.

</details>