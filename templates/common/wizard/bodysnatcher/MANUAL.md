<details open><summary>⚠️ Important info, please read carefully</summary>

To achieve compatibility between Unprompted and ControlNet, you must manually rename the `unprompted` extension folder to `_unprompted`. This is due to [a limitation in the Automatic1111 extension framework](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/8011) whereby priority is determined alphabetically. Restart the WebUI after making this change.

</details>

<details><summary>Recommended inference settings</summary>

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

<details><summary>ControlNet Preset Cheatsheet</summary>

Each of the included ControlNet presets was designed with a specific purpose in mind. 

- (NEW!) `vivarium` (1 unit): This is an experimental preset that utilizes the `instructp2p` model, opinionated inference settings, and some files from Civitai for fast, accurate bodyswaps. It is the only preset that does not necessarily benefit from using an inpainting model or `[txt2mask]`.
- `fidelity` (4 units): Strives to produce the most accurate swap results without making any concessions. As such, it is also the most resource-intensive and slow.
- `magic_mirror` (3 units): Strikes a good balance between accuracy and speed. If we auto-enabled the CN features, this would be the default preset.
- `quickshot` (1 unit): Produces a decent swap with only a single unit, although it's not as accurate as the other presets.
- `face_doctor` (2 unit): Preset designed for face closeups only.

Alternatively, you can set the preset to `none` and configure your ControlNet units manually.

</details>