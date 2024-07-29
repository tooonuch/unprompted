<details open><summary>2.1.1</summary>

### Changed
- Reverted the default `mask_method` to `clipseg`

### Fixed
- Disabled debug mode in the `[txt2mask]` operations

</details>

<details><summary>2.1.0</summary>

### Added
- Support for the new `[txt2mask]` method `panoptic_sam`

### Changed
- The default inference preset is now `general_v3`

### Fixed
- Corrected the `none` mask mode behavior

### Removed
- Less-relevant `[txt2mask]` methods `clip_surgery`, `fastsam`, and `tris`

</details>

<details><summary>2.0.1</summary>

### Changed
- Default `color_correct_method` updated to `mkl`

</details>

<details><summary>2.0.0</summary>

### Added
- New ControlNet preset `xl_quickshot_v1`: Powered by softedge_hed model `controlnetxlCNXL_ecomxlSoftedge`
- Updated img2img preset `full_denoise_v4`: Switched sampler to `Restart` with 10 steps
- Improved the interrogation feature such that it will interrogate the masked subject--rather than the entire canvas--leading to much more accurate prompts

### Changed
- Updated default settings for better SDXL compatibility
- The `bypass_zoom_enhance` option has been inverted to `do_zoom_enhance`

### Fixed
- The `mask_informs_size` option no longer takes effect when `inpaint_full_res` is disabled

</details>

<details><summary>1.5.0 - 25 November 2023</summary>

### Added
- New setting `max_image_size` size to limit the dimensions of the output image
- New setting `mask_informs_size` which uses the mask aspect ratio to determine the inpainting dimensions
- New setting `mask_size_limit` to cap the dimensions of the aforementioned feature
- New setting `mask_padding` to adjust the padding applied by `[txt2mask]`
- Now references `global_subject`, `global_prefix`, and `global_class` for default values
- Now uses the new `vivarium_v3` preset by default
- Minor UI updates

</details>

<details><summary>1.4.4 - 10 November 2023</summary>

### Added
- Minor UI updates

### Changed
- Preset `vivarium_v2`: Adjusted inference settings and enabled `[txt2mask]` support

</details>

<details><summary>1.4.3 - 5 November 2023</summary>

### Added
- Supports `keep_hands` and `keep_feet` even when `mask_mode` is "none"

### Changed
- Sets `inpainting_mask_invert` to true when `mask_mode` is "none"
- Adjusted interrogation syntax

</details>

<details><summary>1.4.2 - 28 October 2023</summary>

### Added
- Added option to disable `[txt2mask]` feature

</details>

<details><summary>1.4.1 - 16 October 2023</summary>

### Changed
- Moved the interrogation result to the back of the prompt

</details>

<details><summary>1.4.0 - 13 October 2023</summary>

### Added
- Optionally interrogate the starting image

</details>

<details><summary>1.3.5 - 13 October 2023</summary>

### Changed
- Updated the default `prefix` from "photo of" to "high detail RAW photo of"
- No longer runs `[img2img_autosize]` when you are on `Only masked` mode
- Now applies 5px of negative mask padding when using the `Keep original hands` option, which can significantly improve blending of new image
- The Zoom Enhance features are now disabled by default, as Facelift is a better fit with Bodysnatcher
- Updated the default `inference_preset` to `subtle_v1`
- Updated documentation
- Updated credits in `README.md`

### Fixed
- Fixed an error that would occur when `Keep hands` was disabled but `Keep feet` was enabled

</details>

<details><summary>1.3.4 - 11 October 2023</summary>

### Changed
- Replaced `[file]` blocks with `[call]`

</details>

<details><summary>1.3.3 - 31 August 2023</summary>

### Fixed
- Now uses mask mode `discard` with `[zoom_enhance]` to ensure compatibility with `[txt2mask]`
- Temporarily switched `[zoom_enhance]` to `_alt` mode as a workaround for ControlNet compatibility issue

</details>

<details><summary>1.3.2 - 28 July 2023</summary>

### Fixed
- Unsets the ControlNet units for `[after]` processing

</details>

<details><summary>1.3.1 - 24 June 2023</summary>

### Added
- Now supports the aforementioned `inherit_negative` feature of `[zoom_enhance]` (true by default)

### Changed
- Improved Wizard GUI

</details>

<details><summary>1.3.0 - 13 May 2023</summary>

### Added
- New setting `inference_preset` that will load settings from the aforementioned directory

### Changed
- Minor UI updates

### Removed
- Removed `use_optimized_inference_settings` in favor of the new `inference_preset` setting

</details>

<details><summary>1.2.0 - 28 April 2023</summary>

### Added
- Now supports `face_controlnet_preset` which is applied during the `[zoom_enhance]` step

### Changed
- Now populates the list of ControlNet presets with files from `templates/common/controlnet_presets`
- Enabled `pixel_perfect` for all ControlNet templates

</details>