<details open><summary>0.2.0</summary>

### Added
- New setting `exclude_content` allows you to specify a region to exclude from face restoration via `[txt2mask]`
- New preset `fast_v2`: Uses the `mobilenet0.25_Final` backbone and `GPENO` method to massively speed up inference

### Fixed
- Fixed `make_embedding` preset not clearing cache before running

</details>