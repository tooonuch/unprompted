Prepare a list of tags which will be evaluated against subsequent `[tags]` blocks. The content of `[tags]` is bypassed if it does not match your filters.

Supports the `_extend` parg to add to the existing list of filters instead of replacing it.

Supports the `_clear` parg to clear all filter rules after the first matching `[tags]` block.

Supports the `_once` parg to remove an individual tag from the filter after the first match.

Supports the `_must_match` kwarg that determines the behavior of the filter when using multiple tags:

- `any` (default): The `[tags]` block must contain at least one matching parg or kwarg.
- `all`: The `[tags]` block must contain all matching pargs and kwargs.
- `selective`: The `[tags]` block must contain the same pargs. It does **not** have to contain the same kwargs, but if it does, the kwarg values must match.

Supports `Config.syntax.not_operator` to exclude tags from the filter. For example, if you want to exclude all blocks with the "outdoors" tag, you can do it like this: `[filter_tags !outdoors]`.

For kwarg tags, the not operator can be used with keys or values as shown:

- `[filter_tags !location="indoors"]` will exclude all blocks that contain a kwarg with the `location` key
- `[filter_tags location="!indoors"]` will exclude all blocks that contain a `location` kwarg with a value of `indoors`

Supports the `_debug` parg to print some diagnostic information to the console.

You can clear the filter at any time by calling `[filter_tags]` without any arguments.

```
[filter_tags location="outdoors"]
[tags location="indoors"]This will not print[/tags]
[tags location="outdoors"]This will print[/tags]
```