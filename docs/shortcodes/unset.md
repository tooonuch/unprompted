Removes one or more variables from memory.

Note that variables are automatically deleted at the end of each run - you do **not** need to manually clean memory in most cases. The `[unset]` shortcode is for advanced use.

Supports pattern matching with `*` to delete many variables at once. This may be useful, for example, if you're trying to disable ControlNet inside of an `[after]` block: `[unset cn_* controlnet_*]`.

```
[set var_a=10 var_b="something"]
[unset var_a var_b]
```