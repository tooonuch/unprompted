Returns the number of items in a delimited string or `[array]` variable.

Supports the optional `_delimiter` argument which lets you specify the separator between each item. It defaults to your config's `syntax.delimiter` value (`|`).

Supports the optional `_max` argument which caps the value returned by this shortcode. Defaults to -1, which is "no cap."

```
[length "item one|item two|item three"]
```
**Result:** 3
