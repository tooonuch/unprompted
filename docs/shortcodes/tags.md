Assigns arbitrary tags to the content. Supports both parg and kwarg-style tags.

On its own, this shortcode merely returns the content, but it can be used in conjunction with `[filter_tags]` to bypass the content if the tags don't match your filter rules. See `[filter_tags]` for more information.

```
[tags "tag_one" tag_two="value_two"]A photo of a dog.[/tags]
```