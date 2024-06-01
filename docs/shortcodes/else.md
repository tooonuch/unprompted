Returns content if a previous conditional shortcode (e.g. `[if]` or `[chance]`) failed its check, otherwise discards content.

Supports the `id` kwarg. You can assign `_else_id` as a kwarg of the conditional block to associate it with a particular `[else]` block. Match the `id` to the `_else_id`. This means the two blocks don't have to appear next to each other.

Supports the `debug` parg which will print some diagnostic information to the console.

```
[if my_var=0]Print something[/if][else]It turns out my_var did not equal 0.[/else]
```