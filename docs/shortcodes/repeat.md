Processes and returns the content a number of `times`.

Supports the optional `_sep` argument which is a string delimiter inserted after each output, excluding the final output. Example: `[repeat 3 _sep="|"]content[/repeat]` will return `content|content|content`.

Supports float values as well. For example, `[repeat 4.2]content[/repeat]` will have a 20% chance to return `content` 5 times instead of 4.

```
[set my_var]0[/set]
[repeat 5]
Variable is currently: [set my_var _out _append]1[/set]
[/repeat]
```