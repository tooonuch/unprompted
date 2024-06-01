The atomic version of `[set]` that allows you to set multiple variables at once.

This shortcode processes your arguments with `[set]` directly, meaning you can take advantage of system arguments supported by `[set]`, such as `_new`.

Supports the optional `_load` kwarg for importing key:value pairs from one or more external JSON files.

```
[sets var_a=10 var_b=something var_c=500]
```