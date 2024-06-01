Converts the content to the specified datatype.

For security reasons, this shortcode is limited to the following datatypes: `int`, `float`, `str`, `bool`, `list`, `dict`.

Please note that Unprompted is a weakly-typed language, which means that you can generally use a variable as any datatype without casting it. However, this shortcode may be useful when trying to pass an Unprompted variable to an outside function.

```
[cast int]34.7[/cast]
```