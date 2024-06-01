Allows you to declare your own named function (arbitrary code) and execute it with `[call]`.

The first parg is the name of your function, e.g. `[function my_method]` can be referenced later with `[call my_method]`.

Supports the `_const` parg which marks your function as a constant function. By including this argument, another script will not be able to initialize a function by the same name.

Supports "default arguments" by way of arbitrary pargs and kwargs:

- Parg example: `[function my_method my_parg]` will set the user variable `my_parg` to 1 when you `[call my_method]`.
- Kwarg example: `[function my_method my_kwarg=apple]` will set the user variable `my_kwarg` to `apple` when you `[call my_method]`.

Supports the `_required` kwarg which lets you specify one or more variable names delimited by `Unprompted.Config.syntax.delimiter`. If any are not set, the function will be bypassed.

```
[function my_method]
A picture of [random 10] houses.
[/function]

[call my_method]
```
```
POSSIBLE RESULT:
A picture of 5 houses.
```