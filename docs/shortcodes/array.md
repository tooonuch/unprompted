Manages a group or list of values.

The first positional argument, `name`, must be a string that corresponds to a variable name for the array. You can later use the same identifier with `[get]` to retrieve every value in the array as a delimited string.

If you want to **retrieve** values at specific indexes, supply the indexes as positional arguments as shown below:

```
[array my_array 2 4 3]
```

If you want to **set** values at specific indexes, supply the indexes as keyword arguments as shown below:

```
[array my_array 2="something" 4=500 3="something else"]
```

You can also use variable names as kwarg values and `[array]` will attempt to parse them for an integer value.

Supports the optional `_delimiter` argument that defines the separator string when retrieving multiple values from the array. Defaults to your `Config.syntax.delimiter` setting.

Supports `_append` which allows you to add values to the end of the array. You can pass multiple values into `_append` with your `_delimiter` string, e.g. `[array my_array _append="something|another thing|third thing"]`.

Similarly, supports `_prepend` which allows you to insert values to the beginning of the array.

Supports `_del` which will remove a value from the array at the specified index, e.g.

```
BEFORE: my_array = 5,7,9,6
```
```
[array my_array _del=1]
```
```
AFTER: my_array = 5,9,6
```

Supports `_remove` which will remove the first matching value from the array, e.g.

```
BEFORE: my_array = 5,7,9,6
```
```
[array my_array _remove=9]
```
```
AFTER: my_array = 5,7,6
```

Supports `_find` which will return the index of the first matching value in the array.

Supports `_shuffle` which will randomize the order of the array.

Supports `_fill` kwarg which will populate the entire array with a given value.

Supports the `_start` and `_end` kwargs, which allow you to retrieve values in a range of indexes from the array. Example:

```
[array my_array 0=a 1=b 2=c 3=d 4=e 5=f 6=g]
[array my_array _start=2 _end=4]
```

**Result:** c,d,e

If you only supply `_start`, then it will retrieve values from that index to the end of the array.

If you only supply `_end`, then it will retrieve values from the start of the array to that index.

If you use `_start` and/or `_end` along with `_fill`, then it will populate that specific range of values with your `_fill` value, as opposed to the entire array.

Supports the `_inclusive` kwarg, which determines whether the `_end` index is inclusive or exclusive. Defaults to 1 (true).

Supports the `_step` kwarg which indicates the step size when processing the `_start` and `_end` range. Defaults to 1.