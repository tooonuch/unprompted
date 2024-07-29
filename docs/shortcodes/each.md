Foreach style loop, similar to Python.

Returns the content `n` times, where `n` is equal to the length of the `list` parg.

## Required pargs

### item

The value of the nth index in the `list` parg.

### list

The list variable to iterate over.

## Optional kwargs

### idx

The variable name to store the iterator counter in. Defaults to `idx`.


## Example

In this example, we'll populate a list variable called `test_array` with `a|b|c`, and then loop through it with `[each]`:

```
[array test_array 0="a" 1="b" 2="c"]
[each item test_array]
	The iteration is: [get idx]
	The value of item is: [get item]
[/each]
```

### Result:

```
The iteration is: 0
The value of item is: a
The iteration is: 1
The value of item is: b
The iteration is: 2
The value of item is: c
```