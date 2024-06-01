[for var "test var" "update var"]

Returns the content an arbitrary number of times until the `test` condition returns false.

Importantly, the `test` and `update` arguments must be enclosed in quotes because they are parsed as advanced expressions!

`var` is initialized as a user variable and can be accessed as normal, e.g. `[get var]` is valid (even inside of the loop.)

`var` is optional. For example, `[for "i<10" "i+1"]` is a valid loop, as long as the variable `i` was previously initialized.

The result of the `update` argument is set as the value of `var` at the end of each loop step.

```
[for i=0 "i<10" "i+1"]
Current value of i: [get i]
[/for]
```