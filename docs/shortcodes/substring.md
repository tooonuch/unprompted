Returns a slice of the content as determined by the keyword arguments.

`start` is the beginning of the slice, zero indexed. Defaults to 0.

`end` is the last position of the slice. Defaults to 0.

Alternatively, you can pass a string into `start` or `end` and it will find the index of that string within the `content`.

`step` is the skip interval. Defaults to 1 (in other words, a continuous substring.)

`unit` is either `characters` or `words` and refers to the unit of the aforementioned arguments. Defaults to `characters`.

```
[substring start=1 end=3 unit=words]A photo of a giant dog.[/substring]
```
```
Result: photo of a
```