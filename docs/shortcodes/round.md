Allows you to round the first parg to a certain level of precision.

Supports the optional `_place` int kwarg which determines the level of precision. Defaults to 0.

Supports the optional `_up` parg which will round the number up (ceiling function) instead of normal rounding.

Supports the optional `_down` parg which will round the number down (flooring function) instead of normal rounding.

Supports rounding of both integer and float values.

```
Float example...
[round 1.345 _place=1]
```
```
RESULT: 1.3
```
```
Integer example...
[round 1678 _place=1]
```
```
RESULT: 1680
```