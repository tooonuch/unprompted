Returns the content prefixed with the correct English indefinite article, in most cases `a` or `an`.

Supports the optional `definite` positional argument which will instead return the definite article as a prefix, usually `the`.

```
[article]tiger[/article]
```

```
RESULT: a tiger
```

```
[article]apple[/article]
```
```
RESULT: an apple
```