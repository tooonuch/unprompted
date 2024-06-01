Allows you to run the `random.seed()` method at will.

It is a more comprehensive operation than `[set seed]`, as it will update several seed-related variables used by the WebUI, including `seed`, `all_seeds` and `seeds`.

The first parg determines the new seed value. If not provided, it will default to the value of `seed` user variable.

```
[seed 100]
```