Performs word-by-word spellcheck on the content, replacing any typos it finds with the most likely correction.

Powered by the [pattern](https://github.com/clips/pattern/wiki/pattern-en) library - see pattern docs for more info.

Supports the optional `confidence` argument, which is a float between 0 and 1 that determines how similar the suggested correction must be to the original content. Defaults to 0.85.

```
[autocorrect]speling is vrey dfficult soemtims, okky!!![/autocorrect]
```
```
RESULT: spelling is very difficult sometimes, okay!!!
```