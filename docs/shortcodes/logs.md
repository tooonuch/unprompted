Prints one or more messages to the console. This is the atomic version of `[log]`.

By default, the message context is `INFO`. You can change this with the optional `_level` argument.

Each parg is a message to be printed. You should enclose your message in quotes if it contains spaces.

```
[logs "This is a message" "This is another message" _level="INFO"]
```