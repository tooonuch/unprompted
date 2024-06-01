Runs a txt2img task inside of an `[after]` block.

The txt2img settings are determined by your user variables. In the following example, we explicitly update the CFG scale and prompt for the task:

```
[after][sets cfg_scale=15 prompt="horse"][txt2img][/after]original prompt goes here
```
