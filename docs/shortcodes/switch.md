Allows you to run different logic blocks with inner case statements that match the value of the given positional argument.

Both `[switch]` and `[case]` support advanced expressions.

Note that `[case]` can be evaluated against multiple pargs. For example, `[case "something" "another_thing"]` will succeed if your switch var is either "something" or "another_thing."

```
[set my_var]100[/set]
[switch my_var]
	[case 1]Does not match[/case]
	[case 2]Does not match[/case]
	[case 100]Matches! This content will be returned[/case]
	[case 4]Does not match[/case]
	[case]If no other case matches, this content will be returned by default[/case]
[/switch]
```