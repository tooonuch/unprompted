Updates your Unprompted settings with the content for the duration of a run. Generally you would put this at the top of a template.

Supports inline JSON as well as external JSON files.

Supports relative and absolute filepaths.

If you do not enter a file extension, `.json` is assumed.

```
[config]{"debug":True,"shortcodes":{"choose_delimiter":"*"}}[/config]
```

```
[config]./my_custom_settings[/config]
```