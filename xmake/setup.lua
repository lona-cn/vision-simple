add_rules("plugin.vsxmake.autoupdate")

includes("plugins/*.lua")
includes("rules/*.lua")
add_repositories(os.scriptdir().."/local-repo".." repo")
includes("options.lua")
includes("modules/*.lua")
includes(path.join(os.projectdir(),"app","xmake.lua"))