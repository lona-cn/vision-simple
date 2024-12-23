add_rules("plugin.vsxmake.autoupdate")

includes("plugins/*.lua")
includes("rules/*.lua")
add_repositories("local-repo repo", {
    rootdir = os.scriptdir()
})
includes("options.lua")
includes("modules/*.lua")
includes(path.join(os.projectdir(), "app", "xmake.lua"))
