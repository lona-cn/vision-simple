add_rules("plugin.vsxmake.autoupdate")

includes("plugins/*.lua")
includes("rules/*.lua")
add_repositories("local-repo repo", {
    rootdir = os.scriptdir()
})
includes("options.lua")
includes("modules/*.lua")
local project_semver = "0.2.0"
set_version(project_semver, { build = "%Y%m%d%H%M" })
set_allowedplats("windows","linux")
set_allowedarchs("x64","x86_64","aarch64")
includes(path.join(os.projectdir(), "app", "xmake.lua"))
