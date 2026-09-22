local target_name = "server"
local kind = "binary"
local group_name = "program"
local pkgs = { "libhv", "turbobase64", "log4cplus" }
local deps = { "runtime", "infer" }
local syslinks = {}
local function callback()
    if is_plat("windows") then
        add_syslinks("mfplat", "mfreadwrite", "mfuuid", "ole32")
    end
    set_basename("vision_simple-server")
    add_extrafiles(path.join(os.projectdir(), "doc", "openapi", "**"))
    add_extrafiles(path.join(os.projectdir(), "app", "config", "base", "**"))
end
CreateTarget(target_name, kind, os.scriptdir(), group_name, pkgs, deps, syslinks, callback)

target("test_subtitle_timeline")
    add_includedirs("private")
    add_files("private/SubtitleTimeline.cpp")
target_end()