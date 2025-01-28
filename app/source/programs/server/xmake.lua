local target_name = "server"
local kind = "binary"
local group_name = "program"
local pkgs = { "libhv", "turbobase64", "log4cplus" }
local deps = { "runtime", "infer" }
local syslinks = {}
local function callback()
    set_basename("vision_simple-server")
    add_extrafiles(path.join(os.projectdir(), "doc", "openapi", "**"))
    add_extrafiles(path.join(os.projectdir(), "app", "config", "base", "**"))
end
CreateTarget(target_name, kind, os.scriptdir(), group_name, pkgs, deps, syslinks, callback)